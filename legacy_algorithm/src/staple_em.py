"""STAPLE-style EM estimation utilities.

`set` ドキュメントの数式 (π, p_j, q_j, LLR 変換) に沿って、
クラスタ×ワーカの二値観測から EM で感度/特異度を推定し、
重み (log-likelihood ratio) を出力する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass
class AnnotationMatrix:
    """Binary observation matrix `d_{ij}`."""

    matrix: np.ndarray  # shape = (n_clusters, n_workers)
    cluster_ids: np.ndarray
    worker_ids: np.ndarray
    cluster_sizes: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_workers(self) -> int:
        return int(self.matrix.shape[1])


@dataclass
class EMConfig:
    max_iter: int = 256
    tol: float = 1e-6
    min_prob: float = 1e-4
    init_pi: float | None = None
    init_sensitivity: float | None = None
    init_specificity: float | None = None
    default_specificity: float = 0.95
    
    # ベイズ的正則化（Beta事前分布相当の疑似カウント）
    # 特異度が極端に高くなるのを防ぐ
    regularize_specificity: bool = True
    prior_specificity: float = 0.75  # 事前の特異度期待値
    pseudo_count_specificity: float = 10.0  # 疑似サンプル数（大きいほど事前分布が強い）
    
    # 感度にも正則化を適用する場合
    regularize_sensitivity: bool = False
    prior_sensitivity: float = 0.5
    pseudo_count_sensitivity: float = 5.0


@dataclass
class EMResult:
    posterior: pd.Series  # w_i = P(t_i=1 | d_i)
    sensitivities: pd.Series  # p_j
    specificities: pd.Series  # q_j
    omega_positive: pd.Series
    omega_negative: pd.Series
    pi: float
    log_likelihood: list[float]
    iterations: int
    converged: bool

    def worker_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "worker_id": self.sensitivities.index.astype(str),
                "sensitivity": self.sensitivities.values,
                "specificity": self.specificities.values,
                "weight_positive": self.omega_positive.values,
                "weight_negative": self.omega_negative.values,
            }
        )
        return df.set_index("worker_id")


def build_annotation_matrix_from_clusters(
    cluster_df: pd.DataFrame,
    *,
    cluster_col: str = "final_cluster_id",
    worker_col: str = "worker_id",
    representative_col: str | None = "is_worker_representative",
    drop_noise: bool = True,
) -> AnnotationMatrix:
    """Convert clustered annotations into d_{ij} matrix."""

    df = cluster_df.copy()
    if drop_noise:
        df = df[df[cluster_col] >= 0]
    if df.empty:
        raise ValueError("No valid clusters to build annotation matrix.")

    if representative_col and representative_col in df.columns:
        df = df[df[representative_col]]
    else:
        df = df.drop_duplicates([cluster_col, worker_col])

    binary = (
        df.assign(value=1)
        .drop_duplicates([cluster_col, worker_col])
        .pivot(index=cluster_col, columns=worker_col, values="value")
        .fillna(0.0)
        .astype(np.float64)
    )

    cluster_sizes = (
        cluster_df.groupby(cluster_col)["point_id"].count().reindex(binary.index).to_numpy()
    )

    return AnnotationMatrix(
        matrix=binary.to_numpy(),
        cluster_ids=binary.index.to_numpy(),
        worker_ids=binary.columns.to_numpy(),
        cluster_sizes=cluster_sizes,
    )


def _clip_prob(values: np.ndarray, min_prob: float) -> np.ndarray:
    return np.clip(values, min_prob, 1.0 - min_prob)


def _init_sensitivity(
    D: np.ndarray, override: float | None, min_prob: float
) -> np.ndarray:
    if override is not None:
        value = _clip_prob(np.array([override]), min_prob)[0]
        return np.full(D.shape[1], value, dtype=float)
    count = D.shape[0]
    base = (D.sum(axis=0) + 1.0) / (count + 2.0)
    return _clip_prob(base, min_prob)


def _init_specificity(
    D: np.ndarray, override: float | None, default_value: float, min_prob: float
) -> np.ndarray:
    if override is not None:
        value = _clip_prob(np.array([override]), min_prob)[0]
    else:
        value = _clip_prob(np.array([default_value]), min_prob)[0]
    return np.full(D.shape[1], value, dtype=float)


def run_staple_em(matrix: AnnotationMatrix, config: EMConfig | None = None) -> EMResult:
    """Run the EM loop described in `set` and return calibrated parameters."""

    config = config or EMConfig()
    D = matrix.matrix.astype(np.float64)
    n_clusters, n_workers = D.shape
    if n_clusters == 0 or n_workers == 0:
        raise ValueError("Annotation matrix must be non-empty.")

    min_prob = max(config.min_prob, EPS)
    global_mean = float(D.mean()) if D.size else 0.5
    pi = config.init_pi if config.init_pi is not None else np.clip(global_mean, 0.05, 0.95)
    pi = float(_clip_prob(np.array([pi]), min_prob)[0])

    p = _init_sensitivity(D, override=config.init_sensitivity, min_prob=min_prob)
    q = _init_specificity(
        D,
        override=config.init_specificity,
        default_value=config.default_specificity,
        min_prob=min_prob,
    )

    log_history: list[float] = []
    converged = False

    for iteration in range(1, config.max_iter + 1):
        # E-step
        log_pi = np.log(np.clip(pi, min_prob, 1.0 - min_prob))
        log_1_minus_pi = np.log(np.clip(1.0 - pi, min_prob, 1.0 - min_prob))

        log_p = np.log(_clip_prob(p, min_prob))
        log_1_minus_p = np.log(_clip_prob(1.0 - p, min_prob))
        log_q = np.log(_clip_prob(q, min_prob))
        log_1_minus_q = np.log(_clip_prob(1.0 - q, min_prob))

        ll_t1 = log_pi + D @ log_p + (1.0 - D) @ log_1_minus_p
        ll_t0 = log_1_minus_pi + D @ log_1_minus_q + (1.0 - D) @ log_q

        max_ll = np.maximum(ll_t1, ll_t0)
        exp_t1 = np.exp(ll_t1 - max_ll)
        exp_t0 = np.exp(ll_t0 - max_ll)
        w = exp_t1 / (exp_t1 + exp_t0)

        log_likelihood = float(np.sum(max_ll + np.log(exp_t1 + exp_t0)))
        log_history.append(log_likelihood)

        # M-step
        w_sum = float(w.sum())
        pi_new = np.clip(w_sum / n_clusters, min_prob, 1.0 - min_prob)

        # Sensitivity update（ベイズ正則化オプション付き）
        if config.regularize_sensitivity:
            pseudo_p = float(config.pseudo_count_sensitivity)
            prior_p = float(config.prior_sensitivity)
            denom_p = w_sum + pseudo_p
            numer_p = (w[:, None] * D).sum(axis=0) + pseudo_p * prior_p
        else:
            denom_p = max(w_sum, min_prob)
            numer_p = (w[:, None] * D).sum(axis=0)
        p_new = _clip_prob(numer_p / max(denom_p, min_prob), min_prob)

        # Specificity update（ベイズ正則化オプション付き）
        one_minus_w_sum = float((1.0 - w).sum())
        if config.regularize_specificity:
            pseudo_q = float(config.pseudo_count_specificity)
            prior_q = float(config.prior_specificity)
            denom_q = one_minus_w_sum + pseudo_q
            numer_q = ((1.0 - w)[:, None] * (1.0 - D)).sum(axis=0) + pseudo_q * prior_q
        else:
            denom_q = max(one_minus_w_sum, min_prob)
            numer_q = ((1.0 - w)[:, None] * (1.0 - D)).sum(axis=0)
        q_new = _clip_prob(numer_q / max(denom_q, min_prob), min_prob)

        delta = max(
            abs(pi_new - pi),
            float(np.max(np.abs(p_new - p))) if p_new.size else 0.0,
            float(np.max(np.abs(q_new - q))) if q_new.size else 0.0,
        )

        pi, p, q = pi_new, p_new, q_new

        if delta < config.tol:
            converged = True
            iterations = iteration
            break
    else:
        iterations = config.max_iter

    worker_index = pd.Index(matrix.worker_ids, name="worker_id")
    cluster_index = pd.Index(matrix.cluster_ids, name="final_cluster_id")

    omega_pos = np.log(_clip_prob(p, min_prob)) - np.log(_clip_prob(1.0 - q, min_prob))
    omega_neg = np.log(_clip_prob(1.0 - p, min_prob)) - np.log(_clip_prob(q, min_prob))

    return EMResult(
        posterior=pd.Series(w, index=cluster_index),
        sensitivities=pd.Series(p, index=worker_index),
        specificities=pd.Series(q, index=worker_index),
        omega_positive=pd.Series(omega_pos, index=worker_index),
        omega_negative=pd.Series(omega_neg, index=worker_index),
        pi=float(pi),
        log_likelihood=log_history,
        iterations=iterations,
        converged=converged,
    )


__all__ = [
    "AnnotationMatrix",
    "EMConfig",
    "EMResult",
    "build_annotation_matrix_from_clusters",
    "run_staple_em",
]

