"""Alternative clustering pipeline that replaces DBSCAN with HDBSCAN.

DBSCAN では epsilon の自動推定に依存していたが、HDBSCAN では
`min_cluster_size` / `min_samples` をクラスタごとに自動調整しつつ、
`cluster_selection_epsilon` を最近傍距離から導出することでタイル内の
密度ゆらぎにもう少し頑健に対応できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import hdbscan


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------
@dataclass
class HDBSCANClusteringConfig:
    """Configuration for the HDBSCAN-based clustering.

    Attributes
    ----------
    min_cluster_size:
        If specified, use this value for every image. Otherwise it is derived
        from the number of annotators that contributed points to the image.
    min_cluster_size_worker_ratio:
        Used only when `min_cluster_size` is None. We multiply the unique worker
        count by this ratio and clamp by `min_cluster_size_floor`.
    min_cluster_size_floor:
        Lower bound when deriving `min_cluster_size` automatically.
    min_samples:
        Same idea as `min_cluster_size`. If None we derive it as
        `min_cluster_size * min_samples_ratio`.
    min_samples_ratio:
        Ratio used for auto `min_samples`.
    eps_scale / eps_min_quantile:
        Parameters for automatic `cluster_selection_epsilon`.
        The local scale is estimated from second-nearest-neighbour distances
        (duplicates removed) and then multiplied by `eps_scale`.
    cluster_selection_epsilon:
        Manual override for HDBSCAN. When None we use the automatic value.
        Set `use_auto_epsilon=False` to skip epsilon entirely.
    cluster_selection_method:
        Forwarded to HDBSCAN. "leaf" tends to keep finer structures.
    metric:
        Distance metric (forwarded).
    noise_as_singletons:
        If True, noise points (`-1`) are converted into singleton clusters so
        downstream処理がそのまま流用できる。
    force_single_cluster_when_small:
        When the image has fewer points than the derived `min_cluster_size`,
        fall back to a single cluster to avoid marking everything as noise.
    """

    min_cluster_size: int | None = None
    min_cluster_size_worker_ratio: float = 0.4
    min_cluster_size_floor: int = 3

    min_samples: int | None = None
    min_samples_ratio: float = 0.6

    eps_scale: float = 1.1
    eps_min_quantile: float = 0.82
    cluster_selection_epsilon: float | None = None
    use_auto_epsilon: bool = True

    cluster_selection_method: str = "leaf"
    metric: str = "euclidean"

    noise_as_singletons: bool = True
    force_single_cluster_when_small: bool = True


@dataclass
class ImageClusteringParams:
    """Per-image parameters actually used during clustering."""

    img_id: Any
    n_points: int
    n_workers: int
    nn_radius: float
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float | None
    n_clusters: int
    n_noise: int


def _ensure_input(path: Path | str | None) -> Path:
    path = Path(path or "")
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    return path


def _ensure_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _estimate_local_radius(
    coords: np.ndarray, *, eps_min_quantile: float = 0.82
) -> float:
    """Robust 2-NN based radius estimation (duplicates removed)."""

    if coords.shape[0] < 3:
        return 1.0

    uniq = np.unique(coords.astype(float), axis=0)
    if uniq.shape[0] < 3:
        return 1.0

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(uniq)
    dists, _ = nn.kneighbors(uniq)
    d = dists[:, 1]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    eps_base = med + 3.0 * mad

    q = float(np.clip(eps_min_quantile, 0.5, 0.95))
    eps_lb = float(np.percentile(d, 100.0 * q))
    eps = max(eps_base if np.isfinite(eps_base) else 0.0, eps_lb)

    if not np.isfinite(eps) or eps <= 0:
        eps = med if (np.isfinite(med) and med > 0) else 1.0

    cap = max(1.0, 2.0 * float(np.percentile(d, 90)))
    eps = float(min(eps, cap))
    return float(max(eps, 1e-6))


def _derive_image_params(
    img_id: Any, group: pd.DataFrame, config: HDBSCANClusteringConfig
) -> tuple[ImageClusteringParams, dict[str, Any]]:
    coords = group[["x", "y"]].to_numpy(float)
    n_points = coords.shape[0]
    n_workers = int(group["worker_id"].nunique()) if "worker_id" in group.columns else 1

    nn_radius = _estimate_local_radius(
        coords, eps_min_quantile=config.eps_min_quantile
    )

    min_cluster_size = config.min_cluster_size
    if min_cluster_size is None:
        min_cluster_size = max(
            config.min_cluster_size_floor,
            int(round(n_workers * config.min_cluster_size_worker_ratio)),
        )

    min_cluster_size = int(max(1, min_cluster_size))

    min_samples = config.min_samples
    if min_samples is None:
        min_samples = int(max(1, round(min_cluster_size * config.min_samples_ratio)))

    min_samples = int(max(1, min(min_samples, n_points)))

    epsilon = config.cluster_selection_epsilon
    if epsilon is None and config.use_auto_epsilon:
        epsilon = float(nn_radius) * float(config.eps_scale)

    params = ImageClusteringParams(
        img_id=img_id,
        n_points=n_points,
        n_workers=n_workers,
        nn_radius=float(nn_radius),
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_epsilon=float(epsilon) if epsilon else None,
        n_clusters=0,
        n_noise=0,
    )

    # kwargs for HDBSCAN instantiation
    kwargs: dict[str, Any] = {
        "min_cluster_size": params.min_cluster_size,
        "min_samples": params.min_samples,
        "metric": config.metric,
        "cluster_selection_method": config.cluster_selection_method,
    }
    if params.cluster_selection_epsilon is not None:
        kwargs["cluster_selection_epsilon"] = params.cluster_selection_epsilon

    return params, kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cluster_annotations_hdbscan(
    config: HDBSCANClusteringConfig,
    *,
    annotations_path: Path | str,
    output_path: Path | str,
    params_output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster annotations per image using HDBSCAN.

    Parameters
    ----------
    config:
        HDBSCANClusteringConfig instance with heuristic defaults.
    annotations_path:
        CSV with at least columns (img_id, point_id, worker_id, x, y).
    output_path:
        Where to store the resulting dataframe (cluster_id / cluster_size added).
    params_output_path:
        Optional CSV summarising per-image hyperparameters actually used.

    Returns
    -------
    cluster_df, params_df
    """

    annotations_path = _ensure_input(annotations_path)
    output_path = Path(output_path)
    _ensure_dir(output_path)

    df = pd.read_csv(annotations_path).sort_values("point_id").reset_index(drop=True)
    required_cols = {"img_id", "point_id", "worker_id", "x", "y"}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"required column(s) missing: {missing_cols}")

    cluster_ids = np.full(len(df), -1, dtype=int)
    cluster_offset = 0

    params_records: list[ImageClusteringParams] = []

    for img_id, group in df.groupby("img_id", sort=False):
        params, kwargs = _derive_image_params(img_id, group, config)
        coords = group[["x", "y"]].to_numpy(float)

        if coords.shape[0] == 0:
            continue

        labels: np.ndarray
        if (
            config.force_single_cluster_when_small
            and coords.shape[0] < params.min_cluster_size
        ):
            labels = np.zeros(coords.shape[0], dtype=int)
        else:
            clusterer = hdbscan.HDBSCAN(**kwargs)
            labels = clusterer.fit_predict(coords).astype(int)

        positive = labels[labels >= 0]
        unique_positive = np.unique(positive) if positive.size else np.array([], dtype=int)
        remap = {int(lbl): int(cluster_offset + i) for i, lbl in enumerate(unique_positive)}
        next_free = int(cluster_offset + len(unique_positive))

        mapped = np.full_like(labels, -1)
        noise_points = 0
        for idx, lbl in enumerate(labels):
            if lbl >= 0:
                mapped[idx] = remap[int(lbl)]
            else:
                noise_points += 1
                if config.noise_as_singletons:
                    mapped[idx] = next_free
                    next_free += 1
                else:
                    mapped[idx] = -1

        cluster_ids[group.index.to_numpy()] = mapped
        cluster_offset = next_free

        params.n_clusters = int(len(unique_positive))
        if config.noise_as_singletons and noise_points:
            params.n_clusters += noise_points
        params.n_noise = int(noise_points)
        params_records.append(params)

    df["cluster_id"] = cluster_ids
    df["cluster_size"] = (
        df.groupby("cluster_id")["point_id"].transform("count").astype(int)
    )

    df.to_csv(output_path, index=False)

    params_df = pd.DataFrame([p.__dict__ for p in params_records])
    if params_output_path:
        params_output_path = Path(params_output_path)
        _ensure_dir(params_output_path)
        params_df.to_csv(params_output_path, index=False)

    return df, params_df

