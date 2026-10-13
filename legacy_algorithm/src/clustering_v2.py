"""Crop-local annotation clustering (v2) using HDBSCAN.

既存の clustering_update.py (HDBSCAN) のロジックを再利用し、
グルーピングの単位を img_id → crop_id に変更したパブリック関数を提供する。
GMM分割は clustering.py のヘルパを再利用する。

出力ファイル:
  - table/clustering_result_v2.csv
  - table/clustering_final_v2.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# HDBSCAN per-image param derivation を再利用
from src.clustering_update import (
    HDBSCANClusteringConfig,
    ImageClusteringParams,
    _ensure_input,
    _ensure_dir,
    _estimate_local_radius,
    _derive_image_params,
)

# GMM 分割用ヘルパを再利用
from src.clustering import (
    GMMSplitConfig,
    _infer_k_by_majority,
    _choose_k_with_bic,
    _split_by_k,
    _enforce_worker_uniqueness,
    _mark_worker_representatives,
)

import hdbscan as _hdbscan


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(".").resolve()
TABLE_DIR = PROJECT_ROOT / "table"
QC_ANNOTATIONS_V2_PATH = TABLE_DIR / "qc_annotations_v2.csv"
CLUSTER_RESULT_V2_PATH = TABLE_DIR / "clustering_result_v2.csv"
CLUSTER_FINAL_V2_PATH = TABLE_DIR / "clustering_final_v2.csv"


# -----------------------------
# Public API
# -----------------------------
def cluster_annotations_by_crop(
    config: HDBSCANClusteringConfig,
    *,
    annotations_path: Path | str | None = None,
    output_path: Path | str | None = None,
    params_output_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster annotations per **crop** using HDBSCAN -> `cluster_id`.

    v1 (cluster_annotations_hdbscan) との違い:
      - groupby("img_id") ではなく groupby("crop_id") でクラスタリング
      - 座標 (x, y) はクロップ内ローカル座標を前提
    """
    annotations_path = _ensure_input(annotations_path or QC_ANNOTATIONS_V2_PATH)
    output_path = Path(output_path) if output_path else CLUSTER_RESULT_V2_PATH
    _ensure_dir(output_path)

    df = pd.read_csv(annotations_path).sort_values("point_id").reset_index(drop=True)
    required_cols = {"crop_id", "point_id", "worker_id", "x", "y"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"required column(s) missing: {', '.join(sorted(missing))}")

    cluster_ids = np.full(len(df), -1, dtype=int)
    cluster_offset = 0
    params_records: list[ImageClusteringParams] = []

    for crop_id, group in df.groupby("crop_id", sort=False):
        # _derive_image_params は img_id を受け取るが、crop_id でも動く
        params, kwargs = _derive_image_params(crop_id, group, config)
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
            clusterer = _hdbscan.HDBSCAN(**kwargs)
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


def refine_clusters_with_gmm_by_crop(
    cluster_data: pd.DataFrame | Path | str,
    config: GMMSplitConfig,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Split touching cells inside each `cluster_id` using majority-inferred K.

    ロジックは既存の refine_clusters_with_gmm と同一。
    """
    if isinstance(cluster_data, pd.DataFrame):
        df = cluster_data.copy()
    else:
        df = pd.read_csv(Path(cluster_data))

    if "cluster_id" not in df.columns:
        raise ValueError("cluster_id column is required.")
    for col in ("point_id", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"required column missing: {col}")

    output_path = Path(output_path) if output_path else CLUSTER_FINAL_V2_PATH
    _ensure_dir(output_path)

    final_ids = np.full(len(df), -1, dtype=int)
    next_id = 0

    for cluster_id, group in df.groupby("cluster_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) == 0:
            continue
        if int(cluster_id) < 0:
            for idx in indices:
                final_ids[idx] = next_id
                next_id += 1
            continue

        subset = df.loc[indices]
        k_vote = _infer_k_by_majority(subset, config)

        coords = subset[["x", "y"]].to_numpy(float)
        if bool(getattr(config, "use_bic_guard", True)) and k_vote > 1:
            k = _choose_k_with_bic(coords, k_vote, config)
        else:
            k = int(k_vote)

        labels = _split_by_k(coords, k, config)

        for comp in range(int(labels.max()) + 1):
            mask = labels == comp
            comp_indices = indices[mask]
            if comp_indices.size == 0:
                continue
            final_ids[comp_indices] = next_id
            next_id += 1

    df["final_cluster_id"] = final_ids

    if bool(getattr(config, "enforce_worker_uniqueness", False)):
        df = _enforce_worker_uniqueness(df, config)

    df["final_cluster_size"] = (
        df.groupby("final_cluster_id")["point_id"].transform("count").astype(int)
    )

    if not config.allow_duplicate_workers:
        df = _mark_worker_representatives(df, "final_cluster_id", "is_worker_representative")
    else:
        df["is_worker_representative"] = True

    df.to_csv(output_path, index=False)
    return df


__all__ = [
    "cluster_annotations_by_crop",
    "refine_clusters_with_gmm_by_crop",
]
