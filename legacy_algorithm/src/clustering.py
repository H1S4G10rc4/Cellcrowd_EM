"""Annotation clustering utilities.

Goals (design constraints)
- Clicks that refer to the same biological cell should end up in the same cluster.
- If multiple cells are fused/overlapping (touching cells), we split *at the level of cell count*,
  using the *majority (mode)* of "how many cells" workers implicitly annotated in that region.
- Keep hyperparameters minimal; infer spatial scales from the data whenever possible.
- Never use recursion for splitting (avoid RecursionError / runaway splitting).

Expected input CSV columns (at minimum)
- img_id, point_id, worker_id, x, y

Outputs
- clustering_result.csv: cluster_id, cluster_size (plus original columns)
- clustering_final.csv: final_cluster_id, final_cluster_size, is_worker_representative (plus original columns)

Note
- This module intentionally keeps all points. If allow_duplicate_workers=False, we mark one representative
  point per (final_cluster_id, worker_id) via `is_worker_representative` for downstream EM/STAPLE-like steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Iterable, Any

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(".").resolve()
TABLE_DIR = PROJECT_ROOT / "table"
QC_ANNOTATIONS_PATH = TABLE_DIR / "qc_annotations.csv"
CLUSTER_RESULT_PATH = TABLE_DIR / "clustering_result.csv"
CLUSTER_FINAL_PATH = TABLE_DIR / "clustering_final.csv"


# -----------------------------
# Configs
# -----------------------------
@dataclass
class ClusteringConfig:
    """Coarse clustering config (per image).

    eps:
        If None or <= 0, estimate an epsilon (pixel radius) from nearest-neighbor distances.
    min_samples:
        DBSCAN min_samples. Use 1 to behave like "epsilon graph connected components" (no noise).
        If >1, points labeled as noise (-1) are converted into singleton clusters.
    """
    eps: float | None = None
    min_samples: int = 1
    metric: str = "euclidean"

    # Heuristic tuning (optional)
    eps_scale: float = 1.0
    # Ensure eps is at least this quantile of NN distances (helps when duplicates make eps too small)
    eps_min_quantile: float = 0.75
    # If False, keep DBSCAN noise as -1 instead of turning each noise point into a singleton cluster
    noise_as_singletons: bool = True


@dataclass
class GMMSplitConfig:
    """Touching-cells splitting config (per coarse cluster).

    max_components:
        Upper bound for the number of cells to split into within a fused region.
        The actual K is inferred from majority vote of per-worker counts and then clipped.
    allow_duplicate_workers:
        If False, we still keep all points but mark a representative per worker per final cluster.
    random_state:
        For reproducibility (GMM + fallback KMeans).
    """
    max_components: int = 5
    allow_duplicate_workers: bool = True
    random_state: int = 42

    # GMM numerical stability (kept as constants; rarely needs tuning)
    covariance_type: str = "full"
    n_init: int = 5
    max_iter: int = 300
    reg_covar: float = 1e-3

    # Split guard: avoid over-splitting when majority vote is noisy
    use_bic_guard: bool = True
    bic_improvement_threshold: float = 10.0  # keep k=1 unless BIC improves by at least this amount
    # Dedupe radius scale for per-worker count inference
    dedupe_eps_scale: float = 0.9

    # Post-processing: enforce that each final cluster contains at most one
    # annotation per worker by re-splitting problematic coarse clusters.
    enforce_worker_uniqueness: bool = True
    worker_uniqueness_max_components: int | None = None
    force_worker_singletons_on_failure: bool = True


# -----------------------------
# Helpers
# -----------------------------
def _ensure_input(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    return path


def _ensure_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _estimate_eps_from_nn(coords: np.ndarray, *, eps_min_quantile: float = 0.75) -> float:
    """Estimate a reasonable epsilon from nearest-neighbor distances (robust).

    Why this exists:
    - If you have many exact/near duplicates (double-clicks), naive NN statistics can become
      artificially tiny, causing massive over-fragmentation. We therefore compute NN distances
      on *unique* coordinates.
    - We also enforce a lower bound based on a high-ish quantile of NN distances to avoid
      eps being dominated by the smallest mode.

    Heuristic:
    - Compute 2-NN distances (nearest neighbor excluding self) on unique coords
    - eps_base = median(d) + 3*MAD(d)
    - eps_lb   = quantile(d, eps_min_quantile)
    - eps = max(eps_base, eps_lb)
    - Cap eps to avoid being dominated by a small number of huge gaps
    """
    if coords.shape[0] < 3:
        return 1.0

    # Drop exact duplicates (common with double-clicks).
    uniq = np.unique(coords.astype(float), axis=0)
    if uniq.shape[0] < 3:
        return 1.0

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(uniq)
    dists, _ = nn.kneighbors(uniq)
    d = dists[:, 1]  # nearest neighbor
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    eps_base = med + 3.0 * mad

    # Lower bound: avoid eps collapsing to the smallest mode.
    q = float(np.clip(eps_min_quantile, 0.5, 0.95))
    eps_lb = float(np.percentile(d, 100.0 * q))

    eps = max(eps_base if np.isfinite(eps_base) else 0.0, eps_lb)

    if not np.isfinite(eps) or eps <= 0:
        eps = med if (np.isfinite(med) and med > 0) else 1.0

    # cap to reduce extreme eps caused by a small number of huge gaps
    q90 = float(np.percentile(d, 90))
    cap = max(1.0, 2.0 * q90)
    eps = float(min(eps, cap))

    return float(max(eps, 1e-6))


def _mode_int(values: Iterable[int]) -> int:
    """Robust mode for small integers. If tie, pick the smaller one."""
    vals = [int(v) for v in values if v is not None]
    if not vals:
        return 1
    uniq, counts = np.unique(vals, return_counts=True)
    maxc = counts.max()
    candidates = uniq[counts == maxc]
    return int(candidates.min())


def _infer_k_by_majority(subset: pd.DataFrame, config: GMMSplitConfig) -> int:
    """Infer number of cells in this coarse cluster from majority vote across workers.

    We try to be robust to accidental double-clicks by the same worker:
    - Within each worker, we first merge *very close* points (dedupe) using a small radius derived
      from the subset's nearest-neighbor scale (no extra hyperparameters).
    """
    if "worker_id" not in subset.columns:
        return 1

    n_workers = int(subset["worker_id"].nunique())
    if n_workers < 2:
        # Cannot define "majority" with a single worker.
        return 1

    coords_all = subset[["x", "y"]].to_numpy(float)
    eps0 = _estimate_eps_from_nn(coords_all)
    # Use a conservative radius to merge double-clicks from the same worker.
    dedupe_eps = max(1.0, float(eps0) * float(config.dedupe_eps_scale))

    worker_counts: list[int] = []
    for _, gw in subset.groupby("worker_id", sort=False):
        pts = gw[["x", "y"]].to_numpy(float)
        if pts.shape[0] <= 1:
            worker_counts.append(int(pts.shape[0]))
            continue
        labels = DBSCAN(eps=float(dedupe_eps), min_samples=1).fit_predict(pts)
        worker_counts.append(int(len(np.unique(labels))))

    if not worker_counts:
        return 1

    k = _mode_int(worker_counts)

    # Clip to feasible range
    k = int(np.clip(k, 1, config.max_components))
    k = min(k, int(len(subset)))  # cannot exceed number of points
    return max(1, k)


def _fit_gmm(coords: np.ndarray, n_components: int, config: GMMSplitConfig) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=config.covariance_type,
        n_init=int(config.n_init),
        random_state=int(config.random_state),
        max_iter=int(config.max_iter),
        reg_covar=float(config.reg_covar),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gmm.fit(coords)
    return gmm


def _choose_k_with_bic(coords: np.ndarray, k_max: int, config: GMMSplitConfig) -> int:
    """Choose number of components by BIC, with a conservative improvement guard.

    Returns:
        k in [1, k_max] (never 0).
    """
    k_max = int(max(1, min(int(k_max), int(coords.shape[0]))))
    if k_max <= 1 or coords.shape[0] < 3:
        return 1

    bics: list[tuple[int, float]] = []
    for k in range(1, k_max + 1):
        try:
            gmm = _fit_gmm(coords, k, config)
            bic = float(gmm.bic(coords))
            if np.isfinite(bic):
                bics.append((k, bic))
        except Exception:
            continue

    if not bics:
        return 1

    # Lower BIC is better.
    bics.sort(key=lambda t: t[1])
    k_best, bic_best = bics[0]

    # Conservative guard: only split if improvement is meaningful vs k=1.
    bic1 = next((b for k, b in bics if k == 1), None)
    if bic1 is None:
        return int(k_best)

    improvement = float(bic1 - bic_best)  # positive means k_best is better
    if int(k_best) > 1 and improvement < float(config.bic_improvement_threshold):
        return 1

    return int(k_best)

def _split_by_k(coords: np.ndarray, k: int, config: GMMSplitConfig) -> np.ndarray:
    """Return labels in [0, k-1] for each point."""
    if k <= 1 or coords.shape[0] <= 1:
        return np.zeros(coords.shape[0], dtype=int)

    k = min(k, coords.shape[0])

    try:
        gmm = _fit_gmm(coords, k, config)
        labels = gmm.predict(coords)
        if labels.shape[0] != coords.shape[0]:
            raise ValueError("GMM returned invalid labels.")
        return labels.astype(int)
    except Exception:
        # Fallback: KMeans is very stable for splitting into a given k.
        km = KMeans(n_clusters=k, n_init=10, random_state=int(config.random_state))
        labels = km.fit_predict(coords)
        return labels.astype(int)


def _labels_satisfy_worker_uniqueness(labels: np.ndarray, workers: np.ndarray) -> bool:
    if workers.size == 0 or labels.size == 0:
        return True
    workers = workers.astype(object)
    for comp in np.unique(labels):
        mask = labels == comp
        if mask.sum() <= 1:
            continue
        subset = workers[mask]
        unique_workers, counts = np.unique(subset, return_counts=True)
        if np.any(counts > 1):
            return False
    return True


def _split_with_worker_constraint(
    coords: np.ndarray,
    workers: np.ndarray,
    k_start: int,
    k_cap: int,
    config: GMMSplitConfig,
) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.zeros(coords.shape[0], dtype=int)

    k_start = max(1, min(k_start, coords.shape[0]))
    k_cap = max(k_start, min(max(1, k_cap), coords.shape[0]))

    last_labels = np.zeros(coords.shape[0], dtype=int)
    for k in range(k_start, k_cap + 1):
        labels = _split_by_k(coords, k, config)
        if _labels_satisfy_worker_uniqueness(labels, workers):
            return labels
        last_labels = labels
    return last_labels


def _resplit_parent_cluster_with_worker_constraint(
    df: pd.DataFrame,
    parent_cluster_id: int,
    config: GMMSplitConfig,
    next_final_id: int,
) -> tuple[pd.DataFrame, int]:
    subset = df[df["cluster_id"] == parent_cluster_id].copy()
    if subset.empty:
        return df, next_final_id

    coords = subset[["x", "y"]].to_numpy(float)
    workers = subset["worker_id"].to_numpy()

    base_k = _infer_k_by_majority(subset, config)
    uniq_workers = max(1, subset["worker_id"].nunique())
    k_min = int(np.ceil(len(subset) / float(uniq_workers))) if uniq_workers else 1
    current_components = max(1, subset["final_cluster_id"].nunique())
    k_start = max(1, base_k, k_min, current_components)

    max_components = config.worker_uniqueness_max_components
    if max_components is None:
        max_components = max(config.max_components, uniq_workers)
    k_cap = int(min(len(subset), max(1, max_components)))
    if k_cap < k_start:
        k_start = k_cap

    labels = _split_with_worker_constraint(coords, workers, k_start, k_cap, config)

    for comp in np.unique(labels):
        mask = labels == comp
        comp_indices = subset.index[mask]
        if comp_indices.size == 0:
            continue
        df.loc[comp_indices, "final_cluster_id"] = next_final_id
        next_final_id += 1

    return df, next_final_id


def _enforce_worker_uniqueness(df: pd.DataFrame, config: GMMSplitConfig) -> pd.DataFrame:
    if "worker_id" not in df.columns or "final_cluster_id" not in df.columns:
        return df

    def _find_problematic_clusters(frame: pd.DataFrame) -> tuple[dict[int, list[str]], set[int]]:
        dup_workers_by_cluster: dict[int, list[str]] = {}
        parent_ids: set[int] = set()
        for final_id, group in frame[frame["final_cluster_id"] >= 0].groupby("final_cluster_id", sort=False):
            worker_counts = group["worker_id"].value_counts()
            dup_workers = worker_counts[worker_counts > 1]
            if dup_workers.empty:
                continue
            dup_workers_by_cluster[int(final_id)] = [str(w) for w in dup_workers.index.tolist()]
            parent_id = int(group["cluster_id"].iloc[0])
            if parent_id >= 0:
                parent_ids.add(parent_id)
        return dup_workers_by_cluster, parent_ids

    dup_info, problematic_parents = _find_problematic_clusters(df)

    if not problematic_parents:
        return df

    result = df.copy()
    next_final_id = int(result["final_cluster_id"].max()) + 1

    for parent_id in problematic_parents:
        result, next_final_id = _resplit_parent_cluster_with_worker_constraint(
            result, parent_id, config, next_final_id
        )

    dup_info, _ = _find_problematic_clusters(result)
    if dup_info and bool(getattr(config, "force_worker_singletons_on_failure", False)):
        for final_id, workers in dup_info.items():
            group = result[result["final_cluster_id"] == final_id]
            for worker in workers:
                worker_rows = group[group["worker_id"].astype(str) == worker]
                if worker_rows.shape[0] <= 1:
                    continue
                keep_idx = worker_rows.index[0]
                for idx in worker_rows.index[1:]:
                    result.at[idx, "final_cluster_id"] = next_final_id
                    next_final_id += 1

    result["final_cluster_size"] = (
        result.groupby("final_cluster_id")["point_id"].transform("count").astype(int)
    )
    return result


def _mark_worker_representatives(df: pd.DataFrame, cluster_col: str, rep_col: str) -> pd.DataFrame:
    """Mark one representative per (cluster, worker) by closest-to-centroid."""
    df = df.copy()
    df[rep_col] = True

    if "worker_id" not in df.columns:
        return df

    reps = np.ones(len(df), dtype=bool)
    for cid, cluster in df.groupby(cluster_col, sort=False):
        if cid < 0:
            continue
        coords = cluster[["x", "y"]].to_numpy(float)
        centroid = coords.mean(axis=0) if coords.size else np.array([0.0, 0.0])

        for _, gw in cluster.groupby("worker_id", sort=False):
            if len(gw) <= 1:
                continue
            pts = gw[["x", "y"]].to_numpy(float)
            dists = np.linalg.norm(pts - centroid, axis=1)
            keep_index = gw.index[int(np.argmin(dists))]
            for idx in gw.index:
                reps[idx] = False
            reps[keep_index] = True

    df[rep_col] = reps
    return df


# -----------------------------
# Public API
# -----------------------------
def cluster_annotations(
    config: ClusteringConfig,
    annotations_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Coarse clustering (per image) -> `cluster_id`.

    - If config.eps is None/<=0: estimate eps per image from NN distances.
    - If DBSCAN labels noise (-1): convert each noise point into its own singleton cluster.
    """
    annotations_path = _ensure_input(annotations_path or QC_ANNOTATIONS_PATH)
    output_path = Path(output_path) if output_path else CLUSTER_RESULT_PATH
    _ensure_dir(output_path)

    df = pd.read_csv(annotations_path).sort_values("point_id").reset_index(drop=True)
    if df.empty:
        raise ValueError("annotations table is empty.")
    for col in ("img_id", "point_id", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"required column missing: {col}")

    cluster_ids = np.full(len(df), -1, dtype=int)
    cluster_offset = 0

    for _, group in df.groupby("img_id", sort=False):
        coords = group[["x", "y"]].to_numpy(dtype=float)
        if coords.shape[0] == 0:
            continue

        eps_img = config.eps if (config.eps is not None and config.eps > 0) else _estimate_eps_from_nn(coords, eps_min_quantile=float(getattr(config, 'eps_min_quantile', 0.75)))

        eps_img = float(eps_img) * float(getattr(config, 'eps_scale', 1.0))
        eps_img = float(max(eps_img, 1e-6))

        # DBSCAN
        if coords.shape[0] >= max(1, int(config.min_samples)):
            dbscan = DBSCAN(eps=float(eps_img), min_samples=int(config.min_samples), metric=str(config.metric))
            labels = dbscan.fit_predict(coords).astype(int)
        else:
            labels = np.full(coords.shape[0], -1, dtype=int)

        # Remap positive labels to global ids
        positive = labels[labels >= 0]
        unique_positive = np.unique(positive) if positive.size else np.array([], dtype=int)
        remap = {int(lbl): int(cluster_offset + i) for i, lbl in enumerate(unique_positive)}
        next_free = int(cluster_offset + len(unique_positive))

        mapped = np.full_like(labels, -1)
        for i, lbl in enumerate(labels):
            if lbl >= 0:
                mapped[i] = remap[int(lbl)]
            else:
                if bool(getattr(config, 'noise_as_singletons', True)):
                    # Convert noise to singleton cluster
                    mapped[i] = next_free
                    next_free += 1
                else:
                    mapped[i] = -1

        cluster_ids[group.index.to_numpy()] = mapped
        cluster_offset = next_free

    df["cluster_id"] = cluster_ids
    df["cluster_size"] = df.groupby("cluster_id")["point_id"].transform("count").astype(int)

    df.to_csv(output_path, index=False)
    return df


def _load_cluster_frame(data: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.read_csv(Path(data))


def refine_clusters_with_gmm(
    cluster_data: pd.DataFrame | Path | str,
    config: GMMSplitConfig,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Split touching cells inside each `cluster_id` using majority-inferred K.

    Strategy (no recursion):
    - For each coarse cluster:
        - infer K_vote = majority of per-worker click counts (deduped, clipped by max_components)
        - (optional) choose K <= K_vote via BIC, and avoid splitting if improvement is weak
        - split into K with GMM (fallback to KMeans)
    - Mark `is_worker_representative` if allow_duplicate_workers is False.
    """
    df = _load_cluster_frame(cluster_data).copy()
    if "cluster_id" not in df.columns:
        raise ValueError("cluster_id column is required.")
    if "point_id" not in df.columns:
        raise ValueError("point_id column is required.")
    for col in ("x", "y"):
        if col not in df.columns:
            raise ValueError(f"required column missing: {col}")

    output_path = Path(output_path) if output_path else CLUSTER_FINAL_PATH
    _ensure_dir(output_path)

    final_ids = np.full(len(df), -1, dtype=int)
    next_id = 0

    # Non-recursive: loop clusters, split once by inferred K.
    for cluster_id, group in df.groupby("cluster_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) == 0:
            continue
        if int(cluster_id) < 0:
            # keep as-is but still assign unique final id(s)
            for idx in indices:
                final_ids[idx] = next_id
                next_id += 1
            continue

        subset = df.loc[indices]
        k_vote = _infer_k_by_majority(subset, config)

        coords = subset[["x", "y"]].to_numpy(float)
        if bool(getattr(config, 'use_bic_guard', True)) and k_vote > 1:
            k = _choose_k_with_bic(coords, k_vote, config)
        else:
            k = int(k_vote)

        labels = _split_by_k(coords, k, config)

        # create one final id per component
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

    df["final_cluster_size"] = df.groupby("final_cluster_id")["point_id"].transform("count").astype(int)

    # Representative marking for downstream worker-unique models (optional)
    if not config.allow_duplicate_workers:
        df = _mark_worker_representatives(df, "final_cluster_id", "is_worker_representative")
    else:
        df["is_worker_representative"] = True

    df.to_csv(output_path, index=False)
    return df


# -----------------------------
# Visualization utilities
# -----------------------------
def resolve_image_path(
    img_id_value: int | str | Path,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Infer raw image path from img_id.

    Supports:
    - already a Path
    - strings like 'QC_000123' -> 'QC_000123.png'
    - numeric ids -> 'QC_{id:06d}.png'
    """
    base_dir = Path(base_dir) if base_dir else PROJECT_ROOT / "data" / "raw"
    if isinstance(img_id_value, Path):
        filename = img_id_value.name
    else:
        img_str = str(img_id_value)
        if img_str.lower().endswith(".png"):
            filename = img_str
        elif img_str.startswith("QC"):
            filename = f"{img_str}.png"
        elif img_str.isdigit():
            filename = f"QC{int(img_str)}.png"
        else:
            filename = f"{img_str}.png"
    return (base_dir / filename).resolve()


def overlay_clusters(
    image: np.ndarray | str | Path | Image.Image,
    df: pd.DataFrame,
    *,
    img_id: int | str,
    cluster_column: str = "cluster_id",
    figsize: tuple[int, int] = (8, 8),
    max_points: int | None = None,
) -> plt.Figure:
    """Overlay clustered points on an image and return the matplotlib Figure."""

    subset = df[df["img_id"] == img_id]
    if subset.empty:
        raise ValueError(f"No points for img_id={img_id}")

    if max_points is not None and len(subset) > max_points:
        subset = subset.sample(max_points, random_state=0)

    cluster_labels = subset[cluster_column].to_numpy(int)
    uniq = np.unique(cluster_labels)

    # simple deterministic color map
    rng = np.random.default_rng(0)
    color_lookup = {int(cid): tuple(rng.random(4)) for cid in uniq}
    colors = [color_lookup[int(cid)] for cid in cluster_labels]

    if isinstance(image, (str, Path)):
        image_array = np.asarray(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        image_array = np.asarray(image.convert("RGB"))
    else:
        image_array = np.asarray(image)
        if image_array.dtype == object:
            raise TypeError("Image array must be numeric; got dtype=object.")

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(image_array)
    ax.scatter(
        subset["x"],
        subset["y"],
        c=colors,
        s=6,
        edgecolors="k",
        linewidths=0.2,
        alpha=0.85,
    )
    ax.set_title(f"img_id={img_id} / {cluster_column}")
    ax.axis("off")
    return fig


def build_cluster_slider_widget(
    df: pd.DataFrame,
    *,
    img_id: int | str,
    cluster_column: str = "cluster_id",
    base_dir: Path | None = None,
) -> Any:
    """Interactive widget to browse clusters on an image."""
    try:
        import ipywidgets as widgets  # type: ignore
    except Exception as e:
        raise ImportError(
            "ipywidgets is required for build_cluster_slider_widget(). Install it or skip widget usage."
        ) from e
    img_path = resolve_image_path(img_id, base_dir=base_dir)
    background = Image.open(img_path).convert("RGB")
    background_array = np.asarray(background)

    subset = df[df["img_id"] == img_id]
    if subset.empty:
        raise ValueError(f"No points for img_id={img_id}")

    clusters = sorted(subset[cluster_column].unique())

    slider = widgets.IntSlider(
        min=0,
        max=len(clusters) - 1,
        step=1,
        value=0,
        description="cluster",
        continuous_update=False,
    )

    out = widgets.Output()

    def render(change=None):
        cid = clusters[int(slider.value)]
        points = subset[subset[cluster_column] == cid]

        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
            ax.imshow(background_array)
            ax.scatter(points["x"], points["y"], s=10, edgecolors="k", linewidths=0.2)
            ax.set_title(f"{cluster_column}={cid} / n={len(points)}")
            ax.axis("off")
            plt.show()

    slider.observe(render, names="value")
    render()

    return widgets.VBox([slider, out])