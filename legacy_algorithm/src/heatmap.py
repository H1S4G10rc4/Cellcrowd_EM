"""Heatmap construction utilities based on STAPLE weights.

`set` ドキュメントの工程:
1. アノテーション点にワーカ固有の LLR 重みを割り当てる
2. 重み付きインパルス列をガウシアン畳み込みして密度疑似ヒートマップを得る
3. 元画像にオーバーレイして可視化
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter


@dataclass
class HeatmapConfig:
    """Heuristics for building weighted heatmaps.

    sigma:
        Per-point Gaussian sigma (local stamp). 小さいほどピークが鋭くなる。
    kernel_radius_px:
        ガウシアンを切り落とす半径。None なら 2.5 * sigma を使用。
    global_sigma:
        必要なら全体に後段のガウシアンブラーを一度かける（例: 0.5〜1.5）。
    normalize_by_total_weight:
        True の場合、累積重みを全ワーカーの重み合計で割る。
        アノテーター数が画像ごとに異なる場合の正規化に有効。
    saturation_scale:
        正規化後に掛けるスケール係数。全員一致時の飽和後の値を調整。
        - scale=3.0: 全員一致 → ~0.95
        - scale=2.0: 全員一致 → ~0.86
        - scale=1.0: 全員一致 → ~0.63
    saturation_mode:
        累積ヒートマップを0-1に収める方式:
        - 'soft': 1 - exp(-x) でソフト飽和（推奨）
        - 'tanh': tanh(x) で飽和
        - 'clip': 単純に[0,1]でクリップ
        - None: 飽和処理なし（従来の動作）
    """

    sigma: float = 2.0
    kernel_radius_px: int | None = None
    global_sigma: float | None = None
    alpha: float = 0.6  # Overlay transparency
    cmap: str = "copper"
    weight_column: str = "weight_positive"
    clip_min: float | None = None
    clip_max: float | None = None
    normalize: bool = False  # 飽和モードでは通常無効化
    normalize_by_total_weight: bool = True  # コンセンサス正規化
    saturation_scale: float = 6.0  # 全員一致で~0.998になるよう設定
    saturation_mode: str | None = "soft"


def _ensure_image_array(
    image: np.ndarray | Image.Image | str | Path | None, image_path: str | Path | None
) -> np.ndarray:
    source = image if image is not None else image_path
    if source is None:
        raise ValueError("Either `image` or `image_path` must be provided.")

    if isinstance(source, np.ndarray):
        array = source
    else:
        if isinstance(source, Image.Image):
            pil_img = source
        else:
            pil_img = Image.open(source)  # type: ignore[arg-type]
        pil_img = pil_img.convert("RGB")
        array = np.asarray(pil_img)

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _worker_weight_lookup(
    worker_table: pd.DataFrame | pd.Series, weight_column: str
) -> Mapping[str, float]:
    if isinstance(worker_table, pd.Series):
        series = worker_table.copy()
    else:
        if weight_column not in worker_table.columns:
            raise KeyError(f"weight column '{weight_column}' not found in worker table.")
        series = worker_table[weight_column].copy()
    series.index = series.index.map(str)
    return series.to_dict()


def _gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """ピーク正規化されたガウシアンカーネルを生成。中心が1.0になる。"""
    sigma = max(1e-3, float(sigma))
    if radius is None:
        radius = int(max(1, np.ceil(2.5 * sigma)))
    radius = int(max(1, radius))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    kernel[kernel < 1e-9] = 0.0
    # ピーク正規化：中心が1.0になるように正規化
    peak = kernel.max()
    if peak <= 0:
        kernel[radius, radius] = 1.0
        peak = 1.0
    return kernel / peak


def build_heatmap_for_image(
    points: pd.DataFrame,
    worker_table: pd.DataFrame | pd.Series,
    *,
    image: np.ndarray | Image.Image | str | Path | None = None,
    image_path: str | Path | None = None,
    config: HeatmapConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign weights to annotation points and blur with a Gaussian kernel."""

    if config is None:
        config = HeatmapConfig()

    for col in ("x", "y", "worker_id"):
        if col not in points.columns:
            raise ValueError(f"required column missing: '{col}'")

    image_array = _ensure_image_array(image, image_path)
    height, width = image_array.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    weight_lookup = _worker_weight_lookup(worker_table, config.weight_column)
    if not weight_lookup:
        raise ValueError("worker weight table is empty.")

    # この画像をアノテーションしたワーカーを特定
    annotating_workers = set()
    for row in points.itertuples(index=False):
        worker_id = getattr(row, "worker_id")
        worker_key = str(worker_id)
        if worker_key not in weight_lookup and isinstance(worker_id, (int, np.integer)):
            worker_key = str(int(worker_id))
        if worker_key in weight_lookup:
            annotating_workers.add(worker_key)

    kernel = _gaussian_kernel(config.sigma, config.kernel_radius_px)
    radius_y = kernel.shape[0] // 2
    radius_x = kernel.shape[1] // 2

    for row in points.itertuples(index=False):
        x = float(getattr(row, "x"))
        y = float(getattr(row, "y"))
        worker_id = getattr(row, "worker_id")
        worker_key = str(worker_id)
        if worker_key not in weight_lookup and isinstance(worker_id, (int, np.integer)):  # type: ignore[name-defined]
            worker_key = str(int(worker_id))
        weight = weight_lookup.get(worker_key)
        if weight is None:
            continue
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= width or yi >= height:
            continue

        x0 = max(0, xi - radius_x)
        y0 = max(0, yi - radius_y)
        x1 = min(width, xi + radius_x + 1)
        y1 = min(height, yi + radius_y + 1)

        kx0 = x0 - (xi - radius_x)
        ky0 = y0 - (yi - radius_y)
        kx1 = kernel.shape[1] - ((xi + radius_x + 1) - x1)
        ky1 = kernel.shape[0] - ((yi + radius_y + 1) - y1)

        heatmap[y0:y1, x0:x1] += float(weight) * kernel[ky0:ky1, kx0:kx1]

    if config.global_sigma and config.global_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=float(config.global_sigma), mode="constant")

    # コンセンサス正規化: この画像をアノテーションしたワーカーの重み合計で割る
    if config.normalize_by_total_weight:
        # 実際にアノテーションしたワーカーの重み合計のみ使用
        total_weight = sum(weight_lookup[w] for w in annotating_workers)
        if total_weight > 0:
            heatmap = heatmap / total_weight
            # スケール係数を適用（全員一致時の飽和後の値を調整）
            heatmap = heatmap * config.saturation_scale

    # 飽和モードの適用（0-1に収める）
    if config.saturation_mode == "soft":
        # 1 - exp(-x): x=0→0, x→∞→1, 単調増加
        heatmap = 1.0 - np.exp(-heatmap)
    elif config.saturation_mode == "tanh":
        heatmap = np.tanh(heatmap)
    elif config.saturation_mode == "clip":
        heatmap = np.clip(heatmap, 0.0, 1.0)
    # saturation_mode が None の場合は従来通り

    if config.clip_min is not None:
        heatmap = np.maximum(heatmap, float(config.clip_min))
    if config.clip_max is not None:
        heatmap = np.minimum(heatmap, float(config.clip_max))

    if config.normalize:
        vmax = heatmap.max()
        if vmax > 0:
            heatmap = heatmap / vmax

    return image_array, heatmap


def overlay_heatmap(
    image: np.ndarray | Image.Image | str | Path,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.6,
    cmap: str = "copper",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 8),
    peaks: Sequence[Sequence[float]] | np.ndarray | None = None,
    peak_style: dict | None = None,
) -> plt.Figure:
    """Overlay a heatmap onto an RGB image and return the matplotlib Figure."""

    image_array = _ensure_image_array(image, None)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(image_array)
    hm = ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    if peaks is not None:
        peaks_arr = np.asarray(peaks)
        if peaks_arr.size:
            style = {
                "c": "red",
                "s": 12,
                "marker": "o",
                "edgecolors": "white",
                "linewidths": 0.3,
            }
            if peak_style:
                style.update(peak_style)
            ax.scatter(peaks_arr[:, 1], peaks_arr[:, 0], **style)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def find_local_peaks(
    heatmap: np.ndarray,
    *,
    min_distance: int = 3,
    threshold_abs: float = 0.1,
    max_peaks: int | None = None,
) -> np.ndarray:
    """Detect local maxima on the normalized heatmap."""

    data = np.asarray(heatmap, dtype=float)
    if data.ndim != 2:
        raise ValueError("heatmap must be 2-D.")
    if data.size == 0:
        return np.zeros((0, 2), dtype=int)

    size = max(1, int(min_distance)) * 2 + 1
    filtered = maximum_filter(data, size=size, mode="constant")
    mask = (data == filtered) & (data >= float(threshold_abs))

    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        return np.zeros((0, 2), dtype=int)

    values = data[mask]
    order = np.argsort(values)[::-1]
    coords = coords[order]

    selected: list[tuple[int, int]] = []
    min_dist_sq = float(max(1, min_distance) ** 2)
    for y, x in coords:
        if all((y - sy) ** 2 + (x - sx) ** 2 >= min_dist_sq for sy, sx in selected):
            selected.append((int(y), int(x)))
            if max_peaks is not None and len(selected) >= max_peaks:
                break

    return np.asarray(selected, dtype=int)


def plot_heatmap_with_peaks(
    image: np.ndarray | Image.Image | str | Path,
    heatmap: np.ndarray,
    peaks: np.ndarray,
    *,
    alpha: float = 0.6,
    cmap: str = "copper",
    figsize: tuple[int, int] = (12, 5),
    title_heatmap: str | None = None,
    title_peaks: str | None = None,
    peak_style: dict | None = None,
) -> plt.Figure:
    """Plot heatmap overlay and peak overlay side by side."""

    image_array = _ensure_image_array(image, None)
    peaks = np.asarray(peaks)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    style_heatmap = {
        "c": "red",
        "s": 12,
        "marker": "o",
        "edgecolors": "white",
        "linewidths": 0.3,
    }
    style_peaks = {
        "c": "red",
        "s": 12,
        "marker": "o",
        "edgecolors": "black",
        "linewidths": 0.3,
    }
    if peak_style:
        style_heatmap.update(peak_style)
        style_peaks.update(peak_style)

    axes[0].imshow(image_array)
    hm = axes[0].imshow(heatmap, cmap=cmap, alpha=alpha)
    axes[0].axis("off")
    if title_heatmap:
        axes[0].set_title(title_heatmap)

    axes[1].imshow(image_array)
    if peaks.size:
        axes[1].scatter(peaks[:, 1], peaks[:, 0], **style_peaks)
    axes[1].axis("off")
    if title_peaks:
        axes[1].set_title(title_peaks)

    fig.colorbar(hm, ax=axes[0], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


__all__ = [
    "HeatmapConfig",
    "build_heatmap_for_image",
    "overlay_heatmap",
    "find_local_peaks",
    "plot_heatmap_with_peaks",
]

