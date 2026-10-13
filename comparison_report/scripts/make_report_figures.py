"""Draw the pairing comparison and real-image overlays from saved arrays."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "point_annotation_trial" / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter


HERE = REPORT_ROOT
WORKSPACE = HERE.parent
SLIDE_SOURCE = FIGURE_SOURCE_ROOT / "make_slide_figures.py"
OUTPUT = HERE / "figures"
DEMO = HERE / "pairing_demo"
BLUE = "#286580"
RUST = "#A46650"
INK = "#25343B"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 17,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#8A969C",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none",
})


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=200, facecolor="white")
    fig.savefig(OUTPUT / f"{name}.svg", facecolor="white")
    plt.close(fig)


def point_axis(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFBFB")
    ax.scatter(points[:, 0], points[:, 1], s=210, facecolor=color,
               edgecolor="white", linewidth=1.3, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#AEB9BE")
        spine.set_linewidth(.8)


def pairing_figure() -> None:
    compare = json.loads((DEMO / "map_comparison.json").read_text())
    if not compare["exactly_equal"] or compare["max_absolute_difference"] != 0:
        raise ValueError("The two native Legacy Algorithm maps are not exactly equal")
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, left=.065, right=.87, bottom=.08, top=.89,
                            height_ratios=(1.0, 1.0), hspace=.32, wspace=.22)
    maps = []
    current = []
    for name in ("diagonal", "vertical"):
        base = DEMO / "results" / name
        maps.append(np.load(base / "legacy/native_heatmaps.npz")["evaluation"][0])
        current.append(json.loads((base / "current/evaluation_outputs.json").read_text())[0])
    norm = Normalize(vmin=min(float(m.min()) for m in maps), vmax=0)
    for col, (name, old, row) in enumerate(zip(("Diagonal", "Vertical"), maps, current)):
        inner = grid[0, col].subgridspec(1, 2, wspace=.16)
        for j in range(2):
            ax = fig.add_subplot(inner[0, j])
            point_axis(ax, np.asarray(row["candidate_points_pixels"][j]),
                       BLUE if j == 0 else RUST)
            ax.set_title(f"5 workers  ·  {100*row['weights'][j]:.1f}%", fontsize=15, pad=8)
        fig.text(.255 if col == 0 else .68, .94,
                 f"{name} pairing", ha="center", fontsize=18, color=INK)
        ax = fig.add_subplot(grid[1, col])
        image = ax.imshow(old, cmap="cividis_r", norm=norm, origin="upper",
                          interpolation="nearest", extent=(0, 100, 100, 0))
        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Legacy Algorithm native score map", fontsize=16, pad=9)
    cax = fig.add_axes([.90, .09, .014, .30])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Signed score", fontsize=13)
    bar.ax.tick_params(labelsize=11)
    save(fig, "pairing_comparison")


def import_slide_source():
    spec = importlib.util.spec_from_file_location("saved_slide_source", SLIDE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load saved slide figure helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_overlay_figure() -> None:
    slide = import_slide_source()
    report = json.loads((slide.SOURCE / "report.json").read_text())
    matched = slide.load_matched_qc(report)
    by_key = {(item["image_id"], item["view_id"]): item for item in matched}
    selected = [by_key[("QC2.png", "280_420_80_80")],
                by_key[("QC3.png", "280_280_80_80")]]
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 3, left=.12, right=.97, top=.91, bottom=.09,
                            wspace=.20, hspace=.30)
    for row_index, item in enumerate(selected):
        raw = slide.raw_crop(item)
        axes = [fig.add_subplot(grid[row_index, i]) for i in range(3)]
        low, high = np.percentile(raw, [1, 99.5])
        display_norm = PowerNorm(gamma=.65, vmin=low, vmax=high)
        for ax in axes:
            slide.image_axis(ax, raw)
            ax.images[0].set_norm(display_norm)
        if row_index == 0:
            for ax, label in zip(axes, ("Raw image", "35 crowd annotations",
                                        "Current expected mass")):
                ax.set_title(label, fontsize=17, pad=10)
        clicks = np.asarray([point for answer in item["view"]["answers"]
                             for point in answer["points"]], dtype=float).reshape(-1, 2)
        axes[1].scatter(clicks[:, 0], clicks[:, 1], s=30,
                        facecolor="#F39B3D", edgecolor="#20282C", linewidth=.55,
                        alpha=.85, rasterized=True)
        smoothed = gaussian_filter(slide.trial_mass(item), sigma=1.4)
        rgba = np.zeros((*smoothed.shape, 4), dtype=float)
        rgba[:, :, :3] = (239 / 255, 105 / 255, 56 / 255)
        rgba[:, :, 3] = np.clip((smoothed - .003) / .045, 0, .88)
        axes[2].imshow(rgba, extent=(0, 80, 80, 0), interpolation="bilinear")
        axes[2].contour(smoothed, levels=[.015], colors=["#4E2223"],
                        linewidths=.75, origin="upper", extent=(0, 80, 80, 0))
        fig.text(.02, .70 if row_index == 0 else .28,
                 f"{item['image_id'].removesuffix('.png')}\n{item['view']['view_id'].split('_')[0]},"
                 f"{item['view']['view_id'].split('_')[1]}",
                 fontsize=16, va="center", color=INK)
    save(fig, "real_image_current_overlay")
    metadata = {
        "views": [{"image_id": item["image_id"], "view_id": item["view_id"],
                   "worker_count": item["trial_worker_count"],
                   "candidate_count": item["candidate_count"],
                   "expected_point_count": item["trial_expected_count"]}
                  for item in selected],
        "source_qc_inputs": str(slide.SOURCE / "qc_final/qc_inputs.json.gz"),
        "source_qc_inputs_sha256": hashlib.sha256(
            (slide.SOURCE / "qc_final/qc_inputs.json.gz").read_bytes()).hexdigest(),
        "source_qc_fit": str(slide.SOURCE / "qc_final/G2_ospa_pair_fit.npz"),
        "source_raw_images": str(slide.RAW_IMAGES),
        "display": "Original 80x80 crop, crowd points in orange with dark edges, and saved G=2 candidate-weighted point mass in orange. Each row uses the same grayscale PowerNorm gamma=0.65 across its three images, with limits at the 1st and 99.5th raw-image percentiles. Expected mass has Gaussian display smoothing sigma=1.4 px and fixed alpha clip((smoothed-0.003)/0.045, 0, 0.88), plus a 0.015 mass contour. Smoothing affects display only.",
        "note": "Real-QC current-model output only; not a same-input legacy performance comparison.",
    }
    (HERE / "real_overlay_provenance.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    pairing_figure()
    real_overlay_figure()
