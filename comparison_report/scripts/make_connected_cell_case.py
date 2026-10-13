"""Render a connected-cell synthetic comparison from saved same-input results.

The two generating regions are specified before the click rules below.  The
generated answer records must match the saved B_count inputs exactly; only
then are the already-fitted outputs read.  Masks are never model inputs.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "point_annotation_trial" / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import run_comparison as rc


ROOT = REPORT_ROOT
SOURCE = ROOT / "synthetic_inputs/B_count"
RESULT = ROOT / "results/B_count"
TARGET = ROOT / "connected_cell_case"
FIGURES = ROOT / "figures"
SHAPE = (100, 100)
REGIONS = {
    "left": {"center": (35.0, 50.0), "radii": (18.0, 24.0)},
    "right": {"center": (65.0, 50.0), "radii": (18.0, 24.0)},
}
CLICKS = {
    "merged": [[50.0, 50.0]],
    "separated": [[35.0, 50.0], [65.0, 50.0]],
}
WORKERS = [f"A{i}" for i in range(1, 8)] + [f"B{i}" for i in range(1, 4)]
BLUE, RUST, INK = "#286580", "#A46650", "#25343B"


def masks() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    result = []
    for region in REGIONS.values():
        cx, cy = region["center"]
        rx, ry = region["radii"]
        result.append(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1)
    return tuple(result)


def generated_input(phase: str) -> dict:
    n = 50 if phase == "qc" else 10
    views = []
    for index in range(n):
        image_id = rc.QC_IDS[index // 10] if phase == "qc" else "EVAL_B_count.png"
        answers = [{"worker_id": worker,
                    "points": CLICKS["merged"] if j < 7 else CLICKS["separated"]}
                   for j, worker in enumerate(WORKERS)]
        views.append({"image_id": image_id, "view_id": str(index % 10),
                      "width": 100, "height": 100, "answers": answers})
    return {"phase": phase, "worker_ids": WORKERS, "views": views}


def verify_and_save() -> tuple[np.ndarray, np.ndarray, dict]:
    left, right = masks()
    if not np.any(left & right) or not np.any(left & ~right) or not np.any(right & ~left):
        raise ValueError("The generating regions must overlap without coinciding")
    points = [CLICKS["merged"][0], *CLICKS["separated"]]
    if not all((left | right)[int(y), int(x)] for x, y in points):
        raise ValueError("An annotation lies outside both regions")
    if not left[50, 50] or not right[50, 50] or not left[50, 35] or not right[50, 65]:
        raise ValueError("Click-to-region membership is not as specified")
    TARGET.mkdir(exist_ok=True)
    matching = {}
    for phase in ("qc", "evaluation"):
        generated = generated_input(phase)
        source = json.loads((SOURCE / f"{phase}_views.json").read_text())
        if generated != source:
            raise ValueError(f"Generated {phase} input does not match saved condition B")
        path = TARGET / f"{phase}_views.json"
        rc.write_json(path, generated)
        matching[phase] = {
            "source_path": str(SOURCE / f"{phase}_views.json"),
            "source_sha256": rc.digest(SOURCE / f"{phase}_views.json"),
            "generated_sha256": rc.digest(path),
            "exact_json_match": rc.digest(path) == rc.digest(SOURCE / f"{phase}_views.json"),
        }
    np.savez_compressed(TARGET / "generating_regions.npz", left=left, right=right)
    old_meta = json.loads((RESULT / "legacy/run.json").read_text())
    new_meta = json.loads((RESULT / "current/run.json").read_text())
    for method, meta in (("legacy", old_meta), ("current", new_meta)):
        if meta["status"] != "converged":
            raise ValueError(f"{method} did not converge")
        for phase in ("qc", "evaluation"):
            if meta[f"{phase}_input_sha256"] != matching[phase]["source_sha256"]:
                raise ValueError(f"{method} result input mismatch: {phase}")
    outputs = json.loads((RESULT / "current/evaluation_outputs.json").read_text())
    first = outputs[0]
    if first["candidate_points_pixels"] != [CLICKS["merged"], CLICKS["separated"]]:
        raise ValueError("Saved candidate order differs from generated input")
    expected = np.load(RESULT / "current/evaluation_expected_mass.npz")["mass"][0]
    if not np.isclose(expected.sum(), first["expected_count"], atol=1e-10):
        raise ValueError("Expected mass is not conserved")
    old = np.load(RESULT / "legacy/native_heatmaps.npz")["evaluation"][0]
    definition = {
        "role": "illustrative synthetic projected regions, not biological ground truth",
        "canvas_pixels": [100, 100], "region_shape": "filled ellipse",
        "regions": REGIONS, "annotation_templates": CLICKS,
        "worker_rule": "A1-A7: central one-point response; B1-B3: left and right two-point response",
        "qc_image_count": 5, "qc_view_count": 50, "evaluation_view_count": 10,
        "seed": None, "regions_supplied_to_methods": False,
        "input_equivalence": matching,
        "legacy_version": "Cellcrowd_EM_2025/pipeline_v2.ipynb settings through comparison_report/scripts/run_comparison.py",
        "legacy_output": "unmodified native signed score map; plotted with a [-1,1] display clip",
        "current_version": "point_annotation_trial G=2 through comparison_report/scripts/run_comparison.py",
        "current_output": "observed candidate configurations, weights, point-count distribution and expected point mass",
        "legacy_raw_map_range": [float(old.min()), float(old.max())],
        "current_view0_weights": first["weights"],
        "current_view0_count_distribution": first["point_count_distribution"],
        "current_view0_expected_count": first["expected_count"],
        "note": "The prescribed regions only govern synthetic setup and drawing. Neither method receives a mask or image. The generated answers exactly match the saved B_count input, so no refitting was needed.",
    }
    rc.write_json(TARGET / "definition_and_results.json", definition)
    return old, expected, first


def region_axis(ax: plt.Axes, left: np.ndarray, right: np.ndarray) -> None:
    rgb = np.full((100, 100, 3), 0.98)
    rgb[left] = (.88, .91, .91)
    rgb[right] = (.91, .89, .86)
    rgb[left & right] = (.79, .82, .81)
    ax.imshow(rgb, extent=(0, 100, 100, 0), interpolation="nearest")
    ax.contour(left, levels=[.5], colors=[BLUE], linewidths=.9, origin="upper", extent=(0, 100, 100, 0))
    ax.contour(right, levels=[.5], colors=[RUST], linewidths=.9, origin="upper", extent=(0, 100, 100, 0))
    ax.set_xlim(15, 85)
    ax.set_ylim(83, 17)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw() -> None:
    old, expected, first = verify_and_save()
    left, right = masks()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0))
    fig.subplots_adjust(left=.06, right=.96, top=.91, bottom=.08, wspace=.22, hspace=.33)
    for ax in axes.flat:
        region_axis(ax, left, right)
    axes[0, 0].set_title("Prescribed cell-like regions", fontsize=16, pad=12)
    axes[0, 1].set_title("1 point · 7 workers", fontsize=16, pad=12)
    axes[0, 2].set_title("2 points · 3 workers", fontsize=16, pad=12)
    for ax, key, color in ((axes[0, 1], "merged", BLUE), (axes[0, 2], "separated", RUST)):
        points = np.asarray(CLICKS[key])
        ax.scatter(points[:, 0], points[:, 1], s=230, color=color,
                   edgecolor="white", linewidth=1.5, zorder=5)
    axes[1, 0].set_title("Legacy Algorithm signed score · clipped", fontsize=16, pad=12)
    display_old = np.clip(old, -1.0, 1.0)
    axes[1, 0].imshow(display_old, cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=(0, 100, 100, 0), interpolation="nearest", alpha=.82)
    axes[1, 1].set_title("Current expected point mass", fontsize=16, pad=12)
    coordinates = np.argwhere(expected > 0)
    masses = expected[coordinates[:, 0], coordinates[:, 1]]
    axes[1, 1].scatter(coordinates[:, 1], coordinates[:, 0], s=1050 * masses,
                       color=BLUE, edgecolor="white", linewidth=1.3, alpha=.88, zorder=5)
    axes[1, 2].set_title("Current point-count distribution", fontsize=16, pad=12)
    axes[1, 2].cla()
    axes[1, 2].bar(["1 point", "2 points"], first["weights"],
                   color=[BLUE, RUST], width=.58)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_yticks([0, .5, 1])
    axes[1, 2].set_ylabel("Weight")
    axes[1, 2].spines[["top", "right"]].set_visible(False)
    for i, weight in enumerate(first["weights"]):
        axes[1, 2].text(i, weight + .035, f"{weight:.3f}", ha="center", fontsize=14)
    FIGURES.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"connected_cell_comparison.{extension}",
                    dpi=200 if extension == "png" else None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
