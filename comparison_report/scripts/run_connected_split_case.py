"""One pre-specified, same-input comparison for two touching cell-like regions.

The region masks set the synthetic scene and are never given to either method.
Seven workers mark the two region centers; three mark their overlap.  Existing
model implementations and fitting settings are imported without modification.
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
import numpy as np

import run_comparison as rc


ROOT = REPORT_ROOT
CASE = ROOT / "connected_split_case"
CONDITION = "connected_split"
INPUT = CASE / "synthetic_inputs" / CONDITION
RESULT = CASE / "results" / CONDITION
FIGURES = ROOT / "figures"
REGIONS = {
    "left": {"center": [35.0, 50.0], "radii": [18.0, 24.0]},
    "right": {"center": [65.0, 50.0], "radii": [18.0, 24.0]},
}
CLICKS = {
    "separated": [[35.0, 50.0], [65.0, 50.0]],
    "merged": [[50.0, 50.0]],
}
WORKERS = [f"A{i}" for i in range(1, 8)] + [f"B{i}" for i in range(1, 4)]
BLUE, RUST, INK = "#286580", "#A46650", "#25343B"


def region_masks() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:100, 0:100]
    output = []
    for region in REGIONS.values():
        cx, cy = region["center"]
        rx, ry = region["radii"]
        output.append(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1)
    left, right = output
    if not np.any(left & right):
        raise ValueError("The prescribed regions must touch/overlap")
    if not (left[50, 35] and not right[50, 35]
            and right[50, 65] and not left[50, 65]
            and left[50, 50] and right[50, 50]):
        raise ValueError("Annotation locations do not match the prescribed regions")
    return left, right


def input_payload(phase: str) -> dict:
    n = 50 if phase == "qc" else 10
    views = []
    for index in range(n):
        image_id = rc.QC_IDS[index // 10] if phase == "qc" else "EVAL_connected_split.png"
        answers = [{"worker_id": worker,
                    "points": CLICKS["separated"] if j < 7 else CLICKS["merged"]}
                   for j, worker in enumerate(WORKERS)]
        views.append({"image_id": image_id, "view_id": str(index % 10),
                      "width": 100, "height": 100, "answers": answers})
    return {"phase": phase, "worker_ids": WORKERS, "views": views}


def inputs() -> None:
    left, right = region_masks()
    INPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CASE / "generating_regions.npz", left=left, right=right)
    definition = {
        "role": "illustrative synthetic two-region setup, not biological ground truth",
        "canvas_pixels": [100, 100], "region_shape": "filled ellipse",
        "regions": REGIONS, "click_templates": CLICKS,
        "worker_rule": "A1-A7: left and right centers; B1-B3: overlap center",
        "known_region_count": 2, "known_group_frequency": {"separated": .7, "merged": .3},
        "qc_image_count": 5, "qc_view_count": 50, "evaluation_view_count": 10,
        "generation_seed": None, "regions_supplied_to_methods": False,
        "method_initializations": list(rc.STARTS),
        "note": "One condition was specified before fitting. No image or mask is supplied to either method.",
    }
    rc.write_json(CASE / "definition.json", definition)
    for phase in ("qc", "evaluation"):
        path = INPUT / f"{phase}_views.json"
        payload = input_payload(phase)
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing {phase} input differs from the prescribed condition")
        rc.write_json(path, payload)
        views, _ = rc.load_views(path)
        rc.point_table(views, INPUT / f"{phase}_points.csv")


def fit_if_needed() -> None:
    archive = rc.check_legacy_archive()
    for method, run in (("legacy", rc.run_legacy), ("current", rc.run_current)):
        status_path = RESULT / method / "run.json"
        if status_path.exists():
            meta = json.loads(status_path.read_text())
            if meta["status"] != "converged":
                raise ValueError(f"Saved {method} result has not converged")
            for phase in ("qc", "evaluation"):
                if meta[f"{phase}_input_sha256"] != rc.digest(INPUT / f"{phase}_views.json"):
                    raise ValueError(f"Saved {method} result used different {phase} input")
            continue
        run(CASE, CONDITION)
    rc.write_json(CASE / "provenance.json", {
        "legacy": archive,
        "current_source_sha256": {rel: rc.digest(source_path(rel))
                                  for rel in ("model.py", "data.py", "run.py")},
        "configs": {"hdbscan": rc.asdict(rc.HDBSCAN_CONFIG),
                    "gmm": rc.asdict(rc.GMM_CONFIG), "em": rc.asdict(rc.EM_CONFIG),
                    "heatmap": rc.asdict(rc.HEATMAP_CONFIG)},
        "qc_input_sha256": rc.digest(INPUT / "qc_views.json"),
        "evaluation_input_sha256": rc.digest(INPUT / "evaluation_views.json"),
        "model_inputs_include_regions_or_group_labels": False,
    })


def region_axis(ax: plt.Axes, left: np.ndarray, right: np.ndarray) -> None:
    rgb = np.full((100, 100, 3), .98)
    rgb[left] = (.88, .91, .91)
    rgb[right] = (.91, .89, .86)
    rgb[left & right] = (.79, .82, .81)
    ax.imshow(rgb, extent=(0, 100, 100, 0), interpolation="nearest")
    ax.contour(left, levels=[.5], colors=[BLUE], linewidths=.9, origin="upper",
               extent=(0, 100, 100, 0))
    ax.contour(right, levels=[.5], colors=[RUST], linewidths=.9, origin="upper",
               extent=(0, 100, 100, 0))
    ax.set_xlim(15, 85)
    ax.set_ylim(83, 17)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render() -> None:
    left, right = region_masks()
    old = np.load(RESULT / "legacy/native_heatmaps.npz")["evaluation"][0]
    outputs = json.loads((RESULT / "current/evaluation_outputs.json").read_text())
    row = outputs[0]
    candidates = row["candidate_points_pixels"]
    weights = np.asarray(row["weights"])
    if len(candidates) != 2 or {json.dumps(c) for c in candidates} != {
            json.dumps(CLICKS["separated"]), json.dumps(CLICKS["merged"])}:
        raise ValueError("Saved candidates differ from the prescribed click configurations")
    split_index = next(i for i, c in enumerate(candidates) if c == CLICKS["separated"])
    merged_index = 1 - split_index
    if not np.allclose([output["weights"][split_index] for output in outputs],
                       weights[split_index]):
        raise ValueError("Evaluation views differ; report per-view results instead")
    assigned = rc.pd.read_csv(RESULT / "legacy/evaluation_final_clusters.csv")
    assigned = assigned.loc[assigned["view_id"].astype(str) == str(row["view_id"])]
    legacy_groups = {str(x): int(assigned.loc[assigned["x"] == x, "final_cluster_id"].iloc[0])
                     for x in (35.0, 50.0, 65.0)}
    summary = {
        "evaluation_view_count": len(outputs),
        "selected_view_id": row["view_id"],
        "candidate_configurations": candidates,
        "current_weights": {"separated": float(weights[split_index]),
                            "merged": float(weights[merged_index])},
        "current_point_count_distribution": row["point_count_distribution"],
        "current_expected_count": row["expected_count"],
        "legacy_native_map_range": [float(old.min()), float(old.max())],
        "legacy_scores_at_left_overlap_right": [float(old[50, x]) for x in (35, 50, 65)],
        "legacy_final_cluster_count": int(assigned["final_cluster_id"].nunique()),
        "legacy_cluster_id_by_x": legacy_groups,
        "legacy_score_display_clip": [-1, 1],
        "comparison_scope": "candidate count/configuration versus signed native score; not commensurate probabilities",
    }
    rc.write_json(CASE / "summary.json", summary)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(left=.06, right=.96, top=.91, bottom=.08, wspace=.22, hspace=.33)
    for ax in axes.flat:
        region_axis(ax, left, right)
    axes[0, 0].set_title("Prescribed two regions", fontsize=16, pad=12)
    axes[0, 1].set_title("2 points · 7 workers", fontsize=16, pad=12)
    axes[0, 2].set_title("1 point · 3 workers", fontsize=16, pad=12)
    for ax, key, color in ((axes[0, 1], "separated", BLUE),
                           (axes[0, 2], "merged", RUST)):
        points = np.asarray(CLICKS[key])
        ax.scatter(points[:, 0], points[:, 1], s=230, color=color,
                   edgecolor="white", linewidth=1.5, zorder=5)
    axes[1, 0].set_title(
        f"Legacy Algorithm score · {summary['legacy_final_cluster_count']} clusters",
        fontsize=16, pad=12)
    axes[1, 0].imshow(np.clip(old, -1, 1), cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=(0, 100, 100, 0), interpolation="nearest", alpha=.82)
    if legacy_groups["35.0"] == legacy_groups["50.0"]:
        axes[1, 0].set_xlabel("Center grouped with left", fontsize=12, labelpad=8)
    axes[1, 1].set_title("Current top configuration", fontsize=16, pad=12)
    best = np.asarray(candidates[int(np.argmax(weights))])
    axes[1, 1].scatter(best[:, 0], best[:, 1], s=250, color=BLUE,
                       edgecolor="white", linewidth=1.5, zorder=5)
    axes[1, 2].cla()
    axes[1, 2].set_title("Current point-count weight", fontsize=16, pad=12)
    values = [float(row["point_count_distribution"].get(str(n), 0)) for n in (1, 2)]
    axes[1, 2].bar(["1 point", "2 points"], values, color=[RUST, BLUE], width=.58)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_yticks([0, .5, 1])
    axes[1, 2].set_ylabel("Weight")
    axes[1, 2].spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(values):
        axes[1, 2].text(i, value + .035, f"{value:.3f}", ha="center", fontsize=14)
    FIGURES.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"connected_split_comparison.{extension}",
                    dpi=200 if extension == "png" else None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    inputs()
    fit_if_needed()
    render()
