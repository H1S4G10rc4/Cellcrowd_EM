"""Render eight slide-ready figures from saved trial and Legacy Algorithm results.

No model fitting or image-level inference is run here. All PNGs are 1920x1080;
SVGs use the same 16:9 canvas and retain editable text.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter


ROOT = PROJECT_ROOT
WORKSPACE = ROOT.parent
SOURCE = ROOT / "outputs" / "convergence_extension"
DESTINATION = SLIDE_OUTPUT_ROOT
from data import DEFAULT_DATA_ROOT
OLD_EM = WORKSPACE / "private_data/legacy_tables"
RAW_IMAGES = DEFAULT_DATA_ROOT

EXAMPLES = (
    ("Sparse", "QC2.png", "280_420_80_80", "low expected count; visible isolated objects"),
    ("Ambiguous", "QC3.png", "280_280_80_80", "competing candidate point counts"),
    ("Dense", "QC1.png", "0_350_80_80", "high expected count; local merged/split pair"),
)
DENSE_ROI = (0, 12, 15, 31)  # x0, y0, x1, y1 in QC1 0_350_80_80

PALETTE = {
    "light": "#B5BEC2",
    "dark": "#58666D",
    "blue": "#286580",
    "rust": "#A46650",
    "ink": "#25343B",
    "grid": "#E8EDEF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "axes.titleweight": "regular",
    "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"],
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "axes.edgecolor": PALETTE["ink"],
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
})


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_figure(*, widths: tuple[float, ...] = (1.0, 1.0)):
    figure = plt.figure(figsize=(16, 9), dpi=120)
    grid = figure.add_gridspec(1, len(widths), width_ratios=widths,
                              left=0.095, right=0.955, bottom=0.17, top=0.87, wspace=0.30)
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(widths))]
    return figure, axes


def style_axis(axis, *, y_grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=15, pad=8)
    if y_grid:
        axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)


def export(figure, name: str) -> None:
    for suffix in ("png", "svg"):
        figure.savefig(DESTINATION / f"{name}.{suffix}", dpi=120)
    plt.close(figure)


def holdout_figure(report: dict) -> None:
    prediction = report["qc_prediction"]
    by_image = prediction["image_mean_log_density"]
    overall = prediction["overall_mean_log_density"]
    methods = (
        ("simple_aggregation", "Simple", PALETTE["light"], "o"),
        ("G1", "G=1", PALETTE["dark"], "s"),
        ("G2", "G=2", PALETTE["blue"], "D"),
    )
    images = tuple(f"QC{i}.png" for i in range(1, 6))
    answer_counts = Counter(
        (row["image_id"], row["method"])
        for row in load_csv(SOURCE / "qc_evaluation" / "heldout_answer_scores.csv")
    )
    rows = []
    for image in (*images, "Overall"):
        for method, label, _, _ in methods:
            value = overall[method] if image == "Overall" else by_image[method][image]
            count = (sum(answer_counts[(item, method)] for item in images)
                     if image == "Overall" else answer_counts[(image, method)])
            if not count:
                raise ValueError(f"No saved holdout answers: {image}, {method}")
            rows.append({"image_id": image, "method": method,
                         "method_label": label, "mean_log_predictive_density": value,
                         "number_of_heldout_answers": count})
    save_csv(DESTINATION / "qc_holdout_comparison.csv", rows)

    figure, (axis, mean_axis) = new_figure(widths=(1.52, 1.0))
    offsets = (-0.20, 0.0, 0.20)
    for offset, (method, label, color, marker) in zip(offsets, methods):
        values = [by_image[method][image] for image in images]
        axis.scatter(np.arange(5) + offset, values, s=112 if method == "G2" else 90,
                     color=color, marker=marker, label=label, zorder=3)
    axis.set_title("By QC image", loc="left", pad=20)
    axis.set_xticks(range(5), [f"QC{i}" for i in range(1, 6)])
    axis.set_xlim(-0.55, 4.55)
    axis.set_ylim(3.75, 9.0)
    axis.set_ylabel("Mean predictive log density")
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.legend(loc="upper left", ncol=3, fontsize=15,
                handletextpad=0.45, columnspacing=1.2)
    style_axis(axis)

    for position, (method, label, color, marker) in zip((2, 1, 0), methods):
        value = overall[method]
        mean_axis.scatter(value, position, s=165 if method == "G2" else 125,
                          color=color, marker=marker, zorder=3)
        mean_axis.text(value + 0.026, position, f"{value:.3f}", va="center", fontsize=16)
    mean_axis.set_title("Overall mean", loc="left", pad=20)
    mean_axis.set_yticks((2, 1, 0), [method[1] for method in methods])
    mean_axis.set_xlim(5.52, 6.13)
    mean_axis.set_xticks((5.6, 5.8, 6.0))
    mean_axis.set_ylim(-0.65, 2.65)
    mean_axis.set_xlabel("Mean predictive log density", labelpad=12)
    mean_axis.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    style_axis(mean_axis, y_grid=False)
    export(figure, "qc_holdout_comparison")


def stable_minority_figure(report: dict) -> None:
    check = report["checks"]["stable_70_30"]
    assert check["passed"]
    weights = [float(check["candidate_weight_A_mean"]), float(check["candidate_weight_B_mean"])]
    rho_groups = [np.asarray(check["rho_group_A"], dtype=float),
                  np.asarray(check["rho_group_B"], dtype=float)]
    assert len(rho_groups[0]) == 7 and len(rho_groups[1]) == 3
    rows = []
    for group, value, target in zip(("A", "B"), weights, (0.7, 0.3)):
        rows.append({"metric": "state_weight", "group": group, "worker_index": "",
                     "value": value, "target_weight": target})
    for group, rho in zip(("A", "B"), rho_groups):
        for index, value in enumerate(rho, 1):
            rows.append({"metric": "rho", "group": group, "worker_index": index,
                         "value": value, "target_weight": ""})
    save_csv(DESTINATION / "stable_minority_diagnostic.csv", rows)

    figure, (weight_axis, rho_axis) = new_figure()
    for index, (value, color) in enumerate(zip(weights, (PALETTE["dark"], PALETTE["blue"]))):
        weight_axis.bar(index, value, width=0.58, color=color, zorder=2)
        weight_axis.text(index, value + 0.025, f"{100 * value:.1f}%",
                         ha="center", va="bottom", fontsize=18)
    weight_axis.scatter((0, 1), (0.7, 0.3), facecolors="white", edgecolors=PALETTE["ink"],
                        linewidth=1.5, s=125, zorder=4, label="Target 70:30")
    weight_axis.set_title("State weights", loc="left", pad=20)
    weight_axis.set_xticks((0, 1), ("Group A", "Group B"))
    weight_axis.set_xlim(-0.6, 1.6)
    weight_axis.set_ylim(0, 1.02)
    weight_axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    weight_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    weight_axis.legend(loc="upper right", fontsize=15)
    style_axis(weight_axis)

    jitter = (np.linspace(-0.16, 0.16, 7), np.linspace(-0.08, 0.08, 3))
    for index, (rho, spread, color) in enumerate(zip(
            rho_groups, jitter, (PALETTE["dark"], PALETTE["blue"]))):
        rho_axis.scatter(index + spread, rho, s=110, color=color,
                         edgecolors="white", linewidth=0.7, zorder=3)
        median = float(np.median(rho))
        rho_axis.hlines(median, index - 0.24, index + 0.24,
                        color=PALETTE["ink"], linewidth=1.7, zorder=4)
        rho_axis.text(index, 0.987, f"Median {median:.3f}", ha="center", va="center", fontsize=16)
    rho_axis.set_title("Worker reliability", loc="left", pad=20)
    rho_axis.set_xticks((0, 1), ("Group A (7)", "Group B (3)"))
    rho_axis.set_xlim(-0.6, 1.6)
    rho_axis.set_ylim(0.82, 1.005)
    rho_axis.set_yticks((0.85, 0.9, 0.95, 1.0))
    rho_axis.set_ylabel("ρ")
    style_axis(rho_axis)
    export(figure, "stable_minority_diagnostic")


def unstable_worker_figure(report: dict) -> None:
    check = report["checks"]["unstable_worker"]
    assert check["passed"]
    stable = np.asarray(check["stable_rho"], dtype=float)
    unstable = float(check["unstable_rho"])
    assert len(stable) == 10
    rows = [{"group": "stable_A" if index <= 7 else "stable_B",
             "worker_index": index if index <= 7 else index - 7, "rho": value}
            for index, value in enumerate(stable, 1)]
    rows.append({"group": "unstable", "worker_index": 1, "rho": unstable})
    save_csv(DESTINATION / "unstable_worker_diagnostic.csv", rows)

    figure, (axis,) = new_figure(widths=(1.0,))
    for center, values, color in ((0, stable[:7], PALETTE["dark"]),
                                  (1, stable[7:], PALETTE["blue"])):
        axis.scatter(center + np.linspace(-0.18, 0.18, len(values)), values,
                     s=140, color=color, edgecolors="white", linewidth=0.7, zorder=3)
        median = float(np.median(values))
        axis.hlines(median, center - 0.26, center + 0.26,
                    color=PALETTE["ink"], linewidth=2.0, zorder=4)
        axis.text(center, 1.025, f"Median {median:.3f}", fontsize=17, ha="center")
    axis.scatter(2.0, unstable, s=200, color=PALETTE["rust"], marker="D", zorder=4)
    axis.text(2.12, unstable, f"{unstable:.3f}", fontsize=18, va="center")
    axis.set_xticks((0, 1, 2), ("Stable A (7)", "Stable B (3)", "Unstable (1)"))
    axis.set_xlim(-0.6, 2.65)
    axis.set_ylim(0, 1.08)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_ylabel("ρ")
    style_axis(axis)
    export(figure, "unstable_worker_diagnostic")


def convergence_figure(report: dict) -> None:
    histories = (
        ("Holdout fit", "G=1", "qc_evaluation/G1_one_history.csv", PALETTE["dark"], "-"),
        ("Holdout fit", "G=2, OSPA init", "qc_evaluation/G2_ospa_pair_history.csv", PALETTE["blue"], "-"),
        ("Holdout fit", "G=2, random init", "qc_evaluation/G2_random_20261008_history.csv", PALETTE["blue"], "--"),
        ("Full QC fit", "G=2, OSPA init", "qc_final/G2_ospa_pair_history.csv", PALETTE["blue"], "-"),
    )
    rows = []
    series = []
    for stage, label, relative, color, style in histories:
        accepted = [row for row in load_csv(SOURCE / relative) if row["accepted"] == "True"]
        iterations = np.asarray([int(row["iteration"]) for row in accepted])
        objectives = np.asarray([float(row["objective"]) for row in accepted])
        if not len(objectives) or np.any(np.diff(objectives) < -1e-8):
            raise ValueError(f"Non-monotone or missing objective history: {relative}")
        series.append((stage, label, color, style, iterations, objectives))
        rows.extend({"stage": stage, "initialization": label, "iteration": int(iteration),
                     "objective": float(objective)}
                    for iteration, objective in zip(iterations, objectives))
    assert (report["qc_prediction"]["attempts"]["G1"][0]["status"] == "converged"
            and all(item["status"] == "converged" for item in report["qc_prediction"]["attempts"]["G2"])
            and report["final_qc"]["converged"])
    save_csv(DESTINATION / "optimization_convergence.csv", rows)

    figure, (evaluation_axis, final_axis) = new_figure(widths=(1.28, 1.0))
    for stage, label, color, style, iterations, objectives in series:
        axis = evaluation_axis if stage == "Holdout fit" else final_axis
        axis.plot(iterations, objectives, color=color, linestyle=style,
                  linewidth=2.5, label=f"{label}  ({iterations[-1]}, converged)", zorder=2)
        axis.scatter(iterations[-1], objectives[-1], s=75, color=color, zorder=4)
    for axis in (evaluation_axis, final_axis):
        axis.set_xlim(0, 750)
        axis.set_xlabel("Iteration", labelpad=12)
        axis.set_ylabel("MAP objective")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.ticklabel_format(style="plain", axis="y", useOffset=False)
        style_axis(axis)
    evaluation_axis.set_title("QC holdout fit (6,860 answers)", loc="left", pad=20)
    evaluation_axis.legend(loc="lower right", fontsize=12.5, handlelength=2.5)
    final_axis.set_title("Full QC fit (8,575 answers)", loc="left", pad=20)
    final_axis.legend(loc="lower right", fontsize=13, handlelength=2.5)
    export(figure, "optimization_convergence")


def non_qc_figure(report: dict) -> None:
    rows = load_csv(SOURCE / "non_qc_one_image" / "comparison.csv")
    assert len(rows) == 49 and all(row["calibrated_output_method"] == "conditional_MAP" for row in rows)
    selected = ["image_id", "view_id", "candidate_count", "inference_status",
                "baseline_expected_count", "calibrated_expected_count",
                "baseline_max_state_weight", "calibrated_max_state_weight",
                "weight_total_variation"]
    save_csv(DESTINATION / "non_qc_one_image_summary.csv",
             [{key: row[key] for key in selected} for row in rows])
    baseline = np.asarray([float(row["baseline_expected_count"]) for row in rows])
    calibrated = np.asarray([float(row["calibrated_expected_count"]) for row in rows])
    variation = np.asarray([float(row["weight_total_variation"]) for row in rows])
    assert np.isfinite(baseline).all() and np.isfinite(calibrated).all() and np.isfinite(variation).all()

    figure, (count_axis, variation_axis) = new_figure(widths=(1.0, 1.0))
    count_axis.plot((2.5, 11), (2.5, 11), color=PALETTE["light"],
                    linewidth=1.6, linestyle="--", zorder=1)
    count_axis.scatter(baseline, calibrated, s=110, color=PALETTE["blue"],
                       alpha=0.78, edgecolors="white", linewidth=0.6, zorder=3)
    count_axis.set_xlim(2.5, 11)
    count_axis.set_ylim(2.5, 11)
    count_axis.set_aspect("equal", adjustable="box")
    count_axis.set_xticks((3, 5, 7, 9, 11))
    count_axis.set_yticks((3, 5, 7, 9, 11))
    count_axis.set_xlabel("Simple aggregation: expected count", labelpad=12)
    count_axis.set_ylabel("Calibrated: expected count")
    count_axis.set_title("Expected count (49 views)", loc="left", pad=20)
    style_axis(count_axis)

    counts, bins, _ = variation_axis.hist(variation, bins=np.linspace(0, 0.09, 10),
                                          color=PALETTE["dark"], edgecolor="white",
                                          linewidth=1.0, zorder=2)
    median = float(np.median(variation))
    variation_axis.axvline(median, color=PALETTE["blue"], linewidth=2.3, zorder=3)
    variation_axis.text(0.085, max(counts) * 0.92, f"Median {median:.3f}",
                        ha="right", va="top", fontsize=17)
    variation_axis.set_xlim(0, 0.09)
    variation_axis.set_ylim(0, max(counts) * 1.12)
    variation_axis.set_xticks((0, 0.02, 0.04, 0.06, 0.08))
    variation_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    variation_axis.set_xlabel("State-weight total variation", labelpad=12)
    variation_axis.set_ylabel("Views")
    variation_axis.set_title("State-weight change (49 views)", loc="left", pad=20)
    style_axis(variation_axis)
    export(figure, "non_qc_one_image_summary")


def load_matched_qc(report: dict) -> list[dict]:
    """Align saved Legacy Algorithm clusters and converged trial states by QC crop."""
    if report["final_qc"]["selected_initialization"] != "ospa_pair":
        raise ValueError("Saved selected QC fit differs from the expected OSPA fit")
    qc_dir = SOURCE / "qc_final"
    inputs_path = qc_dir / "qc_inputs.json.gz"
    with gzip.open(inputs_path, "rt", encoding="utf-8") as stream:
        inputs = json.load(stream)
    with np.load(qc_dir / "G2_ospa_pair_fit.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if (metadata["status"] != "converged"
                or metadata["inputs_sha256"] != hashlib.sha256(inputs_path.read_bytes()).hexdigest()):
            raise ValueError("Saved trial fit is not a matching converged fit")
        q, pi = saved["q"].copy(), saved["pi"].copy()
    worker_q = dict(zip(inputs["worker_ids"], q))

    points = pd.read_csv(OLD_EM / "clustering_final_v2.csv",
                         usecols=["crop_id", "final_cluster_id", "worker_id", "x", "y"])
    posteriors = pd.read_csv(OLD_EM / "staple_cluster_posteriors_v2.csv",
                             usecols=["final_cluster_id", "posterior"])
    centers = (points.groupby(["crop_id", "final_cluster_id"], sort=False)
               .agg(x=("x", "mean"), y=("y", "mean"))
               .reset_index()
               .merge(posteriors, on="final_cluster_id", validate="one_to_one"))
    if centers["posterior"].isna().any():
        raise ValueError("Old EM cluster posterior is missing")
    workers = points.groupby("crop_id")["worker_id"].nunique().to_dict()
    image_workers = {f"QC{number}.png": len(list(
        (DEFAULT_DATA_ROOT / f"QC{number}.png")
        .glob(f"QC{number}.png_*.json"))) for number in range(1, 6)}
    if [image_workers[f"QC{number}.png"] for number in range(1, 6)] != [36, 36, 36, 36, 35]:
        raise ValueError("Old Legacy Algorithm QC worker cohort changed")
    by_crop = {key: frame[["x", "y", "posterior"]].to_numpy(dtype=float)
               for key, frame in centers.groupby("crop_id", sort=False)}

    matched = []
    for index, view in enumerate(inputs["views"]):
        image_id = view["image_id"]
        view_id = view["view_id"]
        crop_id = f"{int(image_id.removeprefix('QC').split('.')[0])}__{view['origin_x']}_{view['origin_y']}"
        if crop_id not in by_crop or len(view["answers"]) == 0:
            raise ValueError(f"No matched QC crop: {image_id}/{view_id}")
        k = len(view["candidates"])
        mixing = np.mean([worker_q[answer["worker_id"]] for answer in view["answers"]], axis=0)
        weights = mixing @ pi[index, :, :k]
        if not np.isclose(weights.sum(), 1.0, atol=1e-9):
            raise ValueError(f"Trial weights do not sum to one: {image_id}/{view_id}")
        counts = np.array([len(candidate["points"]) for candidate in view["candidates"]])
        matched.append({"image_id": image_id, "view_id": view_id, "view": view,
                        "weights": weights, "em_clusters": by_crop[crop_id],
                        "trial_expected_count": float(weights @ counts),
                        "em_expected_count": float(by_crop[crop_id][:, 2].sum()),
                        "candidate_count": k, "trial_worker_count": len(view["answers"]),
                        "em_image_worker_count": image_workers[image_id],
                        "em_workers_with_clicks": int(workers[crop_id]),
                        "count_sd": float(np.sqrt(weights @ (counts - weights @ counts) ** 2))})
    if len(matched) != 245 or len({(item["image_id"], item["view_id"]) for item in matched}) != 245:
        raise ValueError("Expected 245 distinct matched QC views")
    return matched


def point_mass(points: np.ndarray, weights: np.ndarray, size: int = 80) -> np.ndarray:
    """Bilinearly place saved point masses on a display grid; no inference."""
    grid = np.zeros((size, size), dtype=float)
    if len(points) == 0:
        return grid
    x = np.clip(points[:, 0], 0, size - 1)
    y = np.clip(points[:, 1], 0, size - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, size - 1), np.minimum(y0 + 1, size - 1)
    fx, fy = x - x0, y - y0
    for xx, yy, part in ((x0, y0, (1 - fx) * (1 - fy)),
                         (x1, y0, fx * (1 - fy)),
                         (x0, y1, (1 - fx) * fy),
                         (x1, y1, fx * fy)):
        np.add.at(grid, (yy, xx), weights * part)
    return grid


def trial_mass(item: dict) -> np.ndarray:
    grid = np.zeros((80, 80), dtype=float)
    for candidate, weight in zip(item["view"]["candidates"], item["weights"]):
        points = np.asarray(candidate["points"], dtype=float).reshape(-1, 2)
        grid += point_mass(points, np.full(len(points), float(weight)))
    if not np.isclose(grid.sum(), item["trial_expected_count"], atol=1e-9):
        raise ValueError("Trial expected mass is not conserved")
    return grid


def raw_crop(item: dict) -> np.ndarray:
    path = RAW_IMAGES / item["image_id"] / item["image_id"]
    x, y = item["view"]["origin_x"], item["view"]["origin_y"]
    with Image.open(path) as image:
        return np.asarray(image.convert("L").crop((x, y, x + 80, y + 80)))


def image_axis(axis, raw: np.ndarray, *, bounds: tuple[int, int, int, int] | None = None) -> None:
    axis.imshow(raw, cmap="gray", vmin=np.percentile(raw, 1),
                vmax=np.percentile(raw, 99.5), extent=(0, 80, 80, 0), interpolation="nearest")
    if bounds is None:
        axis.set_xlim(0, 80)
        axis.set_ylim(80, 0)
    else:
        x0, y0, x1, y1 = bounds
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["light"])
        spine.set_linewidth(0.7)


def expected_overlay(axis, grid: np.ndarray, color: str) -> None:
    smoothed = gaussian_filter(grid, sigma=1.4)
    rgba = np.zeros((*grid.shape, 4), dtype=float)
    rgba[:, :, :3] = tuple(int(color[pos:pos + 2], 16) / 255 for pos in (1, 3, 5))
    rgba[:, :, 3] = np.clip(smoothed / 0.055, 0, 0.8)
    axis.imshow(rgba, extent=(0, 80, 80, 0), interpolation="bilinear")


def example_rows(matched: list[dict]) -> list[dict]:
    by_key = {(item["image_id"], item["view_id"]): item for item in matched}
    return [(label, by_key[(image_id, view_id)], reason)
            for label, image_id, view_id, reason in EXAMPLES]


def representative_figure(matched: list[dict]) -> None:
    examples = example_rows(matched)
    save_csv(DESTINATION / "representative_visual_examples.csv", [
        {"role": label.lower(), "image_id": item["image_id"], "view_id": item["view_id"],
         "selection_reason": reason, "em_expected_cluster_count": item["em_expected_count"],
         "trial_expected_point_count": item["trial_expected_count"],
         "trial_candidate_count": item["candidate_count"],
         "em_image_worker_count": item["em_image_worker_count"],
         "em_workers_with_clicks": item["em_workers_with_clicks"],
         "trial_worker_count": item["trial_worker_count"]}
        for label, item, reason in examples])
    save_csv(DESTINATION / "selected_views.csv", [
        {"role": label.lower() + ("; dense case study" if label == "Dense" else ""),
         "image_id": item["image_id"], "view_id": item["view_id"],
         "selection_reason": reason}
        for label, item, reason in examples])

    figure = plt.figure(figsize=(16, 9), dpi=120)
    grid = figure.add_gridspec(3, 4, left=0.12, right=0.965, bottom=0.07, top=0.90,
                               hspace=0.24, wspace=0.16)
    headers = ("Raw image", "Crowd clicks", "Legacy Algorithm", "Trial expected mass")
    for row_index, (role, item, _) in enumerate(examples):
        raw = raw_crop(item)
        axes = [figure.add_subplot(grid[row_index, col]) for col in range(4)]
        for col, axis in enumerate(axes):
            image_axis(axis, raw)
            if row_index == 0:
                axis.set_title(headers[col], fontsize=18, pad=12)
        x, y = item["view"]["origin_x"], item["view"]["origin_y"]
        figure.text(0.025, 0.755 - row_index * 0.277,
                    f"{role}\n{item['image_id'].removesuffix('.png')}\n{x},{y}",
                    fontsize=16, va="center", color=PALETTE["ink"])
        crowd = np.asarray([point for answer in item["view"]["answers"]
                            for point in answer["points"]], dtype=float).reshape(-1, 2)
        axes[1].scatter(crowd[:, 0], crowd[:, 1], s=11, color=PALETTE["blue"],
                        alpha=0.26, linewidths=0, rasterized=True)
        em = item["em_clusters"]
        em_grid = point_mass(em[:, :2], em[:, 2])
        if not np.isclose(em_grid.sum(), item["em_expected_count"], atol=1e-9):
            raise ValueError("Legacy Algorithm expected mass is not conserved")
        expected_overlay(axes[2], em_grid, PALETTE["rust"])
        expected_overlay(axes[3], trial_mass(item), PALETTE["blue"])
    export(figure, "representative_visual_examples")


def dense_case_figure(matched: list[dict]) -> None:
    item = example_rows(matched)[2][1]
    view, weights = item["view"], item["weights"]
    raw = raw_crop(item)
    top = np.argsort(weights)[::-1][:3]
    x0, y0, x1, y1 = DENSE_ROI
    rows = []
    for rank, candidate_index in enumerate(top, 1):
        points = np.asarray(view["candidates"][int(candidate_index)]["points"],
                            dtype=float).reshape(-1, 2)
        in_roi = ((points[:, 0] >= x0) & (points[:, 0] < x1)
                  & (points[:, 1] >= y0) & (points[:, 1] < y1))
        rows.append({"image_id": item["image_id"], "view_id": item["view_id"],
                     "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
                     "rank": rank, "candidate_index": int(candidate_index),
                     "state_weight": float(weights[candidate_index]),
                     "full_view_point_count": len(points), "roi_point_count": int(in_roi.sum()),
                     "total_candidate_states": item["candidate_count"],
                     "em_expected_cluster_count_full_view": item["em_expected_count"],
                     "trial_expected_point_count_full_view": item["trial_expected_count"]})
    save_csv(DESTINATION / "dense_region_case_study.csv", rows)

    figure = plt.figure(figsize=(16, 9), dpi=120)
    grid = figure.add_gridspec(2, 12, left=0.065, right=0.96, bottom=0.075, top=0.89,
                               height_ratios=(1.0, 0.90), hspace=0.29, wspace=0.19)
    upper = [figure.add_subplot(grid[0, col * 3:(col + 1) * 3]) for col in range(4)]
    lower = [figure.add_subplot(grid[1, col * 4:(col + 1) * 4]) for col in range(3)]
    for axis, label in zip(upper, ("QC1 0,350 · raw", "Crowd clicks", "Legacy Algorithm", "Trial expected mass")):
        image_axis(axis, raw, bounds=DENSE_ROI)
        axis.set_title(label, fontsize=18, pad=12)
    crowd = np.asarray([point for answer in view["answers"]
                        for point in answer["points"]], dtype=float).reshape(-1, 2)
    upper[1].scatter(crowd[:, 0], crowd[:, 1], s=34, color=PALETTE["blue"],
                     alpha=0.38, linewidths=0)
    em = item["em_clusters"]
    strong = em[:, 2] >= 0.05
    upper[2].scatter(em[strong, 0], em[strong, 1], s=130, facecolors="none",
                     edgecolors=PALETTE["rust"], linewidths=2.4)
    expected_overlay(upper[3], trial_mass(item), PALETTE["blue"])
    for axis, row in zip(lower, rows):
        image_axis(axis, raw, bounds=DENSE_ROI)
        candidate = view["candidates"][row["candidate_index"]]
        pts = np.asarray(candidate["points"], dtype=float).reshape(-1, 2)
        axis.scatter(pts[:, 0], pts[:, 1], s=150, facecolors="none",
                     edgecolors=PALETTE["blue"], linewidths=2.5)
        axis.set_title(f"State {row['rank']} of {item['candidate_count']}  ·  "
                       f"{100 * row['state_weight']:.1f}%", fontsize=18, pad=12)
    export(figure, "dense_region_case_study")


def cellcrowd_trial_figure(matched: list[dict]) -> None:
    save_csv(DESTINATION / "cellcrowd_vs_trial_comparison.csv", [
        {"image_id": item["image_id"], "view_id": item["view_id"],
         "em_expected_cluster_count": item["em_expected_count"],
         "trial_expected_point_count": item["trial_expected_count"],
         "trial_minus_em": item["trial_expected_count"] - item["em_expected_count"],
         "em_image_worker_count": item["em_image_worker_count"],
         "em_workers_with_clicks": item["em_workers_with_clicks"],
         "trial_worker_count": item["trial_worker_count"],
         "trial_candidate_count": item["candidate_count"]}
        for item in matched])
    em = np.asarray([item["em_expected_count"] for item in matched])
    trial = np.asarray([item["trial_expected_count"] for item in matched])
    difference = trial - em
    figure, (scatter_axis, difference_axis) = new_figure(widths=(1.15, 0.85))
    scatter_axis.plot((0, 21), (0, 21), color=PALETTE["light"],
                      linewidth=1.7, linestyle="--", zorder=1)
    scatter_axis.scatter(em, trial, s=69, color=PALETTE["blue"], alpha=0.53,
                         edgecolors="white", linewidth=0.5, zorder=3)
    scatter_axis.set_xlim(0, 21)
    scatter_axis.set_ylim(0, 21)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_xticks((0, 5, 10, 15, 20))
    scatter_axis.set_yticks((0, 5, 10, 15, 20))
    scatter_axis.set_title("245 matched QC views", loc="left", pad=20)
    scatter_axis.set_xlabel("Legacy Algorithm: expected clusters (35–36 workers)", labelpad=12)
    scatter_axis.set_ylabel("Trial: expected points (35 workers)")
    style_axis(scatter_axis)

    counts, _, _ = difference_axis.hist(difference, bins=np.arange(-7.0, 2.6, 0.5),
                                        color=PALETTE["dark"], edgecolor="white", linewidth=0.6)
    median = float(np.median(difference))
    difference_axis.axvline(0, color=PALETTE["light"], linewidth=1.6, linestyle="--")
    difference_axis.axvline(median, color=PALETTE["blue"], linewidth=2.2)
    difference_axis.text(0.02, 0.97, f"Median {median:+.2f}",
                         transform=difference_axis.transAxes, ha="left", va="top", fontsize=17)
    difference_axis.set_ylim(0, max(counts) * 1.12)
    difference_axis.set_xlim(-7, 2.5)
    difference_axis.set_title("View-level count difference", loc="left", pad=20)
    difference_axis.set_xlabel("Trial minus Legacy Algorithm", labelpad=12)
    difference_axis.set_ylabel("Views")
    difference_axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    style_axis(difference_axis)
    export(figure, "cellcrowd_vs_trial_comparison")


def write_manifest() -> None:
    (DESTINATION / "manifest.md").write_text(
        "# Slide figures\n\n"
        "- `qc_holdout_comparison.png` / `.svg`; `qc_holdout_comparison.csv`  \n"
        "  Predictive log density for QC1–QC5 and the overall mean. G=2 has the highest overall mean and exceeds G=1 on four of five images.\n"
        "- `stable_minority_diagnostic.png` / `.svg`; `stable_minority_diagnostic.csv`  \n"
        "  The 70:30 diagnostic recovers weights near 69:31; reliability remains high in both groups.\n"
        "- `unstable_worker_diagnostic.png` / `.svg`; `unstable_worker_diagnostic.csv`  \n"
        "  Reliability for stable groups A and B is compared with that of one unstable worker.\n"
        "- `optimization_convergence.png` / `.svg`; `optimization_convergence.csv`  \n"
        "  Objective histories for three QC holdout fits and the selected full-QC fit; all saved fits converged.\n"
        "- `non_qc_one_image_summary.png` / `.svg`; `non_qc_one_image_summary.csv`  \n"
        "  Expected counts and state-weight changes for all 49 views of one non-QC image: simple aggregation versus QC-calibrated output.\n"
        "- `representative_visual_examples.png` / `.svg`; `representative_visual_examples.csv`  \n"
        "  Three same-view QC examples pair raw images and clicks with saved Legacy Algorithm and trial outputs: sparse QC2/280_420, ambiguous QC3/280_280, and dense QC1/0_350. Selection uses low count, competing candidate counts, and high count with a local split.\n"
        "- `dense_region_case_study.png` / `.svg`; `dense_region_case_study.csv`  \n"
        "  A same-view close-up of QC1/0_350 shows one-versus-two local clicks among the top three trial states beside Legacy Algorithm output. These states carry 3–4% each among 35 candidates.\n"
        "- `cellcrowd_vs_trial_comparison.png` / `.svg`; `cellcrowd_vs_trial_comparison.csv`  \n"
        "  Across 245 matched QC views, Legacy Algorithm's summed cluster posteriors are compared with the trial's candidate-weighted point counts. Worker cohorts differ (EM: 35–36; trial: 35), so this is an output comparison, not a performance ranking.\n\n"
        "`selected_views.csv` records the example IDs and selection reasons. "
        "`make_slide_figures.py` reads saved results and does not refit either model.\n",
        encoding="utf-8",
    )


def main() -> None:
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or not report["final_qc"]["converged"]:
        raise ValueError("The saved convergence-extension result is not complete")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    holdout_figure(report)
    stable_minority_figure(report)
    unstable_worker_figure(report)
    convergence_figure(report)
    non_qc_figure(report)
    matched = load_matched_qc(report)
    representative_figure(matched)
    dense_case_figure(matched)
    cellcrowd_trial_figure(matched)
    write_manifest()
    print(f"Saved eight PNG/SVG pairs, eight figure CSVs, and manifest: {DESTINATION}")


if __name__ == "__main__":
    main()
