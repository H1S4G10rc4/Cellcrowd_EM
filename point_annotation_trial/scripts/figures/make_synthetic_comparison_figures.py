"""Small, same-input synthetic comparisons; never reads or changes real QC fits."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


HERE = SLIDE_OUTPUT_ROOT
ROOT = HERE.parents[2]
TRIAL = ROOT / "point_annotation_trial"
OLD = ROOT / "legacy_algorithm"
REPORT = TRIAL / "outputs/convergence_extension/report.json"
DATA = HERE / "synthetic_qc"
sys.path.insert(0, str(OLD))

from data import Answer, QC_IDS, View, expected_mass, point_count_distribution  # noqa: E402
from model import fit, prepare  # noqa: E402
from src.clustering_update import HDBSCANClusteringConfig  # noqa: E402
from src.clustering import GMMSplitConfig  # noqa: E402
from src.clustering_v2 import cluster_annotations_by_crop, refine_clusters_with_gmm_by_crop  # noqa: E402
from src.staple_em import EMConfig, build_annotation_matrix_from_clusters, run_staple_em  # noqa: E402
from src.heatmap import HeatmapConfig, build_heatmap_for_image  # noqa: E402


COLORS = {"ink": "#26343A", "muted": "#66767C", "a": "#286580", "b": "#A46650",
          "soft": "#CBD3D6", "line": "#DDE3E5"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "axes.labelsize": 18,
    "axes.titlesize": 19, "axes.titleweight": "regular", "axes.edgecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["ink"], "text.color": COLORS["ink"],
    "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"],
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
})

HDBSCAN_CONFIG = HDBSCANClusteringConfig(min_cluster_size_worker_ratio=.35,
                                          min_samples_ratio=.5, eps_scale=1.15,
                                          eps_min_quantile=.82)
GMM_CONFIG = GMMSplitConfig(max_components=3, use_bic_guard=False,
                             dedupe_eps_scale=.3, enforce_worker_uniqueness=False,
                             allow_duplicate_workers=False, bic_improvement_threshold=5.)
EM_CONFIG = EMConfig(max_iter=200, tol=1e-6, init_sensitivity=.7,
                     init_specificity=.7, default_specificity=.7)
HEATMAP_CONFIG = HeatmapConfig(sigma=1.5, kernel_radius_px=3, alpha=.5, cmap="magma")
DISPLAY_SIGMA = 2.0
SCENARIOS = ("stable_70_30", "one_vs_two", "unstable_worker", "complementary_miss")
FIG_NAMES = {"stable_70_30": "stable_minority_diagnostic",
             "one_vs_two": "synthetic_one_vs_two",
             "unstable_worker": "unstable_worker_diagnostic",
             "complementary_miss": "synthetic_complementary_miss"}


def _points(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    if scenario in ("stable_70_30", "unstable_worker"):
        return np.array([[20., 50.]]), np.array([[80., 50.]])
    if scenario == "one_vs_two":
        return np.array([[50., 50.]]), np.array([[35., 50.], [65., 50.]])
    return np.array([[30., 50.]]), np.array([[70., 50.]])


def make_inputs(scenario: str) -> tuple[tuple[View, ...], tuple[str, ...], dict]:
    a, b = _points(scenario)
    n_a = 5 if scenario == "complementary_miss" else 7
    n_b = 5 if scenario == "complementary_miss" else 3
    group_a = tuple(f"A{i}" for i in range(1, n_a + 1))
    group_b = tuple(f"B{i}" for i in range(1, n_b + 1))
    workers = group_a + group_b + (("unstable",) if scenario == "unstable_worker" else ())
    alternating = np.array([0] * 25 + [1] * 25)
    np.random.default_rng(20261008).shuffle(alternating)
    views = []
    for index in range(50):
        answers = [Answer(w, a.copy()) for w in group_a]
        answers += [Answer(w, b.copy()) for w in group_b]
        if scenario == "unstable_worker":
            answers.append(Answer("unstable", (a if alternating[index] == 0 else b).copy()))
        views.append(View(QC_IDS[index // 10], str(index % 10), 0, 0, 100, 100, 100, 100,
                          tuple(answers)))
    truth = [[30., 50.], [70., 50.]] if scenario == "complementary_miss" else None
    definition = {
        "scenario": scenario, "image_equivalents": 5, "views_per_image": 10,
        "views": 50, "coordinate_system": "100 by 100 pixels; normalized coordinates are pixels/100",
        "state_A_points_normalized": (a / 100).tolist(),
        "state_B_points_normalized": (b / 100).tolist(),
        "group_A_workers": list(group_a), "group_B_workers": list(group_b),
        "unstable_worker": "unstable" if scenario == "unstable_worker" else None,
        "unstable_shuffle_seed": 20261008 if scenario == "unstable_worker" else None,
        "generation_seed": 20261008 if scenario == "unstable_worker" else None,
        "unstable_answer_counts": {"A": 25, "B": 25} if scenario == "unstable_worker" else None,
        "generated_two_point_setup_pixels": truth,
        "model_input_excludes_group_labels_and_generated_setup": True,
    }
    return tuple(views), workers, definition


def save_annotations(views: tuple[View, ...], path: Path) -> pd.DataFrame:
    rows = []
    for image_idx, view in enumerate(views):
        crop_id = f"{view.image_id}_{view.view_id}"
        for answer in view.answers:
            for x, y in answer.points:
                rows.append({"crop_id": crop_id, "img_id": view.image_id,
                             "view_id": view.view_id, "point_id": len(rows),
                             "worker_id": answer.worker_id, "x": x, "y": y})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def current_model(views: tuple[View, ...], workers: tuple[str, ...], directory: Path,
                  saved: dict, scenario: str) -> dict:
    prepared = prepare(views, workers)
    attempts = [fit(prepared, 2, start) for start in ("ospa_pair", "random_20261008")]
    usable = [item for item in attempts if item.converged]
    if not usable:
        raise RuntimeError(f"Current model did not converge for {scenario}")
    chosen = max(usable, key=lambda item: item.objective)
    params = chosen.parameters
    weight_rows = []
    for view_idx, view in enumerate(views):
        weights = params.q.mean(axis=0) @ params.pi[view_idx, :, :len(view.candidates)]
        for idx, (candidate, weight) in enumerate(zip(view.candidates, weights)):
            weight_rows.append({"image_id": view.image_id, "view_id": view.view_id,
                                "candidate": idx, "points_pixels": json.dumps(candidate.points.tolist()),
                                "point_count": len(candidate.points), "weight": float(weight)})
    pd.DataFrame(weight_rows).to_csv(directory / "current_candidate_weights.csv", index=False)
    pd.DataFrame({"worker_id": workers, "rho": params.rho,
                  **{f"q_class_{g+1}": params.q[:, g] for g in range(chosen.G)}}).to_csv(
                      directory / "current_worker_parameters.csv", index=False)
    first = views[0]
    first_weights = np.array([r["weight"] for r in weight_rows[:len(first.candidates)]])
    mean_weights = np.array([np.mean([r["weight"] for r in weight_rows
                                      if r["candidate"] == k]) for k in range(len(first.candidates))])
    raw_map = expected_mass(first, first_weights)
    if scenario == "stable_70_30":
        np.testing.assert_allclose(mean_weights,
                                   [saved["stable_70_30"]["candidate_weight_A_mean"],
                                    saved["stable_70_30"]["candidate_weight_B_mean"]], atol=1e-8)
        np.testing.assert_allclose(params.rho[:7], saved["stable_70_30"]["rho_group_A"], atol=1e-8)
    if scenario == "unstable_worker":
        np.testing.assert_allclose(params.rho, saved["unstable_worker"]["stable_rho"] +
                                   [saved["unstable_worker"]["unstable_rho"]], atol=1e-8)
    np.savez_compressed(directory / "current_fit_arrays.npz", omega=params.omega,
                        q=params.q, rho=params.rho, pi=params.pi)
    return {"fit": chosen, "attempts": attempts, "raw_map": raw_map,
            "display_map": gaussian_filter(raw_map, DISPLAY_SIGMA, mode="constant"),
            "first_weights": first_weights, "mean_weights": mean_weights,
            "point_count_distribution": point_count_distribution(first, first_weights),
            "rho": params.rho}


def old_model(annotations: pd.DataFrame, directory: Path) -> dict:
    coarse, cluster_params = cluster_annotations_by_crop(
        HDBSCAN_CONFIG, annotations_path=directory / "annotations.csv",
        output_path=directory / "cellcrowd_coarse_clusters.csv",
        params_output_path=directory / "cellcrowd_cluster_parameters.csv")
    refined = refine_clusters_with_gmm_by_crop(
        coarse, GMM_CONFIG, output_path=directory / "cellcrowd_final_clusters.csv")
    matrix = build_annotation_matrix_from_clusters(refined)
    result = run_staple_em(matrix, EM_CONFIG)
    workers = result.worker_table()
    workers.to_csv(directory / "cellcrowd_worker_parameters.csv")
    centroids = refined.groupby("final_cluster_id")[["x", "y"]].mean()
    posteriors = result.posterior.rename("posterior").to_frame().join(centroids)
    posteriors.to_csv(directory / "cellcrowd_cluster_posteriors.csv")
    crop_id = str(annotations.iloc[0]["crop_id"])
    first_points = annotations[annotations["crop_id"] == crop_id]
    _, heatmap = build_heatmap_for_image(first_points, workers,
                                         image=np.full((100, 100, 3), 255, dtype=np.uint8),
                                         config=HEATMAP_CONFIG)
    return {"result": result, "map": heatmap, "cluster_parameters": cluster_params,
            "first_clusters": posteriors.loc[refined.loc[refined["crop_id"] == crop_id,
                                                      "final_cluster_id"].unique()]}


def _short_axis(ax: plt.Axes) -> None:
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    ax.tick_params(labelsize=15)


def _point_panel(fig: plt.Figure, rect: tuple[float, float, float, float],
                 points: np.ndarray, label: str, color: str = COLORS["a"]) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for edge in ax.spines.values():
        edge.set_color(COLORS["line"])
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=240, color=color, edgecolors="white",
                   linewidths=2, zorder=3)
    ax.set_title(label, pad=17)
    return ax


def _map_panel(fig: plt.Figure, rect: tuple[float, float, float, float], values: np.ndarray,
               label: str, bar_label: str, cmap: str, norm: Normalize,
               extend: str = "neither") -> None:
    ax = fig.add_axes(rect)
    image = ax.imshow(values, origin="upper", extent=[0, 100, 100, 0],
                      cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for edge in ax.spines.values():
        edge.set_color(COLORS["line"])
    ax.set_title(label, pad=17)
    cbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.035, extend=extend)
    cbar.ax.tick_params(labelsize=12, length=3)
    cbar.set_label(bar_label, fontsize=12, labelpad=2)


def _bar_pair(ax: plt.Axes, known: np.ndarray, fitted: np.ndarray, labels: list[str]) -> None:
    xpos = np.arange(len(labels))
    ax.bar(xpos - .16, known, .30, color=COLORS["soft"], label="Known group frequency")
    ax.bar(xpos + .16, fitted, .30, color=COLORS["a"], label="Current model")
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("State weight")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=15)
    for side, values in ((-.16, known), (.16, fitted)):
        for x, y in zip(xpos + side, values):
            ax.text(x, y + .025, f"{y:.2f}", ha="center", va="bottom", fontsize=15)
    _short_axis(ax)


def render(name: str, obj: dict, old_norm: Normalize, current_norm: Normalize) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a, b = _points(name)
    old = obj["old"]
    cur = obj["current"]
    if name in ("stable_70_30", "one_vs_two"):
        xs, width, y, height = (.065, .285, .515, .735), .165, .55, .293
        _point_panel(fig, (xs[0], y, width, height), a,
                     "State A · 7 workers" if name == "stable_70_30" else "1 point · 7 workers")
        _point_panel(fig, (xs[1], y, width, height), b,
                     "State B · 3 workers" if name == "stable_70_30" else "2 points · 3 workers",
                     COLORS["b"])
        _map_panel(fig, (xs[2], y, width, height), old["map"], "Legacy Algorithm",
                   "Score", obj["old_cmap"], old_norm, "min")
        _map_panel(fig, (xs[3], y, width, height), cur["display_map"], "Current model",
                   "Expected mass / pixel", "Blues", current_norm)
        if name == "stable_70_30":
            ax = fig.add_axes((.17, .13, .66, .285))
            _bar_pair(ax, np.array([.7, .3]), cur["mean_weights"], ["State A", "State B"])
        else:
            _point_panel(fig, (.075, .12, .155, .275), a, "Candidate · 1 point")
            _point_panel(fig, (.295, .12, .155, .275), b, "Candidate · 2 points", COLORS["b"])
            ax = fig.add_axes((.61, .14, .32, .23))
            _bar_pair(ax, np.array([.7, .3]), cur["mean_weights"], ["1 point", "2 points"])
            ax.set_ylabel("Probability")
            ax.set_title("State weight = point-count probability", fontsize=16, pad=45)
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2,
                      fontsize=12, frameon=False)
    elif name == "unstable_worker":
        xs, width, y, height = (.035, .227, .419, .611, .803), .16, .58, .284
        _point_panel(fig, (xs[0], y, width, height), a, "Stable A · 7 workers")
        _point_panel(fig, (xs[1], y, width, height), b, "Stable B · 3 workers", COLORS["b"])
        first = obj["views"][0].answers[-1].points
        _point_panel(fig, (xs[2], y, width, height), first, "Switching · 25 A / 25 B", COLORS["muted"])
        _map_panel(fig, (xs[3], y, width, height), old["map"], "Legacy Algorithm",
                   "Score", obj["old_cmap"], old_norm, "min")
        _map_panel(fig, (xs[4], y, width, height), cur["display_map"], "Current model",
                   "Expected mass / pixel", "Blues", current_norm)
        ax = fig.add_axes((.21, .13, .62, .30))
        rho = cur["rho"]
        ax.scatter(rho[:7], np.full(7, 2), color=COLORS["a"], s=80)
        ax.scatter(rho[7:10], np.full(3, 1), color=COLORS["b"], s=80)
        ax.scatter([rho[10]], [0], color=COLORS["muted"], s=125)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(-.5, 2.5)
        ax.set_yticks([0, 1, 2], ["Switching", "Stable B", "Stable A"])
        ax.set_xlabel("Class-consistent response weight, ρ")
        _short_axis(ax)
    else:
        xs, width, y, height = (.035, .227, .419, .611, .803), .16, .58, .284
        generated = np.vstack((a, b))
        _point_panel(fig, (xs[0], y, width, height), generated, "Generated setup · 2 points")
        _point_panel(fig, (xs[1], y, width, height), a, "Group A · left only")
        _point_panel(fig, (xs[2], y, width, height), b, "Group B · right only", COLORS["b"])
        _map_panel(fig, (xs[3], y, width, height), old["map"], "Legacy Algorithm",
                   "Score", obj["old_cmap"], old_norm, "min")
        _map_panel(fig, (xs[4], y, width, height), cur["display_map"], "Current model",
                   "Expected mass / pixel", "Blues", current_norm)
        weights = cur["mean_weights"]
        _point_panel(fig, (.25, .11, .16, .284), a, f"Candidate · left  {weights[0]:.2f}")
        _point_panel(fig, (.59, .11, .16, .284), b, f"Candidate · right  {weights[1]:.2f}", COLORS["b"])
    fig.savefig(HERE / f"{FIG_NAMES[name]}.png", dpi=120)
    fig.savefig(HERE / f"{FIG_NAMES[name]}.svg")
    plt.close(fig)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(results: dict) -> None:
    path = HERE / "manifest.md"
    text = path.read_text(encoding="utf-8")
    marker = "\n## Synthetic point-annotation comparisons\n"
    text = text.split(marker)[0]
    text = text.replace("The 70:30 diagnostic recovers weights near 69:31; reliability remains high in both groups.",
                        "A one-point location split (7 A, 3 B), the two methods' same-input outputs, and known versus fitted state weights.")
    text = text.replace("Reliability for stable groups A and B is compared with that of one unstable worker.",
                        "Stable A and B answers, a switching worker, the two methods' same-input outputs, and the current model's response weights.")
    text += marker
    text += ("The diagrams show synthetic answers, not biological truth or evidence of improved accuracy on real images. "
             "Each method used the same condition-specific synthetic QC (5 image equivalents, 50 views); "
             "real QC parameters and outputs were not used or changed. Generated groups and point setups were excluded from both fits. "
             "The stable and switching checks match saved diagnostics. The saved complementary-miss check "
             "was a single view; its answer pattern was repeated over 50 views here for the same-input fits. "
             "Only the 1-versus-2 pattern was newly specified.\n\n")
    for name in SCENARIOS:
        item = results[name]
        stem = FIG_NAMES[name]
        descriptions = {
            "stable_70_30": "One point at either of two locations; 7/3 stable groups, Legacy Algorithm score map, current expected mass, and known versus fitted weights.",
            "one_vs_two": "Central 1-point versus lateral 2-point answers; candidate configurations, Legacy Algorithm score, current expected mass, state weights, and point-count probabilities.",
            "unstable_worker": "Stable 7/3 groups plus one worker alternating 25/25; same-input output maps and class-consistent response weight for each group.",
            "complementary_miss": "A generated 2-point setup with left-only and right-only answers; same-input output maps and the two observed one-point candidates. The current model does not generate the unseen pair.",
        }
        text += f"- `{stem}.png` / `.svg`  \n  {descriptions[name]}\n"
        text += (f"  Source: `synthetic_qc/{name}/` (`definition.json`, `annotations.csv`, "
                 "`parameters.json`, `current_fit_arrays.npz`, `current_candidate_weights.csv`, `current_worker_parameters.csv`, "
                 "`cellcrowd_worker_parameters.csv`, `cellcrowd_cluster_posteriors.csv`, "
                 "cluster records, and `maps.npz`).\n")
    text += ("\nCellCrowd version: `Cellcrowd_EM_2025/pipeline_v2.ipynb` with "
             "`src/clustering_v2.py`, `src/staple_em.py`, and `src/heatmap.py`; "
             "source SHA-256 values and all configurations are recorded per condition in `parameters.json`. "
             "Input is the synthetic point table; unsupervised crop-local HDBSCAN and GMM splitting produce "
             "binary cluster-by-worker observations. STAPLE-style EM estimates worker sensitivity and specificity "
             "on the same 50 synthetic views. Its output here is the pipeline's positive log-likelihood-ratio "
             "weighted, Gaussian-stamped, consensus-normalized, soft-saturated dimensionless score map, not a state probability. "
             "The current model uses G=2, two fixed initializations and the converged higher-objective fit; "
             "its map is candidate-weighted expected point mass. State weights and point-count probabilities "
             "are defined only for the current model. The plotted maps use the first view; state weights "
             "in the bar charts are means over 50 views. In the 1-versus-2 example each candidate has a "
             "different point count, so its state weight is also the point-count probability. "
             "Map scales are shared across conditions within each method "
             "and are never normalized per image. For legibility, the common Legacy Algorithm color range is fixed "
             "at -1 to 1 with lower values shown as under-range; raw scores, including large negative values "
             "in some conditions, are preserved in `maps.npz`. Legacy Algorithm maps use the original sigma=1.5-pixel peak-normalized "
             "Gaussian stamp (radius 3) and no further display smoothing. Current maps use a sigma=2-pixel "
             "Gaussian filter (`scipy.ndimage.gaussian_filter`, constant boundary) only for display; "
             "unsmoothed and displayed arrays are both in `maps.npz`. The displayed lower-range arrow "
             "marks Legacy Algorithm scores below -1, not missing values. No repeated-view error bars or "
             "significance tests are reported. Two positive Legacy Algorithm peaks in the complementary-miss example "
             "are score-map locations, not a recovered joint two-point configuration. "
             "`make_synthetic_comparison_figures.py` runs only these four "
             "small synthetic conditions and writes no real-QC results. Run `make_slide_figures.py` first "
             "and this script second when regenerating the complete set.\n")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    saved = json.loads(REPORT.read_text(encoding="utf-8"))["checks"]
    DATA.mkdir(exist_ok=True)
    results = {}
    for name in SCENARIOS:
        directory = DATA / name
        directory.mkdir(exist_ok=True)
        views, workers, definition = make_inputs(name)
        (directory / "definition.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
        annotations = save_annotations(views, directory / "annotations.csv")
        current = current_model(views, workers, directory, saved, name)
        old = old_model(annotations, directory)
        result = old["result"]
        if not result.converged:
            raise RuntimeError(f"Legacy Algorithm did not converge for {name}; artifacts retained")
        if name == "complementary_miss":
            assert all(len(candidate.points) == 1 for candidate in views[0].candidates)
            assert np.isclose(current["raw_map"].sum(), 1.0)
        params = {
            "definition": definition,
            "package_versions": {package: importlib.metadata.version(package) for package in (
                "numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "hdbscan")},
            "cellcrowd": {"version": "Legacy Algorithm source snapshot", "converged": result.converged,
                          "iterations": result.iterations, "pi": result.pi,
                          "log_likelihood": result.log_likelihood,
                          "source_sha256": {filename: _hash(OLD / filename) for filename in (
                              "pipeline_v2.ipynb", "src/clustering_v2.py", "src/clustering_update.py",
                              "src/clustering.py", "src/staple_em.py", "src/heatmap.py")},
                          "hdbscan_config": asdict(HDBSCAN_CONFIG), "gmm_config": asdict(GMM_CONFIG),
                          "em_config": asdict(EM_CONFIG), "heatmap_config": asdict(HEATMAP_CONFIG)},
            "current_model": {"G": 2, "chosen_initialization": current["fit"].initialization,
                              "version": "point_annotation_trial",
                              "source_sha256": {filename: _hash(source_path(filename)) for filename in (
                                  "model.py", "data.py", "test_model.py")},
                              "chosen_status": current["fit"].status,
                              "attempts": [{"initialization": attempt.initialization,
                                            "status": attempt.status, "objective": attempt.objective,
                                            "iterations": len(attempt.history)} for attempt in current["attempts"]],
                              "omega": current["fit"].parameters.omega.tolist(),
                              "first_view_state_weights": current["first_weights"].tolist(),
                              "mean_state_weights": current["mean_weights"].tolist(),
                              "first_view_point_count_distribution": current["point_count_distribution"]},
            "display": {"current_gaussian_sigma_pixels": DISPLAY_SIGMA,
                        "current_filter": "scipy.ndimage.gaussian_filter(mode=constant)",
                        "cellcrowd_additional_smoothing": None,
                        "representative_view": f"{views[0].image_id}/{views[0].view_id}"},
        }
        (directory / "parameters.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(directory / "maps.npz", cellcrowd_score=old["map"],
                            current_expected_mass_raw=current["raw_map"],
                            current_expected_mass_display=current["display_map"])
        results[name] = {"views": views, "old": old, "current": current}
        print(f"{name}: Legacy Algorithm converged={result.converged}, current={current['fit'].status}; "
              f"weights={current['mean_weights'].round(4).tolist()}")
    current_max = max(float(np.max(item["current"]["display_map"])) for item in results.values())
    old_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    old_cmap = "RdBu_r"
    current_norm = Normalize(vmin=0, vmax=current_max)
    for name, item in results.items():
        item["old_cmap"] = old_cmap
        render(name, item, old_norm, current_norm)
        path = DATA / name / "parameters.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["display"]["shared_cellcrowd_score_range"] = [float(old_norm.vmin), float(old_norm.vmax)]
        data["display"]["cellcrowd_under_range"] = "Raw scores below -1 are shown with the under-range color."
        data["display"]["cellcrowd_raw_min"] = float(np.min(item["old"]["map"]))
        data["display"]["cellcrowd_raw_max"] = float(np.max(item["old"]["map"]))
        data["display"]["shared_current_mass_range"] = [0., current_max]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_manifest(results)


if __name__ == "__main__":
    main()
