"""Reproducible small pilot; all fitted quantities and failed attempts are saved."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from configuration_model import (Observation, Region, Settings, fit, infer, prepare,
                                 summarize, posterior)
from local_regions import make_candidates, regions_from_views, save_regions
from data import (Answer, View, DEFAULT_DATA_ROOT, QC_IDS, load_image_views,
                  select_qc_workers, write_csv, write_json)

HERE = PROJECT_ROOT
ROOT = HERE.parent
SEED = 20261010
STARTS = ("diagonal", "uniform", "random")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_fit(folder, regions, settings=Settings()):
    folder.mkdir(parents=True, exist_ok=False)
    save_regions(folder / "regions.json", regions)
    write_json(folder / "settings.json", asdict(settings))
    data = prepare(regions, settings=settings)
    attempts = []
    for start in STARTS:
        result = fit(data, start, SEED)
        attempts.append(result)
        write_csv(folder / f"history_{start}.csv", result.history)
        np.savez_compressed(folder / f"parameters_{start}.npz", error_tables=result.error_tables,
                            expected_counts=result.expected_counts, worker_ids=np.asarray(result.worker_ids))
        write_json(folder / f"posteriors_{start}.json", [summarize(r, w) for r, w in zip(regions, result.weights)])
        print(json.dumps({"stage": str(folder), "start": start, "converged": result.converged,
                          "iterations": len(result.history)-1, "objective": result.objective}, ensure_ascii=False), flush=True)
    accepted = [a for a in attempts if a.converged]
    if not accepted:
        write_json(folder / "status.json", {"status": "not_converged"})
        raise RuntimeError("All initializations failed to converge; outputs retained")
    chosen = max(accepted, key=lambda a: a.objective)
    near = [a for a in accepted if chosen.objective - a.objective <= 2.]
    instability = [max((float(abs(w - a.weights[i]).sum() / 2) for a in near), default=0.)
                   for i, w in enumerate(chosen.weights)]
    support = chosen.expected_counts.sum(axis=2)
    rows = []
    for j, worker in enumerate(chosen.worker_ids):
        for m in range(3):
            for n in range(3):
                rows.append({"worker_id": worker, "underlying_count_bin": ("0", "1", "2+")[m],
                             "reported_count_bin": ("0", "1", "2+")[n],
                             "probability": float(chosen.error_tables[j, m, n]),
                             "effective_observations_for_row": float(support[j, m]),
                             "weakly_supported_row": bool(support[j, m] < 5)})
    write_csv(folder / "worker_count_errors.csv", rows)
    info = {"status": "converged", "selected_initialization": chosen.initialization,
            "objective": chosen.objective, "worker_ids": list(chosen.worker_ids),
            "free_parameters_per_worker": 6, "free_group_parameters": 0,
            "attempts": [{"initialization": a.initialization, "converged": a.converged,
                          "objective": a.objective, "iterations": len(a.history)-1,
                          "runtime_seconds": a.runtime_seconds} for a in attempts],
            "near_optimal_start_posterior_tv_by_region": instability,
            "regions_with_start_disagreement": sum(x > .1 for x in instability),
            "weak_worker_count_rows": int((support < 5).sum()),
            "truth_or_worker_groups_used_for_fit": False}
    write_json(folder / "status.json", info)
    return chosen, info


def as_regions(payload, proposals=True):
    regions = []
    for v in payload["views"]:
        bounds = (0., 0., float(v["width"]), float(v["height"]))
        obs = tuple(Observation(a["worker_id"], a["points"], bounds, v["view_id"]) for a in v["answers"])
        candidates, sources = make_candidates(obs, bounds, proposals)
        regions.append(Region(v["image_id"], v["view_id"], bounds, obs, candidates, sources))
    return tuple(regions)


def generate(phase, condition):
    """Generate once from declared count-error tendencies, not fitted outputs."""
    seed = SEED + (0 if phase == "training" else 1) + (100 if condition == "split_majority" else 0)
    rng = np.random.default_rng(seed)
    workers = [f"frequent_{j+1}" for j in range(7)] + [f"careful_{j+1}" for j in range(3)]
    pattern = [0, 1, 2, 1, 2, 2, 0, 1, 2, 1]
    number = 50 if phase == "training" else 10
    views, truths = [], []
    for i in range(number):
        m = pattern[i % 10] if phase == "training" else (2 if condition == "merge_majority" else 1)
        center = np.array([50., 50.]) + rng.uniform(-5, 5, 2)
        true_points = [] if m == 0 else [center.tolist()] if m == 1 else [(center + [-15, 0]).tolist(), (center + [15, 0]).tolist()]
        answers = []
        for j, worker in enumerate(workers):
            if m == 0:
                n = int(rng.random() < .03)
            elif m == 1:
                probabilities = [.05, .15, .80] if condition == "split_majority" and j < 7 else [.05, .92, .03]
                n = int(rng.choice(3, p=probabilities))
            else:
                probabilities = [.05, .80, .15] if condition == "merge_majority" and j < 7 else [.05, .05, .90]
                n = int(rng.choice(3, p=probabilities))
            if n == 0:
                points = np.empty((0, 2))
            elif n == 1:
                points = center[None] + rng.normal(0, .8, (1, 2))
            else:
                separation = 15 if m == 2 else 6
                points = np.array([center + [-separation, 0], center + [separation, 0]]) + rng.normal(0, .8, (2, 2))
            answers.append({"worker_id": worker, "points": points.tolist()})
        image = f"TRAIN_{i//10+1}.png" if phase == "training" else f"EVAL_{condition}.png"
        views.append({"image_id": image, "view_id": str(i % 10), "width": 100, "height": 100, "answers": answers})
        truths.append({"image_id": image, "view_id": str(i % 10), "count": m,
                       "points": true_points, "center": center.tolist(),
                       "regions": ([{"center": p, "radii": [18, 24]} for p in true_points] if m == 2 else
                                   [{"center": center.tolist(), "radii": [20, 24]}] if m == 1 else [])})
    return {"phase": phase, "worker_ids": workers, "views": views}, {"seed": seed, "regions_for_evaluation_only": truths}


def write_point_table(payload, path):
    rows = []
    for v in payload["views"]:
        for a in v["answers"]:
            for x, y in a["points"]:
                rows.append({"crop_id": v["image_id"] + "/" + v["view_id"], "img_id": v["image_id"],
                    "view_id": v["view_id"], "point_id": len(rows), "worker_id": a["worker_id"], "x": x, "y": y})
    write_csv(path, rows, ["crop_id", "img_id", "view_id", "point_id", "worker_id", "x", "y"])


def synthetic(output, legacy=True, conditions=("merge_majority", "split_majority")):
    old = None
    provenance = {"status": "not_run"}
    if legacy:
        sys.path.insert(0, str(REPORT_SCRIPT_ROOT))
        import run_comparison as old
        provenance = old.check_legacy_archive()
    report = {}
    for condition in conditions:
        inp = output / "synthetic_inputs" / condition
        inp.mkdir(parents=True, exist_ok=False)
        generation_seeds = None
        reused_input_source = None
        if condition in ("merge_majority", "split_majority"):
            train, reference_train = generate("training", condition)
            evaluation, reference_eval = generate("evaluation", condition)
            write_json(inp / "training_reference_not_used_by_methods.json", reference_train)
            write_json(inp / "evaluation_reference_not_used_by_methods.json", reference_eval)
            generation_seeds = {"training": reference_train["seed"], "evaluation": reference_eval["seed"]}
        else:
            if condition == "connected_split":
                saved = ROOT / "comparison_report/connected_split_case/synthetic_inputs/connected_split"
            else:
                saved_name = "A_location" if condition == "stable_70_30" else "D_complementary"
                saved = ROOT / "comparison_report/synthetic_inputs" / saved_name
            train = json.loads((saved / "qc_views.json").read_text())
            evaluation = json.loads((saved / "evaluation_views.json").read_text())
            reused_input_source = str(saved)
        # Filenames match the archived comparison wrapper; the payload is identical
        # for both methods. Empty responses remain in JSON and are audited.
        for name, payload in (("qc", train), ("evaluation", evaluation)):
            write_json(inp / f"{name}_views.json", payload)
            write_point_table(payload, inp / f"{name}_points.csv")
        write_json(inp / "definition.json", {"condition": condition, "generation_seeds": generation_seeds,
            "reused_input_source": reused_input_source,
            "training_images": 5, "training_views": 50, "evaluation_views": 10,
            "new_model_settings": asdict(Settings()), "legacy_settings": ({
                "clustering": asdict(old.HDBSCAN_CONFIG), "refinement": asdict(old.GMM_CONFIG),
                "em": asdict(old.EM_CONFIG), "heatmap": asdict(old.HEATMAP_CONFIG)} if old else None),
            "truth_used_for_fitting": False, "worker_groups_used_for_fitting": False,
            "zero_point_training_answers": sum(not a["points"] for v in train["views"] for a in v["answers"]),
            "legacy_empty_answer_handling": "Point-only table; every worker saw every view. Entirely empty views create no cluster."})
        training_regions, evaluation_regions = as_regions(train), as_regions(evaluation)
        target = output / "results" / condition / "configuration"
        fitted, status = save_fit(target, training_regions)
        save_regions(target / "evaluation_regions.json", evaluation_regions)
        weights, inference_audit = infer(evaluation_regions, fitted)
        summaries = [summarize(r, w) for r, w in zip(evaluation_regions, weights)]
        write_json(target / "evaluation_outputs.json", summaries)
        uniform_data = prepare(evaluation_regions, settings=fitted.settings)
        prior_tables = np.tile((fitted.settings.pseudocounts / fitted.settings.prior_strength)[None], (len(uniform_data.worker_ids), 1, 1))
        uncalibrated, _ = posterior(uniform_data, prior_tables)
        write_json(target / "fixed_error_table_ablation.json", [summarize(r, w) for r, w in zip(evaluation_regions, uncalibrated)])
        observed_only, _ = infer(as_regions(evaluation, proposals=False), fitted)
        write_json(target / "observed_candidates_only_ablation.json",
                   [summarize(r, w) for r, w in zip(as_regions(evaluation, proposals=False), observed_only)])
        record = {"fit": status, "application": inference_audit,
                  "mean_expected_count": float(np.mean([r["expected_count"] for r in summaries])),
                  "structural_identifiability_warning": condition in ("connected_split", "stable_70_30", "complementary_miss")}
        if condition in ("merge_majority", "split_majority"):
            truth = [r["count"] for r in reference_eval["regions_for_evaluation_only"]]
            estimated = [len(r.candidates[int(np.argmax(w))]) for r, w in zip(evaluation_regions, weights)]
            record["synthetic_count_matches"] = int(np.sum(np.array(truth) == estimated))
            record["synthetic_total"] = len(truth)
            record["mean_probability_of_generated_count"] = float(np.mean([s["count_distribution"].get(n, 0.) for s, n in zip(summaries, truth)]))
            record["fixed_error_table_mean_probability_of_generated_count"] = float(np.mean([
                sum(float(p) for s, p in zip(r.candidates, w) if len(s) == n)
                for r, w, n in zip(evaluation_regions, uncalibrated, truth)]))
        if legacy:
            record["legacy"] = old.run_legacy(output, condition)
        report[condition] = record
        write_json(output / "synthetic_report.json", report)
    write_json(output / "legacy_provenance.json", provenance)
    return report


def real(output):
    target = output / "real"
    target.mkdir(parents=True, exist_ok=False)
    workers, outside = select_qc_workers(DEFAULT_DATA_ROOT)
    issues, mismatches = [], []
    views = tuple(v for image in QC_IDS for v in load_image_views(DEFAULT_DATA_ROOT, image, workers,
                  issues=issues, mismatches=mismatches, require_complete=True))
    write_json(target / "input_audit.json", {"issues": issues, "count_mismatches": mismatches,
                                            "point_lists_are_authoritative": True})
    if issues:
        raise ValueError("Invalid source records; see input_audit.json")
    regions, audit = regions_from_views(views)
    write_json(target / "region_audit.json", audit)
    fitted, status = save_fit(target / "training", regions)
    write_json(target / "report.json", {"fit": status, "images": len(QC_IDS), "workers": len(workers),
        "regions": len(regions), "original_answers": sum(len(v.answers) for v in views),
        "region_summary": [{k: v for k, v in a.items() if k not in ("selections", "chain_guard_splits")} for a in audit]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("synthetic", "real"))
    parser.add_argument("--output", type=Path, default=HERE / "outputs/configuration_consensus_v2")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=["merge_majority", "split_majority"],
                        choices=("merge_majority", "split_majority", "connected_split",
                                 "stable_70_30", "complementary_miss"),
                        help="The default conditions are generated locally; the others require private saved fixtures.")
    args = parser.parse_args()
    stage_path = args.output / ("synthetic_inputs" if args.stage == "synthetic" else "real")
    if stage_path.exists():
        raise FileExistsError(f"Existing results preserved; choose a fresh output directory: {stage_path}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"source_snapshot_{args.stage}.json", {
        "new": {name: digest(source_path(name)) for name in ("configuration_model.py", "local_regions.py", "run_configuration_pilot.py")},
        "bundled_support": {name: digest(source_path(name)) for name in ("model.py", "data.py", "run.py", "test_model.py")}})
    if args.stage == "synthetic":
        synthetic(args.output, not args.skip_legacy, args.conditions)
    else:
        real(args.output)


if __name__ == "__main__":
    main()
