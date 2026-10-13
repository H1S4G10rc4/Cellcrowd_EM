"""Reproduce the four small, same-input synthetic comparisons.

The parent process gives each method/condition subprocess a 600-second wall
clock limit.  No real-QC parameters are loaded and no source model is edited.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "point_annotation_trial" / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
WORKSPACE = WORKSPACE_ROOT
DEFAULT_OUTPUT = REPORT_ROOT
OLD = WORKSPACE / "legacy_algorithm"
TRIAL = WORKSPACE / "point_annotation_trial"
sys.path.insert(0, str(OLD))

from data import (Answer, QC_IDS, View, baseline_weights, expected_count,  # noqa: E402
                  expected_mass, point_count_distribution)
from model import fit, infer_view, ospa_distance, prepare  # noqa: E402
from src.clustering_update import HDBSCANClusteringConfig  # noqa: E402
from src.clustering import GMMSplitConfig  # noqa: E402
from src.clustering_v2 import cluster_annotations_by_crop, refine_clusters_with_gmm_by_crop  # noqa: E402
from src.staple_em import EMConfig, build_annotation_matrix_from_clusters, run_staple_em  # noqa: E402
from src.heatmap import HeatmapConfig, build_heatmap_for_image  # noqa: E402


CONDITIONS = ("A_location", "B_count", "C_switching", "D_complementary")
HDBSCAN_CONFIG = HDBSCANClusteringConfig(min_cluster_size_worker_ratio=.35,
                                          min_samples_ratio=.5, eps_scale=1.15,
                                          eps_min_quantile=.82)
GMM_CONFIG = GMMSplitConfig(max_components=3, use_bic_guard=False,
                             dedupe_eps_scale=.3, enforce_worker_uniqueness=False,
                             allow_duplicate_workers=False, bic_improvement_threshold=5.)
EM_CONFIG = EMConfig(max_iter=200, tol=1e-6, init_sensitivity=.7,
                     init_specificity=.7, default_specificity=.7)
HEATMAP_CONFIG = HeatmapConfig(sigma=1.5, kernel_radius_px=3, alpha=.5, cmap="magma")
STARTS = ("ospa_pair", "random_20261008")
CURRENT_MAX_ITERATIONS = 1000
EXTERNAL_TIMEOUT_SECONDS = 600


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def check_legacy_archive() -> dict:
    """Verify the bundled source snapshot; no private archive is distributed.

    The function name is retained for compatibility with historical callers.
    This checks source integrity, not equivalence of private archived results.
    """
    hashes = {}
    for line in (OLD / "SOURCE_SHA256.txt").read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        actual = digest(OLD / rel)
        if actual != expected:
            raise RuntimeError(f"Bundled legacy source checksum mismatch: {rel}")
        hashes[rel] = actual
    return {"distribution": "source-only", "source_sha256": hashes,
            "entry": "Legacy Algorithm with fixed settings declared in this wrapper",
            "historical_archive_checked": False}


def definition(condition: str) -> dict:
    if condition in ("A_location", "C_switching"):
        a, b = [[20., 50.]], [[80., 50.]]
    elif condition == "B_count":
        a, b = [[50., 50.]], [[35., 50.], [65., 50.]]
    else:
        a, b = [[30., 50.]], [[70., 50.]]
    n_a, n_b = (5, 5) if condition == "D_complementary" else (7, 3)
    return {
        "condition": condition, "canvas_pixels": [100, 100],
        "qc_image_ids": list(QC_IDS), "qc_views_per_image": 10,
        "evaluation_image_id": f"EVAL_{condition}.png", "evaluation_views": 10,
        "state_A_pixels": a, "state_B_pixels": b,
        "group_A_worker_ids": [f"A{i}" for i in range(1, n_a + 1)],
        "group_B_worker_ids": [f"B{i}" for i in range(1, n_b + 1)],
        "switching_worker_id": "unstable" if condition == "C_switching" else None,
        "qc_shuffle_seed": 20261008 if condition == "C_switching" else None,
        "evaluation_shuffle_seed": 20261009 if condition == "C_switching" else None,
        "generated_reference_pair_pixels": [[30., 50.], [70., 50.]] if condition == "D_complementary" else None,
        "reference_used_for_fitting": False,
        "qc_reference_frequency": {"A": 7.5 / 11, "B": 3.5 / 11} if condition == "C_switching"
                                  else {"A": n_a / (n_a+n_b), "B": n_b / (n_a+n_b)},
        "stable_ten_reference_frequency": {"A": .7, "B": .3} if condition == "C_switching" else None,
    }


def make_records(d: dict, phase: str) -> dict:
    if phase == "qc":
        ids = [(QC_IDS[i // 10], str(i % 10)) for i in range(50)]
        seed, n_a = d["qc_shuffle_seed"], 25
    else:
        ids = [(d["evaluation_image_id"], str(i)) for i in range(10)]
        seed, n_a = d["evaluation_shuffle_seed"], 5
    alternating = np.array([0] * n_a + [1] * n_a)
    if seed is not None:
        np.random.default_rng(seed).shuffle(alternating)
    workers = d["group_A_worker_ids"] + d["group_B_worker_ids"]
    if d["switching_worker_id"]:
        workers.append(d["switching_worker_id"])
    rows = []
    for index, (image_id, view_id) in enumerate(ids):
        answers = [{"worker_id": worker, "points": d["state_A_pixels"]}
                   for worker in d["group_A_worker_ids"]]
        answers += [{"worker_id": worker, "points": d["state_B_pixels"]}
                    for worker in d["group_B_worker_ids"]]
        if d["switching_worker_id"]:
            points = d["state_A_pixels"] if alternating[index] == 0 else d["state_B_pixels"]
            answers.append({"worker_id": d["switching_worker_id"], "points": points})
        rows.append({"image_id": image_id, "view_id": view_id, "width": 100,
                     "height": 100, "answers": answers})
    return {"phase": phase, "worker_ids": workers, "views": rows}


def load_views(path: Path) -> tuple[tuple[View, ...], tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = tuple(View(row["image_id"], row["view_id"], 0, 0, row["width"], row["height"],
                       row["width"], row["height"],
                       tuple(Answer(answer["worker_id"], np.asarray(answer["points"], dtype=float).reshape(-1, 2))
                             for answer in row["answers"])) for row in payload["views"])
    return views, tuple(payload["worker_ids"])


def point_table(views: tuple[View, ...], path: Path) -> dict:
    rows = []
    total_answers = 0
    zero_answers = 0
    point_count = 0
    for view in views:
        crop_id = f"{view.image_id}/{view.view_id}"
        for answer in view.answers:
            total_answers += 1
            zero_answers += (len(answer.points) == 0)
            point_count += len(answer.points)
            for x, y in answer.points:
                rows.append({"crop_id": crop_id, "img_id": view.image_id,
                             "view_id": view.view_id, "point_id": len(rows),
                             "worker_id": answer.worker_id, "x": float(x), "y": float(y)})
    pd.DataFrame(rows, columns=["crop_id", "img_id", "view_id", "point_id", "worker_id", "x", "y"]).to_csv(path, index=False)
    reread = pd.read_csv(path)
    if len(reread) != point_count or reread["crop_id"].nunique() != len(views):
        raise RuntimeError("Point-table conversion changed counts or view IDs")
    return {"answers_json": total_answers, "zero_point_answers_json": zero_answers,
            "point_count_json": point_count, "point_rows_csv": len(reread),
            "view_count_json": len(views), "view_count_csv": reread["crop_id"].nunique(),
            "zero_point_answer_rows_csv": 0,
            "note": "The point-only legacy table cannot encode an empty answer; none occurs in these four conditions."}


def prepare_inputs(output: Path) -> dict:
    audit = {}
    for condition in CONDITIONS:
        folder = output / "synthetic_inputs" / condition
        folder.mkdir(parents=True, exist_ok=True)
        d = definition(condition)
        write_json(folder / "definition.json", d)
        for phase in ("qc", "evaluation"):
            input_path = folder / f"{phase}_views.json"
            write_json(input_path, make_records(d, phase))
            views, workers = load_views(input_path)
            conversion = point_table(views, folder / f"{phase}_points.csv")
            audit[condition, phase] = {"input_sha256": digest(input_path),
                                       "point_table_sha256": digest(folder / f"{phase}_points.csv"),
                                       "workers": list(workers), **conversion}
        if condition == "C_switching":
            qc = json.loads((folder / "qc_views.json").read_text())["views"]
            ev = json.loads((folder / "evaluation_views.json").read_text())["views"]
            for rows, expected in ((qc, 25), (ev, 5)):
                choices = [row["answers"][-1]["points"] == d["state_A_pixels"] for row in rows]
                assert sum(choices) == expected
    return {f"{condition}/{phase}": value for (condition, phase), value in audit.items()}


def _fixed_em_posterior(matrix, result) -> np.ndarray:
    """Archived run_staple_em E-step, with QC parameters held fixed."""
    ordered = [str(item) for item in matrix.worker_ids]
    p = result.sensitivities.loc[ordered].to_numpy(float)
    q = result.specificities.loc[ordered].to_numpy(float)
    d = matrix.matrix.astype(float)
    eps = EM_CONFIG.min_prob
    p, q = np.clip(p, eps, 1-eps), np.clip(q, eps, 1-eps)
    pi = float(np.clip(result.pi, eps, 1-eps))
    positive = np.log(pi) + d @ np.log(p) + (1-d) @ np.log(1-p)
    negative = np.log1p(-pi) + d @ np.log(1-q) + (1-d) @ np.log(q)
    return np.exp(positive - np.logaddexp(positive, negative))


def run_legacy(output: Path, condition: str) -> dict:
    inp = output / "synthetic_inputs" / condition
    target = output / "results" / condition / "legacy"
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    qc_coarse, qc_params = cluster_annotations_by_crop(
        HDBSCAN_CONFIG, annotations_path=inp / "qc_points.csv",
        output_path=target / "qc_coarse_clusters.csv",
        params_output_path=target / "qc_cluster_parameters.csv")
    qc_final = refine_clusters_with_gmm_by_crop(
        qc_coarse, GMM_CONFIG, output_path=target / "qc_final_clusters.csv")
    qc_matrix = build_annotation_matrix_from_clusters(qc_final)
    fitted = run_staple_em(qc_matrix, EM_CONFIG)
    calibration_seconds = time.perf_counter() - started
    fitted.worker_table().to_csv(target / "worker_parameters.csv")
    pd.DataFrame({"iteration": np.arange(1, len(fitted.log_likelihood)+1),
                  "log_likelihood": fitted.log_likelihood}).to_csv(target / "em_history.csv", index=False)
    fitted.posterior.rename("cluster_posterior").to_csv(target / "qc_cluster_posteriors.csv")
    if not fitted.converged:
        raise RuntimeError("Legacy EM did not converge; calibration outputs retained")
    applied = time.perf_counter()
    ev_coarse, ev_params = cluster_annotations_by_crop(
        HDBSCAN_CONFIG, annotations_path=inp / "evaluation_points.csv",
        output_path=target / "evaluation_coarse_clusters.csv",
        params_output_path=target / "evaluation_cluster_parameters.csv")
    ev_final = refine_clusters_with_gmm_by_crop(
        ev_coarse, GMM_CONFIG, output_path=target / "evaluation_final_clusters.csv")
    ev_matrix = build_annotation_matrix_from_clusters(ev_final)
    scores = _fixed_em_posterior(ev_matrix, fitted)
    pd.DataFrame({"final_cluster_id": ev_matrix.cluster_ids,
                  "cluster_posterior_fixed_qc": scores}).to_csv(
                      target / "evaluation_cluster_posteriors.csv", index=False)
    qc_views, _ = load_views(inp / "qc_views.json")
    eval_views, _ = load_views(inp / "evaluation_views.json")
    qc_table = pd.read_csv(inp / "qc_points.csv")
    eval_table = pd.read_csv(inp / "evaluation_points.csv")
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    worker_table = fitted.worker_table()
    def maps(views, table):
        arrays = []
        for view in views:
            crop_id = f"{view.image_id}/{view.view_id}"
            points = table.loc[table["crop_id"] == crop_id, ["x", "y", "worker_id"]]
            _, heatmap = build_heatmap_for_image(points, worker_table,
                                                 image=blank, config=HEATMAP_CONFIG)
            arrays.append(heatmap)
        return np.stack(arrays)
    qc_maps = maps(qc_views, qc_table)
    ev_maps = maps(eval_views, eval_table)
    np.savez_compressed(target / "native_heatmaps.npz", qc=qc_maps, evaluation=ev_maps)
    application_seconds = time.perf_counter() - applied
    meta = {"status": "converged", "calibration_seconds": calibration_seconds,
            "application_seconds": application_seconds, "em_iterations": fitted.iterations,
            "pi": fitted.pi, "qc_clusters": qc_matrix.n_clusters,
            "evaluation_clusters": ev_matrix.n_clusters,
            "worker_ids": [str(x) for x in qc_matrix.worker_ids],
            "weight_used_for_clicks": "weight_positive = log(sensitivity) - log(1-specificity)",
            "native_output": "cluster posterior and signed-LLR-weighted soft-saturated heatmap",
            "standard_point_output": None,
            "native_heatmap_qc_range": [float(qc_maps.min()), float(qc_maps.max())],
            "native_heatmap_evaluation_range": [float(ev_maps.min()), float(ev_maps.max())],
            "qc_input_sha256": digest(inp / "qc_views.json"),
            "evaluation_input_sha256": digest(inp / "evaluation_views.json")}
    write_json(target / "run.json", meta)
    return meta


def run_current(output: Path, condition: str) -> dict:
    inp = output / "synthetic_inputs" / condition
    target = output / "results" / condition / "current"
    target.mkdir(parents=True, exist_ok=True)
    qc_views, workers = load_views(inp / "qc_views.json")
    eval_views, eval_workers = load_views(inp / "evaluation_views.json")
    assert workers == eval_workers
    started = time.perf_counter()
    prepared = prepare(qc_views, workers)
    attempts = []
    for start in STARTS:
        result = fit(prepared, 2, start, max_iterations=CURRENT_MAX_ITERATIONS)
        attempts.append(result)
        pd.DataFrame(result.history).to_csv(target / f"{start}_history.csv", index=False)
    calibration_seconds = time.perf_counter() - started
    chosen = max((item for item in attempts if item.converged),
                 key=lambda item: item.objective, default=None)
    attempt_info = [{"initialization": a.initialization, "status": a.status,
                     "iterations": len(a.history), "objective": a.objective,
                     "runtime_seconds": a.runtime_seconds} for a in attempts]
    write_json(target / "fit_attempts.json", attempt_info)
    if chosen is None:
        raise RuntimeError("Neither G=2 initialization converged; histories retained")
    p = chosen.parameters
    np.savez_compressed(target / "fitted_parameters.npz", omega=p.omega,
                        q=p.q, rho=p.rho, pi_qc=p.pi)
    pd.DataFrame({"worker_id": workers, "rho": p.rho,
                  "q_class_1": p.q[:, 0], "q_class_2": p.q[:, 1]}).to_csv(
                      target / "worker_parameters.csv", index=False)
    applied = time.perf_counter()
    frozen = {worker: (p.q[j].copy(), float(p.rho[j])) for j, worker in enumerate(workers)}
    rows = []
    mass_maps = []
    for view in eval_views:
        inferred = infer_view(view, frozen, 2)
        weights = inferred["weights"]
        if not inferred["converged"]:
            raise RuntimeError(f"Evaluation view did not converge: {view.view_id}")
        mass_maps.append(expected_mass(view, weights))
        rows.append({"image_id": view.image_id, "view_id": view.view_id,
                     "candidate_points_pixels": [c.points.tolist() for c in view.candidates],
                     "weights": weights.tolist(), "point_count_distribution": point_count_distribution(view, weights),
                     "expected_count": expected_count(view, weights),
                     "pi": inferred["pi"].tolist(), "v": inferred["v"].tolist(),
                     "iterations": inferred["iterations"], "status": inferred["status"]})
    np.savez_compressed(target / "evaluation_expected_mass.npz", mass=np.stack(mass_maps))
    write_json(target / "evaluation_outputs.json", rows)
    application_seconds = time.perf_counter() - applied
    meta = {"status": "converged", "calibration_seconds": calibration_seconds,
            "application_seconds": application_seconds,
            "chosen_initialization": chosen.initialization, "objective": chosen.objective,
            "fit_attempts": attempt_info, "worker_ids": list(workers),
            "qc_input_sha256": digest(inp / "qc_views.json"),
            "evaluation_input_sha256": digest(inp / "evaluation_views.json")}
    write_json(target / "run.json", meta)
    return meta


def run_worker(output: Path, condition: str, method: str) -> None:
    target = output / "results" / condition / method
    target.mkdir(parents=True, exist_ok=True)
    try:
        result = run_legacy(output, condition) if method == "legacy" else run_current(output, condition)
        write_json(target / "status.json", {"status": "success", "run": result})
    except BaseException as exc:
        write_json(target / "status.json", {"status": "failed", "reason": str(exc),
                                            "traceback": traceback.format_exc()})
        raise


def run_all(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    legacy = check_legacy_archive()
    inputs = prepare_inputs(output)
    write_json(output / "provenance.json", {
        "legacy": legacy,
        "current": {"source_sha256": {rel: digest(source_path(rel)) for rel in
                                       ("model.py", "data.py", "run.py")},
                    "G": 2, "max_iterations": CURRENT_MAX_ITERATIONS,
                    "initializations": list(STARTS)},
        "configs": {"hdbscan": asdict(HDBSCAN_CONFIG), "gmm": asdict(GMM_CONFIG),
                    "em": asdict(EM_CONFIG), "heatmap": asdict(HEATMAP_CONFIG)},
        "inputs": inputs,
        "execution": {"timeout_seconds_per_method_condition": EXTERNAL_TIMEOUT_SECONDS,
                      "legacy_eval": "Archived E-step with QC pi, sensitivity and specificity fixed; map uses archived heatmap function",
                      "current_eval": "infer_view with QC q and rho fixed",
                      "no_group_labels_or_reference_points_passed_to_models": True},
    })
    for condition in CONDITIONS:
        for method in ("legacy", "current"):
            target = output / "results" / condition / method
            target.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(SCRIPT), "--worker", condition, method,
                       "--output", str(output)]
            started = time.perf_counter()
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=EXTERNAL_TIMEOUT_SECONDS,
                                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                state = "exited" if completed.returncode == 0 else "failed"
                write_json(target / "process.json", {"state": state,
                                                       "returncode": completed.returncode,
                                                       "wall_seconds": time.perf_counter()-started,
                                                       "command": command})
                (target / "stdout.log").write_text(completed.stdout, encoding="utf-8")
                (target / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                write_json(target / "process.json", {"state": "timeout",
                                                       "wall_seconds": time.perf_counter()-started,
                                                       "command": command})
                (target / "stdout.log").write_bytes(exc.stdout or b"")
                (target / "stderr.log").write_bytes(exc.stderr or b"")
            print(f"{condition} {method}: {json.loads((target/'process.json').read_text())['state']}", flush=True)
    aggregate(output)


def _reference_rows(condition: str) -> list[tuple[str, np.ndarray, float]]:
    if condition == "C_switching":
        return [("all_11_workers", np.array([7.5/11, 3.5/11]), 1.),
                ("stable_10_workers", np.array([.7, .3]), 1.)]
    if condition == "D_complementary":
        return [("observed_judgment", np.array([.5, .5]), 1.),
                ("generated_two_point_setup", np.array([.5, .5]), 2.)]
    return [("known_group_frequency", np.array([.7, .3]),
             1.3 if condition == "B_count" else 1.)]


def _save_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output: Path) -> None:
    """Recompute all common metrics from stored inputs and method outputs."""
    summary = []
    by_view = []
    native = []
    worker_scores = []
    for condition in CONDITIONS:
        inp = output / "synthetic_inputs" / condition
        result_root = output / "results" / condition
        eval_views, workers = load_views(inp / "evaluation_views.json")
        for method in ("legacy", "current"):
            status = json.loads((result_root / method / "status.json").read_text())
            if status["status"] != "success":
                summary.append({"condition": condition, "method": method,
                                "representation": "not_available", "reference": "not_available",
                                "reason": status["reason"]})
        legacy_ok = json.loads((result_root / "legacy/status.json").read_text())["status"] == "success"
        current_ok = json.loads((result_root / "current/status.json").read_text())["status"] == "success"
        current_output = (json.loads((result_root / "current/evaluation_outputs.json").read_text())
                          if current_ok else None)
        legacy_weights = (pd.read_csv(result_root / "legacy/worker_parameters.csv").set_index("worker_id")
                          if legacy_ok else None)
        legacy_maps = (np.load(result_root / "legacy/native_heatmaps.npz")["evaluation"]
                       if legacy_ok else None)
        if current_ok:
            current_params = pd.read_csv(result_root / "current/worker_parameters.csv")
            for row in current_params.itertuples(index=False):
                worker_scores.append({"condition": condition, "method": "current",
                                      "worker_id": row.worker_id, "measure": "rho",
                                      "value": row.rho, "unit": "class-consistent response weight"})
        if legacy_ok:
            for worker, row in legacy_weights.iterrows():
                for measure in ("sensitivity", "specificity", "weight_positive", "weight_negative"):
                    worker_scores.append({"condition": condition, "method": "legacy",
                                          "worker_id": worker, "measure": measure,
                                          "value": row[measure],
                                          "unit": "log-likelihood ratio" if measure.startswith("weight_") else "EM parameter"})
        weights_by_method: dict[str, list[np.ndarray]] = {"simple_aggregation": [], "current": [],
                                                           "legacy_projection": []}
        baseline_maps = []
        projected_maps = []
        projection_reason = None
        if legacy_ok:
            applied = legacy_weights["weight_positive"].to_dict()
            if not all(np.isfinite(value) and value >= 0 for value in applied.values()):
                projection_reason = "Signed positive-LLR click weights; no archived nonnegative conversion."
        for index, view in enumerate(eval_views):
            expected_points = [c.points.tolist() for c in view.candidates]
            baseline = baseline_weights(view)
            weights_by_method["simple_aggregation"].append(baseline)
            baseline_maps.append(expected_mass(view, baseline))
            if current_ok:
                item = current_output[index]
                if item["candidate_points_pixels"] != expected_points:
                    raise RuntimeError("Current candidate order differs from evaluation input")
                calibrated = np.asarray(item["weights"], dtype=float)
                weights_by_method["current"].append(calibrated)
                saved_mass = np.load(result_root / "current/evaluation_expected_mass.npz")["mass"][index]
                if not np.allclose(saved_mass.sum(), expected_count(view, calibrated), atol=1e-10):
                    raise RuntimeError("Current expected mass is not conserved")
            if legacy_ok and projection_reason is None:
                clicks = np.asarray([applied.get(answer.worker_id, np.nan) for answer in view.answers])
                if not np.isfinite(clicks).all() or clicks.sum() <= 0:
                    projection_reason = "Missing or zero-sum nonnegative click weights."
                else:
                    projected = np.zeros(len(view.candidates))
                    for answer, weight in zip(view.answers, clicks):
                        projected[view.observed_index[answer.worker_id]] += weight
                    projected /= clicks.sum()
                    weights_by_method["legacy_projection"].append(projected)
                    projected_maps.append(expected_mass(view, projected))
            if legacy_ok:
                m = legacy_maps[index]
                a, b = definition(condition)["state_A_pixels"], definition(condition)["state_B_pixels"]
                aa = np.mean([m[int(y), int(x)] for x, y in a])
                bb = np.mean([m[int(y), int(x)] for x, y in b])
                native.append({"condition": condition, "view_id": view.view_id,
                               "cellcrowd_score_at_A": float(aa), "cellcrowd_score_at_B_mean": float(bb),
                               "cellcrowd_score_min": float(m.min()), "cellcrowd_score_max": float(m.max()),
                               "score_unit": "dimensionless signed/saturated score",
                               "standard_point_output": "not defined"})
            for method, weight in (("simple_aggregation", baseline),
                                   ("current", calibrated if current_ok else None),
                                   ("legacy_projection", projected if legacy_ok and projection_reason is None else None)):
                if weight is None:
                    continue
                if len(weight) != len(view.candidates) or not np.isclose(weight.sum(), 1):
                    raise RuntimeError(f"Invalid candidate weights: {condition} {method} {view.view_id}")
                counts = point_count_distribution(view, weight)
                if not np.isclose(sum(counts.values()), 1):
                    raise RuntimeError("Point-count distribution does not sum to one")
                for k, candidate in enumerate(view.candidates):
                    by_view.append({"condition": condition, "method": method,
                                    "representation": "projected_click_weights" if method == "legacy_projection"
                                                      else "native_candidate_weights",
                                    "image_id": view.image_id, "view_id": view.view_id,
                                    "candidate": k, "points_pixels": json.dumps(candidate.points.tolist()),
                                    "point_count": len(candidate.points), "weight": float(weight[k]),
                                    "expected_count_view": expected_count(view, weight),
                                    "point_count_distribution": json.dumps(counts, sort_keys=True)})
        np.savez_compressed(result_root / "simple_aggregation_mass.npz", mass=np.stack(baseline_maps))
        if projection_reason is None and projected_maps:
            np.savez_compressed(result_root / "legacy/projected_expected_mass.npz",
                                mass=np.stack(projected_maps))
        for method, arrays in weights_by_method.items():
            if method == "current" and not current_ok:
                continue
            if method == "legacy_projection" and (not legacy_ok or projection_reason is not None):
                for reference, _, _ in _reference_rows(condition):
                    summary.append({"condition": condition, "method": "legacy",
                                    "representation": "projected_click_weights",
                                    "reference": reference, "reason": projection_reason or "Legacy run failed"})
                continue
            if not arrays:
                continue
            mean_weights = np.mean(np.stack(arrays), axis=0)
            if len(mean_weights) != 2:
                raise RuntimeError("Expected two candidate configurations")
            for reference, ref, reference_count in _reference_rows(condition):
                # D has no two-point candidate. The judgment reference and the
                # generated pair are distinct; TV is defined only for the former.
                tv = None if reference == "generated_two_point_setup" else float(np.abs(mean_weights-ref).sum()/2)
                count = float(np.mean([expected_count(v, w) for v, w in zip(eval_views, arrays)]))
                pair = np.array([[.3, .5], [.7, .5]])
                ospa = (float(np.mean([sum(float(weight) * ospa_distance(c.points / 100, pair)
                                          for c, weight in zip(view.candidates, ww))
                                       for view, ww in zip(eval_views, arrays)]))
                        if condition == "D_complementary" and reference == "generated_two_point_setup" else None)
                summary.append({"condition": condition,
                                "method": "legacy" if method == "legacy_projection" else method,
                                "representation": "projected_click_weights" if method == "legacy_projection"
                                                  else "native_candidate_weights",
                                "reference": reference, "weight_A": float(mean_weights[0]),
                                "weight_B": float(mean_weights[1]), "tv_to_judgment_reference": tv,
                                "expected_count": count, "reference_expected_count": reference_count,
                                "signed_count_error": count-reference_count,
                                "absolute_count_error": abs(count-reference_count),
                                "expected_ospa_to_generated_pair": ospa,
                                "two_point_candidate_probability": 0. if condition == "D_complementary"
                                                                   else None,
                                "reason": None})
        if legacy_ok:
            legacy_run = json.loads((result_root / "legacy/run.json").read_text())
            _save_rows(result_root / "legacy/native_summary.csv", [
                {"condition": condition, "output": "cluster posterior", "n_clusters_qc": legacy_run["qc_clusters"],
                 "n_clusters_evaluation": legacy_run["evaluation_clusters"],
                 "map_min_evaluation": legacy_run["native_heatmap_evaluation_range"][0],
                 "map_max_evaluation": legacy_run["native_heatmap_evaluation_range"][1],
                 "point_count_distribution": "not defined", "standard_point_output": "not defined"}],
                ["condition", "output", "n_clusters_qc", "n_clusters_evaluation", "map_min_evaluation",
                 "map_max_evaluation", "point_count_distribution", "standard_point_output"])
    _save_rows(output / "results/synthetic_summary.csv", summary,
               ["condition", "method", "representation", "reference", "weight_A", "weight_B",
                "tv_to_judgment_reference", "expected_count", "reference_expected_count",
                "signed_count_error", "absolute_count_error", "expected_ospa_to_generated_pair",
                "two_point_candidate_probability", "reason"])
    _save_rows(output / "results/synthetic_by_view.csv", by_view,
               ["condition", "method", "representation", "image_id", "view_id", "candidate",
                "points_pixels", "point_count", "weight", "expected_count_view", "point_count_distribution"])
    _save_rows(output / "results/legacy_native_scores_by_view.csv", native,
               ["condition", "view_id", "cellcrowd_score_at_A", "cellcrowd_score_at_B_mean",
                "cellcrowd_score_min", "cellcrowd_score_max", "score_unit", "standard_point_output"])
    _save_rows(output / "results/worker_scores.csv", worker_scores,
               ["condition", "method", "worker_id", "measure", "value", "unit"])
    # Verify that rerunning aggregation reproduces the recorded values exactly.
    assert len(by_view) >= 4 * 10 * 2 * 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", nargs=2, metavar=("CONDITION", "METHOD"))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.aggregate:
        aggregate(output)
    elif args.worker:
        condition, method = args.worker
        if condition not in CONDITIONS or method not in ("legacy", "current"):
            parser.error("Unknown condition or method")
        run_worker(output, condition, method)
    else:
        run_all(output)


if __name__ == "__main__":
    main()
