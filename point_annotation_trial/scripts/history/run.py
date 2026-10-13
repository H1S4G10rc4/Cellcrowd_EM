"""Run the fixed artificial checks, QC prediction, final fit and gated images."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _ensure_dependencies() -> None:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        environment = PROJECT_ROOT / ".venv"
        if not environment.exists():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        executable = environment / "bin" / "python"
        subprocess.run([str(executable), "-m", "pip", "install", "--disable-pip-version-check",
                        "numpy", "scipy", "Pillow"], check=True)
        os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_dependencies()

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from data import (DEFAULT_DATA_ROOT, QC_IDS, Answer, View, baseline_weights,  # noqa: E402
                  load_image_views, next_output_dir, resolve_data_root, save_image_outputs,
                  select_qc_workers, write_csv, write_json)
from model import (BANDWIDTH, MAX_ITERATIONS, MAX_SECONDS_PER_START, Q_COUNT,  # noqa: E402
                   UNIFORM_FRACTION, Fit, fit, full_objective, infer_view, load_fit_npz,
                   log_f_for_answer, predictive_from_log_f, prepare, save_fit_npz,
                   save_qc_inputs)
from test_model import run_eight_checks  # noqa: E402

QC_MAX_ITERATIONS = 1000
PREVIOUS_REPORT = PROJECT_ROOT / "outputs" / "trial" / "report.json"


SOURCE_FILES = [str(source_path(name).relative_to(PROJECT_ROOT)) for name in
                ("data.py", "model.py", "run.py", "test_model.py", "README.md")]
LIMITATIONS = [
    "rhoは生物学的な正確さの確率ではない。",
    "rhoとpiには同じ観測尤度を与える組合せがあり、絶対尺度には事前分布が関与する。",
    "今回はこの非識別性を解消せず、同じモデルの数値収束と出力を確認した。",
    "保留回答の予測改善は、真の細胞数や検出器性能の改善を示さない。",
    "固定観測カーネル、最大2クラス、条件付きMAPは今回の試作上の仮定である。",
    "候補は実回答から作るため、推定はその候補集合に条件付けられる。",
    "同じ画像の重複view間の依存性は完全にはモデル化していない。",
    "候補サンプルは、未観測の補完的な和集合を新しく生成しない。",
    "生成モデルや検出器の学習は含まない。",
]
KERNEL = {"q_count": Q_COUNT, "h": BANDWIDTH, "uniform_fraction": UNIFORM_FRACTION,
          "count_support": "all nonnegative integers", "coordinate_domain": "[0,1]^2",
          "density_convention": "count mass times ordered coordinate density; same for every method"}
PRIORS = {"omega": "Dirichlet(2,...,2)", "rho": "Beta(2,2)",
          "pi": "Dirichlet(1+1/K,...,1+1/K)", "pi_MAP_added_pseudocount_sum": 1.0,
          "pi_Dirichlet_parameter_sum": "K+1"}


def _announce(**message) -> None:
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), flush=True)


def _report_sections(report: dict) -> list[dict]:
    prediction = report.get("qc_prediction") or {}
    return [
        {"number": 1, "title": "変更ファイルと実行コマンド",
         "changed_files": ["model.py", "run.py", "test_model.py", "README.md"],
         "unchanged_input_reader": "data.py", "command": report["command"],
         "output_dir": report["output_dir"]},
        {"number": 2, "title": "QC範囲", "qc_scope": report.get("qc_scope"),
         "input_audit": report.get("input_audit")},
        {"number": 3, "title": "8種類の検査", "checks": report.get("checks")},
        {"number": 4, "title": "初期値別の終了状態と変化量",
         "qc_evaluation_attempts": prediction.get("attempts"),
         "qc_final_attempts": (report.get("final_qc") or {}).get("attempts")},
        {"number": 5, "title": "保留予測と前回比較・選択理由",
         "scores": {key: prediction.get(key) for key in
                    ("overall_mean_log_density", "image_mean_log_density", "previous_run_comparison",
                     "comparison_G2_minus_G1", "comparison_G1_minus_baseline", "comparison_G2_minus_baseline",
                     "selected_model", "selection_reason", "adoption_condition_met")}},
        {"number": 6, "title": "全QC最終推定", "final_qc": report.get("final_qc")},
        {"number": 7, "title": "非QC1枚と固定値不変確認", "non_qc": report.get("non_qc")},
        {"number": 8, "title": "保存物の数値再計算", "saved_result_recalculation": report.get("saved_result_recalculation"),
         "qc_inputs": report.get("qc_inputs")},
        {"number": 9, "title": "限界", "limitations": LIMITATIONS},
    ]


def _markdown(report: dict) -> str:
    scope = report.get("qc_scope") or {}
    checks = report.get("checks") or {}
    prediction = report.get("qc_prediction") or {}
    final = report.get("final_qc") or {}
    non_qc = report.get("non_qc") or {}
    lines = ["# 点アノテーション統合：保存と収束確認", "",
             "## 1. 変更ファイルと実行コマンド", "",
             "`model.py`、`run.py`、`test_model.py`、`README.md`を変更。`data.py`は変更せず使用。",
             f"`{report['command']}`", "",
             "## 2. QC範囲", "",
             f"QC1–5、{scope.get('number_of_workers', '未確認')}人、{scope.get('number_of_views', '未確認')} view。"
             f"学習{scope.get('number_of_training_answers', '未確認')}件、"
             f"保留{(prediction.get('heldout_answer_count') or scope.get('number_of_heldout_answers_expected', '未確認'))}件、"
             f"全QC{scope.get('number_of_answers', '未確認')}件、非QC混入{scope.get('non_qc_answers_in_fit', '未確認')}件。", "",
             "## 3. 8種類の検査", ""]
    lines.extend(f"- {name}: {'合格' if result.get('passed') else '未達'}" for name, result in checks.items())
    lines.extend(["", "## 4. 初期値別の終了状態", ""])
    for stage, attempts in (("評価", prediction.get("attempts") or {}),
                            ("全QC", {f"G{final.get('G')}": final.get("attempts") or []} if final.get("attempts") else {})):
        for label, records in attempts.items():
            for entry in records:
                lines.append(f"- {stage} {label}/{entry['initialization']}: {entry['status']}、"
                             f"{entry.get('last_accepted_iteration', 0)}反復、{entry.get('runtime_seconds')}秒、"
                             f"objective={entry.get('objective')}、最終変化量={entry.get('final_deltas')}、"
                             f"最大項={entry.get('largest_final_change')}。")
    if not prediction.get("attempts"):
        lines.append("未実施。")
    lines.extend(["", "## 5. 保留予測と選択", ""])
    overall = prediction.get("overall_mean_log_density") or {}
    lines.append(f"画像別平均の5画像平均: 単純集計={overall.get('simple_aggregation')}、"
                 f"G=1={overall.get('G1')}、G=2={overall.get('G2')}。")
    previous = prediction.get("previous_run_comparison") or {}
    lines.append(f"前回比較: {previous.get('file', previous.get('status', '未実施'))}。"
                 "前回の未収束スコアは今回の選択に使わない。")
    lines.append(f"選択={prediction.get('selected_model')}、"
                 f"採用条件={'成立' if prediction.get('adoption_condition_met') else '不成立'}。"
                 f"理由: {prediction.get('selection_reason', '未評価')}。")
    lines.extend(["", "## 6. 全QC最終推定", "",
                  f"状態={final.get('status', '未実施')}。"
                  f"G={final.get('G', '対象外')}、収束={'はい' if final.get('converged') else 'いいえ'}。", "",
                  "## 7. 非QC1枚", "",
                  f"状態={non_qc.get('status', '未実施')}。件数={non_qc.get('counts')}。"
                  f"固定値の前後一致={'はい' if non_qc.get('fixed_parameter_file_unchanged') else '対象外'}。"])
    comparison = (non_qc.get("one_image") or {}).get("comparison_file")
    if comparison:
        summary = (non_qc.get("one_image") or {}).get("comparison_summary") or {}
        lines.append(f"view別比較: `{comparison}`。")
        lines.append(f"比較可能{summary.get('comparable_views')} viewの平均期待点数は"
                     f"単純集計{summary.get('mean_baseline_expected_count')}、"
                     f"校正出力{summary.get('mean_calibrated_expected_count')}。"
                     f"候補重みの全変動距離のview平均は{summary.get('mean_weight_total_variation')}。")
    recalc = report.get("saved_result_recalculation") or {}
    lines.extend(["", "## 8. 保存物からの再計算", ""])
    for stage in ("qc_evaluation", "qc_final"):
        checked = recalc.get(stage)
        if checked:
            maximum = max((item["absolute_difference"] for item in checked["checked_initializations"]), default=None)
            score = checked.get("heldout_answer_scores")
            lines.append(f"- {stage}: {checked['status']}、objective最大絶対差={maximum}。"
                         + (f"保留{score['rows']}スコアの最大絶対差={score['maximum_absolute_difference']}。" if score else ""))
    if not recalc:
        lines.append("未実施。")
    lines.extend(["", "## 9. 限界", ""])
    lines.extend(f"- {item}" for item in LIMITATIONS)
    lines.extend(["", f"出力先: `{report['output_dir']}`。"])
    return "\n".join(lines) + "\n"


def _save_report(output: Path, report: dict) -> None:
    write_json(output / "report.json", {"sections": _report_sections(report),
                                        **{key: value for key, value in report.items() if key != "sections"}})
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")


def _history_callback(output: Path, stage: str, label: str):
    path = output / stage / f"{label}_history.csv"
    fields = ["iteration", "objective", "relative_objective_change", "maximum_parameter_change",
              "max_delta_omega", "max_delta_q", "max_delta_rho", "max_delta_pi",
              "stable_iterations", "accepted", "runtime_seconds", "reason"]

    def callback(row: dict, history: list[dict]) -> None:
        if row["iteration"] % 20 == 0 or not row["accepted"]:
            write_csv(path, history, fields)
            _announce(stage=stage, initialization=label, iteration=row["iteration"],
                      objective=row["objective"], maximum_parameter_change=row.get("maximum_parameter_change"),
                      runtime_seconds=row.get("runtime_seconds"))

    return callback, path, fields


def _fit_info(fitted: Fit) -> dict:
    last = next((row for row in reversed(fitted.history) if row.get("accepted")), None)
    final_deltas = {name: last.get(name) if last else None for name in
                    ("max_delta_omega", "max_delta_q", "max_delta_rho", "max_delta_pi")}
    limiting = max(final_deltas, key=lambda name: final_deltas[name]) if last else None
    return {"G": fitted.G, "initialization": fitted.initialization, "status": fitted.status,
            "converged": fitted.converged, "objective": fitted.objective,
            "iterations": len(fitted.history), "last_accepted_iteration": last["iteration"] if last else 0,
            "stable_iterations": last["stable_iterations"] if last else 0,
            "final_deltas": final_deltas, "largest_final_change": limiting,
            "runtime_seconds": fitted.runtime_seconds,
            "representative_workers": fitted.representative_workers}


def _fit_attempts(prepared, G: int, stage: str, output: Path) -> tuple[Fit | None, list[dict], list[Fit]]:
    starts = ("one",) if G == 1 else ("ospa_pair", "random_20261008")
    attempts: list[Fit] = []
    summaries: list[dict] = []
    for start in starts:
        label = f"G{G}_{start}"
        callback, path, fields = _history_callback(output, stage, label)
        try:
            result = fit(prepared, G, start, on_iteration=callback, max_iterations=QC_MAX_ITERATIONS)
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            record = {"G": G, "initialization": start, "status": "numerical_failure",
                      "converged": False, "reason": str(exc), "history_file": str(path.relative_to(output))}
            summaries.append(record)
            write_json(output / stage / f"{label}_failure.json", record)
            _announce(stage=stage, initialization=label, status="numerical_failure", reason=str(exc))
            continue
        write_csv(path, result.history, fields)
        attempts.append(result)
        record = _fit_info(result)
        record["history_file"] = str(path.relative_to(output))
        summaries.append(record)
        _announce(stage=stage, initialization=label, status=result.status,
                  iterations=len(result.history), objective=result.objective)
    converged = [result for result in attempts if result.converged]
    selected = max(converged, key=lambda item: item.objective) if converged else None
    if selected is None:
        usable = [result for result in attempts if result.status != "numerical_failure"]
        selected = max(usable, key=lambda item: item.objective) if usable else None
    if selected is not None:
        p = selected.parameters
        write_json(output / stage / f"G{G}_evaluation_parameters.json", {
            **_fit_info(selected), "worker_ids": list(prepared.worker_ids),
            "omega": p.omega.tolist(),
            "workers": {worker: {"q": p.q[j].tolist(), "rho": float(p.rho[j])}
                        for j, worker in enumerate(prepared.worker_ids)},
            "kernel": KERNEL, "priors": PRIORS,
            "parameter_use": "evaluation only" if stage == "qc_evaluation" else "not yet adopted",
        })
    return selected, summaries, attempts


def _heldout_split(worker_ids: tuple[str, ...]) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    held = {name: tuple(worker for j, worker in enumerate(worker_ids) if j % 5 == index)
            for index, name in enumerate(QC_IDS)}
    trained = {name: tuple(worker for worker in worker_ids if worker not in set(held[name])) for name in QC_IDS}
    return trained, held


def _read_qc_views(root: Path, workers_by_image: dict[str, tuple[str, ...]], issues: list[dict],
                   mismatches: list[dict]) -> tuple[View, ...]:
    return tuple(view for name in QC_IDS for view in load_image_views(
        root, name, workers_by_image[name], issues, mismatches, require_complete=True))


def _score_heldout(
    root: Path, train: tuple[View, ...], held_ids: dict[str, tuple[str, ...]],
    workers: tuple[str, ...], models: dict[str, Fit | None], output: Path,
    issues: list[dict], mismatches: list[dict],
) -> tuple[dict, tuple[View, ...]]:
    """Read held-out JSON only now, after candidate construction and fitting."""
    training = {(view.image_id, view.view_id): (r, view) for r, view in enumerate(train)}
    lookup = {worker: j for j, worker in enumerate(workers)}
    rows = []
    all_held_views: list[View] = []
    for name in QC_IDS:
        held_views = load_image_views(root, name, held_ids[name], issues, mismatches, require_complete=True)
        all_held_views.extend(held_views)
        for held in held_views:
            if (name, held.view_id) not in training:
                issues.append({"image_id": name, "view_id": held.view_id, "reason": "heldout_view_not_in_training"})
                continue
            r, view = training[(name, held.view_id)]
            assert not set(held.observed_index).intersection(view.observed_index)
            for answer in held.answers:
                log_f = log_f_for_answer(view, answer)
                baseline = float(logsumexp(np.log(baseline_weights(view)) + log_f))
                scores = {"simple_aggregation": baseline}
                for label, model in models.items():
                    if model is not None:
                        scores[label] = predictive_from_log_f(log_f, model, lookup[answer.worker_id], r)
                for label, score in scores.items():
                    rows.append({"image_id": name, "view_id": view.view_id,
                                 "worker_id": answer.worker_id, "method": label,
                                 "log_predictive_density": score})
    write_csv(output / "qc_evaluation" / "heldout_answer_scores.csv", rows)
    labels = ("simple_aggregation", "G1", "G2")
    by_image = {label: {} for label in labels}
    by_worker = []
    for label in labels:
        subset = [row for row in rows if row["method"] == label]
        if not subset:
            continue
        for name in QC_IDS:
            values = [row["log_predictive_density"] for row in subset if row["image_id"] == name]
            by_image[label][name] = float(np.mean(values)) if values else None
        for worker in workers:
            values = [row["log_predictive_density"] for row in subset if row["worker_id"] == worker]
            by_worker.append({"worker_id": worker, "method": label,
                              "number_of_heldout_answers": len(values),
                              "mean_log_predictive_density": float(np.mean(values)) if values else None})
    write_csv(output / "qc_evaluation" / "heldout_worker_means.csv", by_worker)
    image_rows = [{"method": label, "image_id": name, "mean_log_predictive_density": value}
                  for label, images in by_image.items() for name, value in images.items()]
    write_csv(output / "qc_evaluation" / "heldout_image_means.csv", image_rows)
    overall = {label: float(np.mean(list(images.values()))) for label, images in by_image.items() if images}
    expected_baseline_answers = sum(len(held_ids[name]) for name in QC_IDS) * 49
    actual_baseline_answers = len([row for row in rows if row["method"] == "simple_aggregation"])
    if actual_baseline_answers != expected_baseline_answers:
        issues.append({"reason": "heldout_answer_count_mismatch", "expected": expected_baseline_answers,
                       "actual": actual_baseline_answers})
    return {"status": "completed", "heldout_answer_count": len([row for row in rows if row["method"] == "simple_aggregation"]),
            "image_mean_log_density": by_image, "overall_mean_log_density": overall,
            "worker_mean_file": "qc_evaluation/heldout_worker_means.csv",
            "answer_score_file": "qc_evaluation/heldout_answer_scores.csv"}, tuple(all_held_views)


def _save_stage_fits(output: Path, stage: str, prepared, inputs: Path,
                     attempts: list[Fit], summaries: list[dict]) -> None:
    for fitted in attempts:
        label = f"G{fitted.G}_{fitted.initialization}"
        path = output / stage / f"{label}_fit.npz"
        metadata = save_fit_npz(path, fitted, prepared, inputs)
        summary = next(item for item in summaries if item["G"] == fitted.G
                       and item["initialization"] == fitted.initialization)
        summary["fit_file"] = str(path.relative_to(output))
        summary["qc_inputs_sha256"] = metadata["inputs_sha256"]


def _verify_saved_results(output: Path, stage: str, inputs: Path,
                          attempts: list[Fit], selected: dict[str, Fit | None] | None = None) -> dict:
    """Recompute objective and, for evaluation, every held-out CSV score without EM."""
    if not attempts:
        return {"status": "no_saved_fit", "checked_initializations": []}
    reloaded: dict[tuple[int, str], Fit] = {}
    checks = []
    reconstructed = None
    for fitted in attempts:
        label = f"G{fitted.G}_{fitted.initialization}"
        path = output / stage / f"{label}_fit.npz"
        loaded, prepared, heldout, metadata = load_fit_npz(path, inputs)
        recalculated = full_objective(prepared, loaded.parameters)
        if not np.isclose(recalculated, metadata["objective"], rtol=1e-10, atol=1e-8):
            raise ValueError(f"Saved objective differs for {stage}/{label}: {recalculated} vs {metadata['objective']}")
        checks.append({"initialization": label, "stored_objective": metadata["objective"],
                       "recalculated_objective": recalculated,
                       "absolute_difference": abs(recalculated - metadata["objective"]),
                       "inputs_sha256": metadata["inputs_sha256"]})
        reloaded[(fitted.G, fitted.initialization)] = loaded
        reconstructed = (prepared, heldout)
    answer_score_check = None
    if selected is not None:
        assert reconstructed is not None
        prepared, heldout = reconstructed
        views = {(view.image_id, view.view_id): (r, view) for r, view in enumerate(prepared.views)}
        workers = {worker: j for j, worker in enumerate(prepared.worker_ids)}
        selected_loaded = {label: reloaded[(fitted.G, fitted.initialization)]
                           for label, fitted in selected.items() if fitted is not None}
        expected = {}
        for image_id, view_id, answer in heldout:
            r, view = views[(image_id, view_id)]
            log_f = log_f_for_answer(view, answer)
            key = (image_id, view_id, answer.worker_id)
            expected[(*key, "simple_aggregation")] = float(logsumexp(np.log(baseline_weights(view)) + log_f))
            for label, fitted in selected_loaded.items():
                expected[(*key, label)] = predictive_from_log_f(log_f, fitted, workers[answer.worker_id], r)
        with (output / stage / "heldout_answer_scores.csv").open(newline="", encoding="utf-8") as stream:
            saved_rows = list(csv.DictReader(stream))
        if len(saved_rows) != len(expected):
            raise ValueError(f"Held-out score row count differs: {len(saved_rows)} vs {len(expected)}")
        max_difference = 0.0
        for row in saved_rows:
            key = (row["image_id"], row["view_id"], row["worker_id"], row["method"])
            if key not in expected:
                raise ValueError(f"Unexpected held-out score row: {key}")
            value = float(row["log_predictive_density"])
            recomputed = expected.pop(key)
            max_difference = max(max_difference, abs(value - recomputed))
            if not np.isclose(value, recomputed, rtol=1e-10, atol=1e-8):
                raise ValueError(f"Held-out score mismatch: {key}")
        if expected:
            raise ValueError(f"Missing held-out score rows: {list(expected)[:3]}")
        answer_score_check = {"rows": len(saved_rows), "maximum_absolute_difference": max_difference,
                              "tolerance": {"rtol": 1e-10, "atol": 1e-8}, "passed": True}
    return {"status": "passed", "checked_initializations": checks,
            "heldout_answer_scores": answer_score_check,
            "recalculation_did_not_run_EM": True}


def _improvements(prediction: dict, candidate: str, reference: str) -> dict | None:
    by_image = prediction["image_mean_log_density"]
    overall = prediction["overall_mean_log_density"]
    if candidate not in overall or reference not in overall:
        return None
    deltas = {name: by_image[candidate][name] - by_image[reference][name] for name in QC_IDS}
    return {"overall": overall[candidate] - overall[reference], "by_image": deltas,
            "positive_image_count": sum(value > 0 for value in deltas.values())}


def _compare_previous_scores(output: Path, prediction: dict) -> dict:
    if not PREVIOUS_REPORT.is_file():
        return {"status": "previous_report_missing", "path": str(PREVIOUS_REPORT)}
    previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8")).get("qc_prediction", {})
    rows = []
    for method in ("simple_aggregation", "G1", "G2"):
        for name in (*QC_IDS, "overall_five_image_mean"):
            if name == "overall_five_image_mean":
                old = previous.get("overall_mean_log_density", {}).get(method)
                new = prediction.get("overall_mean_log_density", {}).get(method)
            else:
                old = previous.get("image_mean_log_density", {}).get(method, {}).get(name)
                new = prediction.get("image_mean_log_density", {}).get(method, {}).get(name)
            rows.append({"method": method, "image_id": name, "previous_mean_log_density": old,
                         "current_mean_log_density": new,
                         "current_minus_previous": new - old if new is not None and old is not None else None,
                         "previous_fit_status": "unconverged_diagnostic" if method != "simple_aggregation" else "baseline"})
    path = output / "qc_evaluation" / "previous_vs_current_scores.csv"
    write_csv(path, rows)
    return {"status": "completed", "file": str(path.relative_to(output)),
            "previous_report": str(PREVIOUS_REPORT),
            "previous_model_scores_are_diagnostic_only": True,
            "overall": [row for row in rows if row["image_id"] == "overall_five_image_mean"]}


def _choose_model(prediction: dict, g1: Fit | None, g2: Fit | None, checks_passed: bool) -> tuple[Fit | None, bool, str]:
    vs_g1 = _improvements(prediction, "G2", "G1")
    vs_baseline_g2 = _improvements(prediction, "G2", "simple_aggregation")
    if g1 is not None and g1.converged and g2 is not None and g2.converged:
        if vs_g1 is not None and vs_g1["overall"] >= 0.001 and vs_g1["positive_image_count"] >= 3:
            selected, reason = g2, "両モデルが収束し、G=2がG=1より全体平均0.001以上・QC5枚中3枚以上で改善"
        else:
            selected, reason = g1, "両モデルが収束したが、G=2はG=1に対する固定改善条件を満たさなかった"
    elif g1 is not None and g1.converged:
        selected, reason = g1, "G=1のみ収束して比較可能"
    elif (g1 is None or g1.status == "numerical_failure") and g2 is not None and g2.converged:
        selected, reason = g2, "G=1が数値的に失敗したため、収束したG=2を単純集計と比較"
    else:
        return None, False, "収束と比較可能性の条件を満たすモデルがない"
    vs_baseline = _improvements(prediction, f"G{selected.G}", "simple_aggregation")
    adopted = bool(vs_baseline is not None and vs_baseline["overall"] >= 0.001
                   and vs_baseline["positive_image_count"] >= 3 and checks_passed)
    if not adopted:
        reason += "。採用には単純集計比で全体平均0.001以上・QC5枚中3枚以上の改善と全人工検査の合格が必要"
    return selected, adopted, reason


def _parameter_file(output: Path, fitted: Fit, prepared, attempts: list[dict]) -> Path:
    p = fitted.parameters
    views = prepared.views
    file = output / "qc_fixed_parameters.json"
    write_json(file, {"status": "converged", "estimate_method": "conditional_MAP",
                      "G": fitted.G, "omega": p.omega.tolist(),
                      "workers": {worker: {"q": p.q[j].tolist(), "rho": float(p.rho[j])}
                                  for j, worker in enumerate(prepared.worker_ids)},
                      "view_pi": [{"image_id": view.image_id, "view_id": view.view_id,
                                   "pi": p.pi[r, :, :len(view.candidates)].tolist()}
                                  for r, view in enumerate(views)],
                      "kernel": KERNEL, "priors": PRIORS,
                      "qc_image_ids": list(QC_IDS), "worker_ids": list(prepared.worker_ids),
                      "number_of_qc_views": len(views),
                      "number_of_qc_answers": sum(len(view.answers) for view in views),
                      "number_of_non_qc_answers_used": 0,
                      "converged": True, "selected_initialization": fitted.initialization,
                      "objective": fitted.objective, "attempts": attempts,
                      "selection": "maximum full MAP objective among converged starts of selected G"})
    return file


def _worker_parameters_from_saved(file: Path) -> tuple[dict[str, tuple[np.ndarray, float]], int, str]:
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    saved = json.loads(file.read_text(encoding="utf-8"))
    if not saved["converged"] or saved["kernel"] != KERNEL or saved["priors"] != PRIORS:
        raise ValueError("Stage B requires converged QC parameters for this fixed model")
    values = {worker: (np.array(entry["q"], dtype=np.float64), float(entry["rho"]))
              for worker, entry in saved["workers"].items()}
    return values, int(saved["G"]), digest


def _non_qc_ids(root: Path) -> tuple[str, ...]:
    return tuple(sorted(child.name for child in root.iterdir()
                        if child.is_dir() and child.name not in QC_IDS and (child / child.name).is_file()))


def _first_non_qc(ids: tuple[str, ...]) -> str:
    preferred = "Cortical_subplate_0000_240_8.png"
    return preferred if preferred in ids else ids[0]


def _process_image(root: Path, name: str, directory: Path,
                   fixed: dict[str, tuple[np.ndarray, float]] | None = None,
                   G: int | None = None, fixed_digest: str | None = None) -> dict:
    problems: list[dict] = []
    mismatches: list[dict] = []
    try:
        views = load_image_views(root, name, issues=problems, mismatches=mismatches)
    except Exception as exc:
        return {"image_id": name, "status": "input_invalid", "issues": [{"image_id": name, "reason": str(exc)}],
                "cell_count_mismatches": [], "counts": {"successful_views": 0, "input_invalid_views": 1,
                                                  "unconverged_views": 0, "baseline_views": 0}}
    if not views:
        return {"image_id": name, "status": "input_invalid", "issues": problems,
                "cell_count_mismatches": mismatches,
                "counts": {"successful_views": 0, "input_invalid_views": 1,
                           "unconverged_views": 0, "baseline_views": 0}}
    unknown = set()
    if fixed is not None:
        assert G in (1, 2)
        mean_q = np.mean([value[0] for value in fixed.values()], axis=0)
        mean_rho = float(np.mean([value[1] for value in fixed.values()]))
        for view in views:
            for answer in view.answers:
                if answer.worker_id not in fixed:
                    unknown.add(answer.worker_id)
        applied = {**fixed, **{worker: (mean_q.copy(), mean_rho) for worker in unknown}}
    else:
        applied = {}
    selected_weights, methods, statuses, details = [], [], [], []
    counters = {"successful_views": 0, "input_invalid_views": 0,
                "unconverged_views": 0, "baseline_views": 0}
    for view in views:
        affected = [item for item in problems if item.get("view_id") in (None, view.view_id)]
        method, status = "simple_aggregation", "baseline_only"
        detail: dict = {"used_answers": len(view.answers), "fixed_parameter_sha256": fixed_digest}
        weights = baseline_weights(view)
        if fixed is not None and not affected:
            try:
                inference = infer_view(view, applied, G)
                if inference["converged"]:
                    weights = inference["weights"]
                    method, status = "conditional_MAP", inference["status"]
                    counters["successful_views"] += 1
                else:
                    status = "baseline_after_nonconvergence"
                    counters["unconverged_views"] += 1
                detail.update({"iterations": inference["iterations"], "objective": inference["objective"],
                               "v": inference["v"].tolist(), "class_pi": inference["pi"].tolist(),
                               "fixed_worker_rho": {a.worker_id: applied[a.worker_id][1] for a in view.answers}})
            except (ValueError, FloatingPointError, ArithmeticError) as exc:
                status = "baseline_after_inference_error"
                counters["unconverged_views"] += 1
                detail["reason"] = str(exc)
        elif fixed is not None:
            status = "baseline_after_input_error"
            counters["input_invalid_views"] += 1
            detail["reason"] = affected
        if method == "simple_aggregation":
            counters["baseline_views"] += 1
        selected_weights.append(weights)
        methods.append(method)
        statuses.append(status)
        details.append(detail)
    saved = save_image_outputs(views, tuple(selected_weights), directory,
                               tuple(methods), tuple(statuses), tuple(details))
    return {**saved, "status": "completed" if not problems else "completed_with_input_issues",
            "counts": counters, "issues": problems, "cell_count_mismatches": mismatches,
            "unknown_worker_ids_using_QC_means": sorted(unknown),
            "fixed_parameter_sha256": fixed_digest}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _compare_one_image(root: Path, name: str, output: Path,
                       fixed: dict[str, tuple[np.ndarray, float]], G: int,
                       digest: str) -> dict:
    parent = output / "non_qc_one_image"
    baseline = _process_image(root, name, parent / "baseline")
    calibrated = _process_image(root, name, parent / "calibrated", fixed, G, digest)
    baseline_dir = parent / "baseline" / name
    calibrated_dir = parent / "calibrated" / name
    if not (baseline_dir / "states.csv").is_file() or not (calibrated_dir / "states.csv").is_file():
        return {"status": "input_invalid", "image_id": name, "baseline": baseline,
                "calibrated": calibrated, "comparison_file": None,
                "counts": calibrated["counts"]}
    b_states, c_states = _read_csv(baseline_dir / "states.csv"), _read_csv(calibrated_dir / "states.csv")
    b_views, c_views = _read_csv(baseline_dir / "views.csv"), _read_csv(calibrated_dir / "views.csv")
    if len(b_states) != len(c_states) or len(b_views) != len(c_views):
        raise ValueError("Baseline and calibrated candidate/view counts differ on the same image")
    groups_b: dict[str, list[dict]] = {}
    groups_c: dict[str, list[dict]] = {}
    for row in b_states:
        groups_b.setdefault(row["view_id"], []).append(row)
    for row in c_states:
        groups_c.setdefault(row["view_id"], []).append(row)
    comparison = []
    for b_view, c_view in zip(b_views, c_views):
        view_id = b_view["view_id"]
        if view_id != c_view["view_id"]:
            raise ValueError(f"View order differs in one-image comparison: {view_id} vs {c_view['view_id']}")
        bs, cs = groups_b[view_id], groups_c[view_id]
        if len(bs) != len(cs):
            raise ValueError(f"Candidate count differs at {name}/{view_id}")
        for b, c in zip(bs, cs):
            if b["state_id"] != c["state_id"] or b["points_local"] != c["points_local"]:
                raise ValueError(f"Candidate order/content differs at {name}/{view_id}")
            for field in ("points_local", "points_normalized", "points_global"):
                if not np.isfinite(np.asarray(json.loads(c[field]), dtype=np.float64)).all():
                    raise ValueError(f"Non-finite candidate coordinates at {name}/{view_id}")
        bw = np.asarray([float(row["state_weight"]) for row in bs])
        cw = np.asarray([float(row["state_weight"]) for row in cs])
        if not all(np.isfinite(a).all() and abs(a.sum() - 1) < 1e-10 for a in (bw, cw)):
            raise ValueError(f"Invalid candidate weights at {name}/{view_id}")
        b_count = float(b_view["expected_point_count"])
        c_count = float(c_view["expected_point_count"])
        if not np.isfinite([b_count, c_count]).all():
            raise ValueError(f"Non-finite expected point count at {name}/{view_id}")
        point_counts = np.asarray([int(row["point_count"]) for row in bs])
        if not np.isclose(b_count, bw @ point_counts, atol=1e-10):
            raise ValueError(f"Baseline expected count mismatch at {name}/{view_id}")
        if not np.isclose(c_count, cw @ point_counts, atol=1e-10):
            raise ValueError(f"Calibrated expected count mismatch at {name}/{view_id}")
        is_calibrated = c_view["estimate_method"] == "conditional_MAP"
        comparison.append({"image_id": name, "view_id": view_id,
                           "candidate_count": len(bs), "inference_status": c_view["status"],
                           "baseline_expected_count": b_count,
                           "calibrated_expected_count": c_count if is_calibrated else None,
                           "baseline_max_state_weight": float(bw.max()),
                           "calibrated_max_state_weight": float(cw.max()) if is_calibrated else None,
                           "weight_total_variation": float(0.5 * np.abs(cw - bw).sum()) if is_calibrated else None,
                           "calibrated_output_method": c_view["estimate_method"],
                           "fallback_reason": json.loads(c_view["detail"]).get("reason") if not is_calibrated else None})
    for directory, rows in ((baseline_dir, b_views), (calibrated_dir, c_views)):
        with np.load(directory / "mass_maps.npz", allow_pickle=False) as maps:
            for i, row in enumerate(rows):
                if not np.isclose(float(maps[f"view_{i}"].sum()), float(row["expected_point_count"]), atol=1e-10):
                    raise ValueError(f"Mass conservation failed at {name}/{row['view_id']}")
    comparison_file = parent / "comparison.csv"
    write_csv(comparison_file, comparison)
    counts = {"calibrated_views": sum(row["calibrated_output_method"] == "conditional_MAP" for row in comparison),
              "fallback_views": sum(row["calibrated_output_method"] != "conditional_MAP" for row in comparison),
              "unconverged_views": calibrated["counts"]["unconverged_views"],
              "input_invalid_views": calibrated["counts"]["input_invalid_views"]}
    valid = [row for row in comparison if row["calibrated_output_method"] == "conditional_MAP"]
    summary = {"comparable_views": len(valid)}
    if valid:
        for field in ("baseline_expected_count", "calibrated_expected_count",
                      "baseline_max_state_weight", "calibrated_max_state_weight",
                      "weight_total_variation"):
            summary[f"mean_{field}"] = float(np.mean([row[field] for row in valid]))
    return {"status": "completed", "image_id": name, "baseline": baseline,
            "calibrated": calibrated, "counts": counts,
            "comparison_summary": summary,
            "comparison_file": str(comparison_file.relative_to(output)),
            "validation": {"candidate_order_same": True, "weights_sum_to_one": True,
                           "finite_coordinates_and_expected_counts": True,
                           "mass_sum_matches_expected_count": True}}


def _baseline_one(root: Path, output: Path, ids: tuple[str, ...], report: dict, reason: str) -> None:
    if not ids:
        report["non_qc"] = {"status": "no_non_qc_images", "reason": reason,
                            "counts": {"successful_views": 0, "input_invalid_views": 0,
                                       "unconverged_views": 0, "baseline_views": 0}}
        return
    name = _first_non_qc(ids)
    result = _process_image(root, name, output / "non_qc_baseline_one")
    report["non_qc"] = {"status": "baseline_diagnostic_only", "reason": reason,
                        "one_image": result, "all_non_qc": "not_executed",
                        "counts": result["counts"]}
    _announce(stage="baseline_one_image", image_id=name, counts=result["counts"])


def _run(root: Path | None, output: Path, report: dict, non_qc_limit: int) -> None:
    def after_check(name, result):
        report["checks"][name] = result
        _announce(stage="artificial_checks", name=name, passed=result["passed"])
    report["checks"] = run_eight_checks(after_check)
    report["artificial_checks_passed"] = all(item["passed"] for item in report["checks"].values())
    report["status"] = "artificial_checks_completed"
    _save_report(output, report)
    if root is None:
        report["status"] = "missing_input_root"
        report["input_audit"] = {"requested": report["requested_data_root"],
                                  "reason": "QC1–5 were not found at the requested root or one immediate child"}
        return
    problems: list[dict] = []
    mismatches: list[dict] = []
    try:
        workers, excluded = select_qc_workers(root)
    except Exception as exc:
        report["status"] = "invalid_qc_worker_records"
        report["input_audit"] = {"issues": [{"reason": str(exc)}]}
        return
    train_ids, held_ids = _heldout_split(workers)
    train = _read_qc_views(root, train_ids, problems, mismatches)
    report["qc_scope"] = {"qc_image_ids": list(QC_IDS), "number_of_workers": len(workers),
                          "worker_ids": list(workers), "outside_worker_ids": list(excluded),
                          "number_of_views": len(train),
                          "number_of_training_answers": sum(len(v.answers) for v in train),
                          "number_of_heldout_answers_expected": len(workers) * 49,
                          "number_of_answers": len(workers) * len(train),
                          "non_qc_answers_in_fit": 0,
                          "training_workers_by_image": {name: list(values) for name, values in train_ids.items()},
                          "heldout_workers_by_image": {name: list(values) for name, values in held_ids.items()}}
    report["input_audit"] = {"issues": problems, "cell_count_mismatches": mismatches}
    if problems:
        report["status"] = "invalid_qc_input"
        _announce(stage="input_audit", issues=len(problems), mismatches=len(mismatches))
        return
    if len(train) != 245 or any(sum(v.image_id == name for v in train) != 49 for name in QC_IDS):
        report["status"] = "unexpected_qc_view_count"
        return
    prepared = prepare(train, workers)
    report["qc_scope"]["number_of_learning_views"] = prepared.number_of_learning_views
    report["qc_scope"]["number_of_training_candidates"] = sum(len(v.candidates) for v in train)
    if prepared.number_of_learning_views == 0:
        report["status"] = "all_qc_single_candidate"
        report["qc_prediction"] = {"status": "unidentifiable", "selection_reason": "Every QC view has K=1"}
        _baseline_one(root, output, _non_qc_ids(root), report, "all_QC_K1")
        return
    _announce(stage="qc_training_prepared", views=len(train), answers=sum(len(v.answers) for v in train),
              candidate_count=report["qc_scope"]["number_of_training_candidates"])
    g1, g1_attempts, g1_fits = _fit_attempts(prepared, 1, "qc_evaluation", output)
    g2, g2_attempts, g2_fits = _fit_attempts(prepared, 2, "qc_evaluation", output)
    report["qc_prediction"] = {"status": "fitting_completed", "attempts": {"G1": g1_attempts, "G2": g2_attempts}}
    _save_report(output, report)
    held_problems: list[dict] = []
    prediction, held_views = _score_heldout(root, train, held_ids, workers, {"G1": g1, "G2": g2},
                                            output, held_problems, mismatches)
    if held_problems:
        report["status"] = "invalid_heldout_qc_input"
        report["input_audit"]["issues"].extend(held_problems)
        report["input_audit"]["cell_count_mismatches"] = mismatches
        return
    eval_inputs = output / "qc_evaluation" / "qc_inputs.json.gz"
    save_qc_inputs(eval_inputs, train, workers, held_views)
    _save_stage_fits(output, "qc_evaluation", prepared, eval_inputs, g1_fits, g1_attempts)
    _save_stage_fits(output, "qc_evaluation", prepared, eval_inputs, g2_fits, g2_attempts)
    report["saved_result_recalculation"] = {"qc_evaluation": _verify_saved_results(
        output, "qc_evaluation", eval_inputs, g1_fits + g2_fits, {"G1": g1, "G2": g2})}
    report["qc_inputs"] = {"evaluation_file": str(eval_inputs.relative_to(output)),
                            "evaluation_sha256": hashlib.sha256(eval_inputs.read_bytes()).hexdigest()}
    report["qc_prediction"].update(prediction)
    report["qc_prediction"]["previous_run_comparison"] = _compare_previous_scores(output, prediction)
    report["qc_prediction"]["comparison_G2_minus_G1"] = _improvements(prediction, "G2", "G1")
    report["qc_prediction"]["comparison_G1_minus_baseline"] = _improvements(prediction, "G1", "simple_aggregation")
    report["qc_prediction"]["comparison_G2_minus_baseline"] = _improvements(prediction, "G2", "simple_aggregation")
    selected, adopt, reason = _choose_model(prediction, g1, g2, report["artificial_checks_passed"])
    report["qc_prediction"]["selected_model"] = f"G{selected.G}" if selected else None
    report["qc_prediction"]["selection_reason"] = reason
    report["qc_prediction"]["adoption_condition_met"] = adopt
    _announce(stage="qc_heldout_prediction", overall=prediction["overall_mean_log_density"],
              selected=report["qc_prediction"]["selected_model"], adoption=adopt)
    report["status"] = "qc_prediction_completed"
    _save_report(output, report)
    ids = _non_qc_ids(root)
    if not adopt or selected is None:
        report["status"] = "calibration_not_adopted"
        report["final_qc"] = {"status": "not_run_after_qc_prediction", "converged": False}
        _baseline_one(root, output, ids, report, "artificial_checks_or_qc_prediction_failed_adoption")
        return
    full_problems: list[dict] = []
    full_views = _read_qc_views(root, {name: workers for name in QC_IDS}, full_problems, mismatches)
    if full_problems:
        report["status"] = "invalid_full_qc_input"
        report["input_audit"]["issues"].extend(full_problems)
        report["input_audit"]["cell_count_mismatches"] = mismatches
        return
    full_prepared = prepare(full_views, workers)
    report["qc_scope"].update(number_of_views=len(full_views),
                               number_of_answers=sum(len(v.answers) for v in full_views),
                               number_of_full_candidates=sum(len(v.candidates) for v in full_views))
    final_inputs = output / "qc_final" / "qc_inputs.json.gz"
    save_qc_inputs(final_inputs, full_views, workers)
    report["qc_inputs"]["final_file"] = str(final_inputs.relative_to(output))
    report["qc_inputs"]["final_sha256"] = hashlib.sha256(final_inputs.read_bytes()).hexdigest()
    final, attempts, final_fits = _fit_attempts(full_prepared, selected.G, "qc_final", output)
    _save_stage_fits(output, "qc_final", full_prepared, final_inputs, final_fits, attempts)
    report["saved_result_recalculation"]["qc_final"] = _verify_saved_results(
        output, "qc_final", final_inputs, final_fits)
    report["final_qc"] = {"status": final.status if final else "numerical_failure", "G": selected.G,
                           "converged": bool(final and final.converged), "attempts": attempts,
                           "number_of_views": len(full_views),
                           "number_of_answers": sum(len(v.answers) for v in full_views),
                           "number_of_non_qc_answers_used": 0,
                           "kernel": KERNEL, "priors": PRIORS}
    _save_report(output, report)
    if final is None or not final.converged:
        report["status"] = "final_qc_not_converged"
        _baseline_one(root, output, ids, report, "final_QC_not_converged")
        return
    parameter_path = _parameter_file(output, final, full_prepared, attempts)
    fixed, G, digest = _worker_parameters_from_saved(parameter_path)
    report["final_qc"].update(selected_initialization=final.initialization,
                               fixed_parameter_file=str(parameter_path.relative_to(output)),
                               fixed_parameter_sha256=digest,
                               omega=final.parameters.omega.tolist(),
                               rho_by_worker={w: float(final.parameters.rho[j]) for j, w in enumerate(workers)},
                               q_by_worker={w: final.parameters.q[j].tolist() for j, w in enumerate(workers)})
    if not ids:
        report["status"] = "final_qc_converged_no_non_qc_images"
        report["adopted"] = True
        report["non_qc"] = {"status": "no_non_qc_images", "counts": {}}
        return
    first = _first_non_qc(ids)
    comparison = _compare_one_image(root, first, output, fixed, G, digest)
    report["non_qc"] = {"status": "one_image_compared", "one_image": comparison,
                        "selected_image_reason": "preferred_fixed_id" if first == "Cortical_subplate_0000_240_8.png"
                                                 else "preferred_id_missing; first_sorted_non_qc_id",
                        "fixed_parameter_sha256_before": digest,
                        "non_qc_limit": non_qc_limit}
    _announce(stage="non_qc_first_image", image_id=first, counts=comparison["counts"])
    _save_report(output, report)
    other_ids = [name for name in ids if name != first][:max(0, non_qc_limit - 1)]
    other_results = [_process_image(root, name, output / "non_qc_limited", fixed, G, digest)
                     for name in other_ids]
    after = hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    if after != digest:
        raise RuntimeError("QC fixed parameter file changed during Stage B")
    report["non_qc"].update(status="completed_one_image" if non_qc_limit == 1 else "completed_limited_images",
                            other_image_results=other_results,
                            number_of_non_qc_images_available=len(ids),
                            number_of_non_qc_images_processed=1 + len(other_results),
                            counts=comparison["counts"],
                            fixed_parameter_sha256_after=after,
                            fixed_parameter_file_unchanged=True,
                            unknown_workers=sorted(set(worker for result in [comparison["calibrated"], *other_results]
                                                       for worker in result.get("unknown_worker_ids_using_QC_means", []))))
    report["status"] = "completed"
    report["adopted"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "convergence_extension")
    parser.add_argument("--non-qc-limit", type=int, default=1)
    args = parser.parse_args()
    if args.non_qc_limit < 1:
        parser.error("--non-qc-limit must be at least 1")
    output = next_output_dir(args.output_dir)
    root = resolve_data_root(args.data_root)
    command = f"python {Path(__file__).resolve()} --data-root {args.data_root} --output-dir {args.output_dir} --non-qc-limit {args.non_qc_limit}"
    report = {"status": "starting", "command": command,
              "source_files": SOURCE_FILES, "output_dir": str(output),
              "requested_data_root": str(args.data_root), "resolved_data_root": str(root) if root else None,
              "kernel": KERNEL, "priors": PRIORS, "checks": {}, "adopted": False,
              "limitations": LIMITATIONS,
              "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    started = time.perf_counter()
    try:
        _run(root, output, report, args.non_qc_limit)
    except Exception as exc:
        report["status"] = "implementation_error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _save_report(output, report)
        _announce(stage="finished", status=report["status"], output_dir=str(output),
                  elapsed_seconds=report["elapsed_seconds"])


if __name__ == "__main__":
    main()
