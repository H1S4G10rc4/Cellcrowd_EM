"""Re-aggregate saved results and build verified report assets."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "point_annotation_trial" / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = REPORT_ROOT
WORKSPACE = HERE.parent
TRIAL = WORKSPACE / "point_annotation_trial"
SAVED = TRIAL / "outputs/convergence_extension"
SLIDES = TRIAL / "outputs/slide_figures"

from data import Answer, View, baseline_weights, expected_mass  # noqa: E402
from model import load_fit_npz, log_f_for_answer, predictive_from_log_f  # noqa: E402

FIGURES = {
    "A_location": ("stable_70_30", "stable_minority_diagnostic"),
    "B_count": ("one_vs_two", "synthetic_one_vs_two"),
    "C_switching": ("unstable_worker", "unstable_worker_diagnostic"),
    "D_complementary": ("complementary_miss", "synthetic_complementary_miss"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qc_reaggregate() -> dict:
    report = json.loads((SAVED / "report.json").read_text())
    rows = []
    for image in report["qc_scope"]["qc_image_ids"]:
        rows.append({"image_id": image,
                     **{method: report["qc_prediction"]["image_mean_log_density"][method][image]
                        for method in ("simple_aggregation", "G1", "G2")}})
    rows.append({"image_id": "five_image_mean",
                 **report["qc_prediction"]["overall_mean_log_density"]})
    save_csv(HERE / "results/real_qc_heldout.csv", rows,
             ["image_id", "simple_aggregation", "G1", "G2"])

    folder = SAVED / "qc_evaluation"
    fit, prepared, heldout, _ = load_fit_npz(folder / "G2_ospa_pair_fit.npz",
                                             folder / "qc_inputs.json.gz")
    view_index = {(v.image_id, v.view_id): i for i, v in enumerate(prepared.views)}
    worker_index = {w: j for j, w in enumerate(prepared.worker_ids)}
    source = pd.read_csv(folder / "heldout_answer_scores.csv")
    saved_scores = {(r.image_id, r.view_id, r.worker_id, r.method): r.log_predictive_density
                    for r in source.itertuples(index=False)}
    prediction_rows = []
    for image_id, view_id, answer in heldout:
        i = view_index[image_id, view_id]
        view = prepared.views[i]
        log_f = log_f_for_answer(view, answer)
        k = len(view.candidates)
        observed = [worker_index[a.worker_id] for a in view.answers]
        v = fit.parameters.q[observed].mean(axis=0)
        weights = v @ fit.parameters.pi[i, :, :k]
        integrated = float(logsumexp(np.log(weights) + log_f))
        baseline = float(logsumexp(np.log(baseline_weights(view)) + log_f))
        personalized = predictive_from_log_f(log_f, fit, worker_index[answer.worker_id], i)
        key = (image_id, view_id, answer.worker_id)
        if not np.isclose(baseline, saved_scores[*key, "simple_aggregation"], atol=1e-10):
            raise RuntimeError("Saved baseline score did not recompute")
        if not np.isclose(personalized, saved_scores[*key, "G2"], atol=1e-10):
            raise RuntimeError("Saved G2 score did not recompute")
        prediction_rows.append({"image_id": image_id, "view_id": view_id,
                                "worker_id": answer.worker_id, "simple_aggregation": baseline,
                                "G2_integrated_weights": integrated,
                                "G2_worker_specific": personalized})
    if len(prediction_rows) != 1715:
        raise RuntimeError("Unexpected held-out answer count")
    save_csv(HERE / "results/real_qc_predictor_by_answer.csv", prediction_rows,
             ["image_id", "view_id", "worker_id", "simple_aggregation",
              "G2_integrated_weights", "G2_worker_specific"])
    frame = pd.DataFrame(prediction_rows)
    means = []
    for image in (*report["qc_scope"]["qc_image_ids"], "five_image_mean"):
        subset = frame[frame.image_id == image] if image != "five_image_mean" else None
        means.append({"image_id": image,
                      **{key: float(subset[key].mean()) if subset is not None
                         else float(frame.groupby("image_id")[key].mean().mean())
                         for key in ("simple_aggregation", "G2_integrated_weights",
                                     "G2_worker_specific")}})
    save_csv(HERE / "results/real_qc_predictor_means.csv", means,
             ["image_id", "simple_aggregation", "G2_integrated_weights", "G2_worker_specific"])
    return report


def non_qc_reaggregate(report: dict) -> None:
    frame = pd.read_csv(SAVED / "non_qc_one_image/comparison.csv")
    if len(frame) != 49 or not (frame.inference_status == "converged").all():
        raise RuntimeError("Saved non-QC result is incomplete")
    one = report["non_qc"]["one_image"]
    if frame.image_id.nunique() != 1 or frame.image_id.iloc[0] != one["image_id"]:
        raise RuntimeError("Non-QC image mismatch")
    row = {"image_id": frame.image_id.iloc[0], "view_count": len(frame),
           "mean_baseline_expected_count": float(frame.baseline_expected_count.mean()),
           "mean_calibrated_expected_count": float(frame.calibrated_expected_count.mean()),
           "mean_weight_total_variation": float(frame.weight_total_variation.mean()),
           "mean_baseline_max_state_weight": float(frame.baseline_max_state_weight.mean()),
           "mean_calibrated_max_state_weight": float(frame.calibrated_max_state_weight.mean())}
    for key in ("mean_baseline_expected_count", "mean_calibrated_expected_count",
                "mean_weight_total_variation"):
        if not np.isclose(row[key], one["comparison_summary"][key], atol=1e-12):
            raise RuntimeError(f"Non-QC saved mean mismatch: {key}")
    save_csv(HERE / "results/real_non_qc_summary.csv", [row], list(row))


def verify_and_copy_figures() -> dict:
    target_dir = HERE / "figures"
    target_dir.mkdir(exist_ok=True)
    provenance = {}
    for condition, (old_folder, stem) in FIGURES.items():
        previous = np.load(SLIDES / "synthetic_qc" / old_folder / "maps.npz")
        legacy = np.load(HERE / "results" / condition / "legacy/native_heatmaps.npz")["qc"][0]
        if not np.array_equal(previous["cellcrowd_score"], legacy):
            raise RuntimeError(f"Legacy map differs from figure source: {condition}")
        parameters = np.load(HERE / "results" / condition / "current/fitted_parameters.npz")
        payload = json.loads((HERE / "synthetic_inputs" / condition / "qc_views.json").read_text())
        row = payload["views"][0]
        view = View(row["image_id"], row["view_id"], 0, 0, 100, 100, 100, 100,
                    tuple(Answer(a["worker_id"], np.asarray(a["points"], dtype=float).reshape(-1, 2))
                          for a in row["answers"]))
        weights = parameters["q"].mean(axis=0) @ parameters["pi_qc"][0, :, :len(view.candidates)]
        if not np.array_equal(expected_mass(view, weights), previous["current_expected_mass_raw"]):
            raise RuntimeError(f"Current mass differs from figure source: {condition}")
        for suffix in ("png", "svg"):
            source = SLIDES / f"{stem}.{suffix}"
            destination = target_dir / source.name
            shutil.copy2(source, destination)
            provenance[destination.name] = {"sha256": sha(destination), "source": str(source),
                                            "depicts": "synthetic QC view 0"}
    for stem in ("qc_holdout_comparison", "non_qc_one_image_summary"):
        for suffix in ("png", "svg"):
            source = SLIDES / f"{stem}.{suffix}"
            destination = target_dir / source.name
            shutil.copy2(source, destination)
            provenance[destination.name] = {"sha256": sha(destination), "source": str(source),
                                            "depicts": "saved real-data result"}
    return provenance


def fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value) or value == "":
        return r"\textemdash"
    x = float(value)
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        mantissa, exponent = f"{x:.2e}".split("e")
        return "$" + f"{mantissa}\\times10^{{{int(exponent)}}}" + "$"
    return f"{x:.{digits}f}"


def table(name: str, alignment: str, header: list[str], rows: list[list[str]]) -> None:
    target = HERE / "tables" / f"{name}.tex"
    target.parent.mkdir(exist_ok=True)
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables() -> None:
    summary = pd.read_csv(HERE / "results/synthetic_summary.csv")
    labels = {"A_location": "A：位置", "B_count": "B：点数",
              "C_switching": "C：切替", "D_complementary": "D：補完"}
    method = {"simple_aggregation": "単純集計", "current": "現行G=2",
              "legacy": "旧版の投影"}
    main_rows = []
    for row in summary.itertuples(index=False):
        if row.reference == "generated_two_point_setup":
            continue
        ref = {"known_group_frequency": "7:3", "all_11_workers": "11人",
               "stable_10_workers": "安定10人", "observed_judgment": "5:5"}[row.reference]
        main_rows.append([labels[row.condition], method[row.method], ref,
                          fmt(row.weight_A, 3), fmt(row.tv_to_judgment_reference, 3),
                          fmt(row.expected_count, 3),
                          fmt(row.absolute_count_error, 3)])
    table("synthetic_common", "lllrrrr",
          ["条件", "手法", "参照", "$w_A$", "TV", "$E[N]$", "点数誤差"],
          main_rows)
    pair = summary[summary.reference == "generated_two_point_setup"]
    table("complementary_reference", "lrrr",
          ["表現", "$E[N]-2$", "平均OSPA", "2点候補重み"],
          [[method[row.method], fmt(row.signed_count_error, 3),
            fmt(row.expected_ospa_to_generated_pair, 3),
            fmt(row.two_point_candidate_probability, 3)]
           for row in pair.itertuples(index=False)])
    native = pd.concat(
        [pd.read_csv(HERE / "results" / condition / "legacy/native_summary.csv")
         for condition in labels], ignore_index=True)
    table("legacy_native", "lrrr",
          ["条件", "QCクラスタ", "評価クラスタ", "評価マップの範囲"],
          [[labels[row.condition], str(int(row.n_clusters_qc)),
            str(int(row.n_clusters_evaluation)),
            f"[{fmt(row.map_min_evaluation, 2)}, {fmt(row.map_max_evaluation, 2)}]"]
           for row in native.itertuples(index=False)])
    qc = pd.read_csv(HERE / "results/real_qc_heldout.csv")
    table("real_qc_holdout", "lrrr",
          ["画像", "単純集計", "$G=1$", "$G=2$"],
          [[("5画像平均" if row.image_id == "five_image_mean" else row.image_id.replace(".png", "")),
            fmt(row.simple_aggregation),
            fmt(row.G1), fmt(row.G2)] for row in qc.itertuples(index=False)])
    pred = pd.read_csv(HERE / "results/real_qc_predictor_means.csv")
    table("real_qc_predictors", "lrrr",
          ["画像", "単純集計", "統合重み", "回答者別"],
          [[("5画像平均" if row.image_id == "five_image_mean" else row.image_id.replace(".png", "")),
            fmt(row.simple_aggregation),
            fmt(row.G2_integrated_weights), fmt(row.G2_worker_specific)]
           for row in pred.itertuples(index=False)])
    non = pd.read_csv(HERE / "results/real_non_qc_summary.csv")
    table("real_non_qc", "rrrr",
          ["view数", "単純集計の期待点数", "校正後の期待点数", "平均TV"],
          [[str(row.view_count), fmt(row.mean_baseline_expected_count, 3),
            fmt(row.mean_calibrated_expected_count, 3),
            fmt(row.mean_weight_total_variation, 3)]
           for row in non.itertuples(index=False)])
    scores = pd.read_csv(HERE / "results/worker_scores.csv")
    subset = scores[scores.condition == "C_switching"].copy()
    subset["group"] = subset.worker_id.str.slice(0, 1).str.upper()
    score_rows = []
    for group, label in (("A", "安定A群"), ("B", "安定B群"), ("U", "切替回答者")):
        values = subset[subset.group == group]
        rho = values[(values.method == "current") & (values.measure == "rho")].value
        old = values[(values.method == "legacy") &
                     (values.measure == "weight_positive")].value
        if len(rho) == 0 or len(old) == 0:
            raise RuntimeError(f"Missing worker scores for {group}")
        score_rows.append({"group": label, "workers": len(rho),
                           "median_rho": float(rho.median()),
                           "median_legacy_click_weight": float(old.median())})
    save_csv(HERE / "results/worker_score_summary.csv", score_rows, list(score_rows[0]))
    table("worker_scores", "lrrr",
          ["回答群", "人数", r"$\rho$の中央値", "旧版クリック重みの中央値"],
          [[r["group"], str(r["workers"]), fmt(r["median_rho"], 3),
            fmt(r["median_legacy_click_weight"], 3)] for r in score_rows])
    times = []
    for condition in labels:
        for name in ("legacy", "current"):
            run = json.loads((HERE / "results" / condition / name / "run.json").read_text())
            process = json.loads((HERE / "results" / condition / name / "process.json").read_text())
            times.append({"condition": condition, "method": name,
                          "calibration_seconds": run["calibration_seconds"],
                          "evaluation_seconds": run["application_seconds"],
                          "subprocess_wall_seconds": process["wall_seconds"]})
    save_csv(HERE / "results/timings.csv", times, list(times[0]))
    table("timings", "llrr",
          ["条件", "手法", "校正 [s]", "評価 [s]"],
          [[labels[r["condition"]], "旧版" if r["method"] == "legacy" else "現行G=2",
            fmt(r["calibration_seconds"], 2), fmt(r["evaluation_seconds"], 2)]
           for r in times])


def main() -> None:
    report = qc_reaggregate()
    non_qc_reaggregate(report)
    build_tables()
    figures = verify_and_copy_figures()
    provenance_path = HERE / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["report_assets"] = {
        "runner_sha256": sha(source_path("run_comparison.py")),
        "builder_sha256": sha(Path(__file__)),
        "saved_real_report_sha256": sha(SAVED / "report.json"),
        "figures": figures,
        "tables": {p.name: sha(p) for p in sorted((HERE / "tables").glob("*.tex"))},
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(f"Built {len(figures)//2} PNG/SVG figure pairs and "
          f"{len(provenance['report_assets']['tables'])} tables")


if __name__ == "__main__":
    main()
