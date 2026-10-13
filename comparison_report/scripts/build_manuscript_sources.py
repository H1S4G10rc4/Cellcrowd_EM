"""Collect figure-source tables from saved, same-input runs without refitting."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "point_annotation_trial" / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import csv
import json
from pathlib import Path
import shutil


ROOT = REPORT_ROOT
SOURCES = ROOT / "figure_sources"
CONDITIONS = ("A_location", "B_count", "C_switching", "D_complementary")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_saved_runs() -> None:
    for condition in CONDITIONS:
        pair = []
        for method in ("legacy", "current"):
            status = read_json(ROOT / "results" / condition / method / "status.json")
            assert status["status"] == "success", (condition, method)
            run = status["run"]
            assert run["status"] == "converged", (condition, method)
            pair.append(run)
        assert pair[0]["qc_input_sha256"] == pair[1]["qc_input_sha256"]
        assert pair[0]["evaluation_input_sha256"] == pair[1]["evaluation_input_sha256"]
    pair = [read_json(ROOT / "connected_split_case/results/connected_split" /
                      method / "run.json") for method in ("legacy", "current")]
    assert all(row["status"] == "converged" for row in pair)
    assert pair[0]["qc_input_sha256"] == pair[1]["qc_input_sha256"]
    assert pair[0]["evaluation_input_sha256"] == pair[1]["evaluation_input_sha256"]


def main() -> None:
    verify_saved_runs()
    SOURCES.mkdir(exist_ok=True)
    for source, target in (
        (ROOT / "results/synthetic_by_view.csv", "synthetic_candidate_weights.csv"),
        (ROOT / "results/legacy_native_scores_by_view.csv", "legacy_native_scores.csv"),
        (ROOT / "results/worker_score_summary.csv", "worker_group_scores.csv"),
        (ROOT / "results/real_qc_heldout.csv", "qc_holdout.csv"),
        (ROOT.parent / "point_annotation_trial/outputs/convergence_extension/"
         "non_qc_one_image/comparison.csv", "non_qc_49_views.csv"),
    ):
        shutil.copyfile(source, SOURCES / target)

    summary = read_json(ROOT / "connected_split_case/summary.json")
    definition = read_json(ROOT / "connected_split_case/definition.json")
    rows = []
    for name, count, workers in (("separated", 2, 7), ("merged", 1, 3)):
        rows.append({"configuration": name, "point_count": count,
                     "workers": workers,
                     "known_annotation_frequency": definition["known_group_frequency"][name],
                     "current_state_weight": summary["current_weights"][name]})
    save_rows(SOURCES / "connected_split.csv", rows, list(rows[0]))
    positions = ("left", "overlap", "right")
    old_rows = [{"position": name, "x_pixel": x, "y_pixel": 50,
                 "legacy_native_score": score,
                 "legacy_final_cluster_id": summary["legacy_cluster_id_by_x"][str(float(x))]}
                for name, x, score in zip(positions, (35, 50, 65),
                                          summary["legacy_scores_at_left_overlap_right"])]
    save_rows(SOURCES / "connected_split_legacy.csv", old_rows, list(old_rows[0]))

    overlay = read_json(ROOT / "real_overlay_provenance.json")
    save_rows(SOURCES / "real_overlay_views.csv", overlay["views"],
              list(overlay["views"][0]))

    with (ROOT / "results/synthetic_summary.csv").open(newline="", encoding="utf-8") as stream:
        input_rows = list(csv.DictReader(stream))
    comparison = []
    for condition in CONDITIONS:
        reference = ("observed_judgment" if condition == "D_complementary"
                     else "stable_10_workers" if condition == "C_switching"
                     else "known_group_frequency")
        by_method = {r["method"]: r for r in input_rows
                     if r["condition"] == condition and r["reference"] == reference}
        simple = by_method["simple_aggregation"]
        current = by_method["current"]
        known = 0.5 if condition == "D_complementary" else 0.7
        comparison.append({"condition": condition,
                           "reference": reference,
                           "known_or_observed_A_frequency": known,
                           "simple_A_weight": simple["weight_A"],
                           "current_A_weight": current["weight_A"],
                           "simple_TV_to_reference": simple["tv_to_judgment_reference"],
                           "current_TV_to_reference": current["tv_to_judgment_reference"],
                           "simple_expected_count": simple["expected_count"],
                           "current_expected_count": current["expected_count"]})
    save_rows(SOURCES / "synthetic_common_quantities.csv", comparison,
              list(comparison[0]))


if __name__ == "__main__":
    main()
