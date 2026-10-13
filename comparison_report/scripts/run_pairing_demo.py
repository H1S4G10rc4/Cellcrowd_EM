"""Small, same-input check of two different corner pairings.

The shared 50-view QC is the previously saved 7/3 location condition. In the
evaluation views every worker gives each of the two pairs five times, making
the marginal weighted clicks at every corner identical between conditions.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "point_annotation_trial" / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_comparison as rc


ROOT = REPORT_ROOT / "pairing_demo"
WORKERS = [f"A{i}" for i in range(1, 8)] + [f"B{i}" for i in range(1, 4)]
POINTS = {"TL": [25.0, 25.0], "TR": [75.0, 25.0],
          "BL": [25.0, 75.0], "BR": [75.0, 75.0]}
PAIRINGS = {
    "diagonal": (("TL", "BR"), ("TR", "BL")),
    "vertical": (("TL", "BL"), ("TR", "BR")),
}


def payload(name: str, *, qc: bool) -> dict:
    rows = []
    count = 50 if qc else 10
    for index in range(count):
        if qc:
            image_id = rc.QC_IDS[index // 10]
            answers = [{"worker_id": worker,
                        "points": [[20.0, 50.0]] if j < 7 else [[80.0, 50.0]]}
                       for j, worker in enumerate(WORKERS)]
        else:
            image_id = f"PAIR_{name}.png"
            pair_a, pair_b = PAIRINGS[name]
            answers = [{"worker_id": worker,
                        "points": [POINTS[k] for k in
                                   (pair_a if (j < 5) == (index % 2 == 0) else pair_b)]}
                       for j, worker in enumerate(WORKERS)]
        rows.append({"image_id": image_id, "view_id": str(index % 10),
                     "width": 100, "height": 100, "answers": answers})
    return {"phase": "qc" if qc else "evaluation",
            "worker_ids": WORKERS, "views": rows}


def main() -> None:
    if ROOT.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite: {ROOT}")
    rc.check_legacy_archive()
    ROOT.mkdir(parents=True)
    for name in PAIRINGS:
        folder = ROOT / "synthetic_inputs" / name
        folder.mkdir(parents=True)
        rc.write_json(folder / "qc_views.json", payload(name, qc=True))
        rc.write_json(folder / "evaluation_views.json", payload(name, qc=False))
        for phase in ("qc", "evaluation"):
            views, _ = rc.load_views(folder / f"{phase}_views.json")
            rc.point_table(views, folder / f"{phase}_points.csv")
        rc.write_json(folder / "definition.json", {
            "seed": None, "canvas_pixels": [100, 100], "workers": WORKERS,
            "qc": "5 images x 10 views; seven workers choose (20,50), three choose (80,50)",
            "evaluation": "10 views; five workers per pair, assignment reversed on odd views",
            "pairings": PAIRINGS[name], "corner_pixels": POINTS,
            "reference_used_for_fitting": False,
            "purpose": "Isolate pairing representation with identical per-point click frequencies",
        })
    hashes = {}
    for name in PAIRINGS:
        folder = ROOT / "synthetic_inputs" / name
        hashes[name] = {phase: rc.digest(folder / f"{phase}_views.json")
                        for phase in ("qc", "evaluation")}
    rc.write_json(ROOT / "provenance.json", {
        "legacy": rc.check_legacy_archive(),
        "current": {"G": 2, "starts": rc.STARTS,
                    "max_iterations": rc.CURRENT_MAX_ITERATIONS,
                    "model_sha256": rc.digest(source_path("model.py")),
                    "data_sha256": rc.digest(source_path("data.py"))},
        "configs": {"hdbscan": rc.asdict(rc.HDBSCAN_CONFIG),
                    "gmm": rc.asdict(rc.GMM_CONFIG),
                    "em": rc.asdict(rc.EM_CONFIG),
                    "heatmap": rc.asdict(rc.HEATMAP_CONFIG)},
        "input_sha256": hashes,
        "same_qc_for_both_conditions": hashes["diagonal"]["qc"] == hashes["vertical"]["qc"],
        "group_labels_or_reference_passed_to_models": False,
    })
    for name in PAIRINGS:
        for method, function in (("legacy", rc.run_legacy), ("current", rc.run_current)):
            print(f"Running {name} {method}", flush=True)
            try:
                result = function(ROOT, name)
                rc.write_json(ROOT / "results" / name / method / "status.json",
                              {"status": "success", "run": result})
            except Exception as exc:
                rc.write_json(ROOT / "results" / name / method / "status.json",
                              {"status": "failed", "reason": str(exc)})
                raise
    aggregate()


def aggregate() -> None:
    rows = []
    maps = []
    for name in PAIRINGS:
        base = ROOT / "results" / name
        old = np.load(base / "legacy" / "native_heatmaps.npz")["evaluation"]
        new = json.loads((base / "current" / "evaluation_outputs.json").read_text())
        workers = pd.read_csv(base / "legacy" / "worker_parameters.csv")
        canonical = [json.dumps(points) for points in new[0]["candidate_points_pixels"]]
        aligned = []
        for row in new:
            keys = [json.dumps(points) for points in row["candidate_points_pixels"]]
            aligned.append([row["weights"][keys.index(key)] for key in canonical])
        mean_weights = np.mean(np.array(aligned), axis=0)
        if not np.array_equal(old, np.repeat(old[:1], len(old), axis=0)):
            raise RuntimeError(f"Legacy map varies across views: {name}")
        rows.append({"condition": name, "view_count": len(new),
                     "candidate_1": json.dumps(new[0]["candidate_points_pixels"][0]),
                     "candidate_2": json.dumps(new[0]["candidate_points_pixels"][1]),
                     "mean_weight_1": mean_weights[0], "mean_weight_2": mean_weights[1],
                     "view0_weight_1": new[0]["weights"][0],
                     "view0_weight_2": new[0]["weights"][1],
                     "mean_expected_points": float(np.mean([row["expected_count"] for row in new])),
                     "legacy_map_min": float(old.min()),
                     "legacy_map_max": float(old.max()),
                     "legacy_worker_weight_range": float(workers["weight_positive"].max()
                                                          - workers["weight_positive"].min())})
        maps.append(old[0])
    difference = np.abs(maps[0] - maps[1])
    pd.DataFrame(rows).to_csv(ROOT / "summary.csv", index=False)
    rc.write_json(ROOT / "map_comparison.json", {
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "exactly_equal": bool(np.array_equal(maps[0], maps[1])),
        "definition": "Unmodified legacy native evaluation heatmap for view 0 in each condition; all ten views are exactly identical within condition",
    })
    print((ROOT / "summary.csv").read_text(), flush=True)
    print((ROOT / "map_comparison.json").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    aggregate() if args.aggregate_only else main()
