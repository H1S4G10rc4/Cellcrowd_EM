"""Audit saved pilot outputs without fitting or modifying those outputs."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from configuration_model import Settings, objective, prepare, summarize
from local_regions import load_regions, regions_from_views
from data import DEFAULT_DATA_ROOT, QC_IDS, load_image_views, select_qc_workers, write_json, write_csv

HERE = PROJECT_ROOT
ROOT = HERE / "outputs/configuration_consensus_v2"
DEST = ROOT / "final_audit"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_fit(folder, refresh=False):
    status = read(folder / "status.json")
    assert status["status"] == "converged", folder
    selected = status["selected_initialization"]
    parameters = np.load(folder / f"parameters_{selected}.npz")
    table = parameters["error_tables"]
    assert np.isfinite(table).all() and np.all(table > 0)
    np.testing.assert_allclose(table.sum(axis=2), 1, atol=1e-13)
    outputs = read(folder / f"posteriors_{selected}.json")
    smallest_step = 0.
    for attempt in status["attempts"]:
        a = attempt["initialization"]
        with (folder / f"history_{a}.csv").open() as stream:
            history = list(csv.DictReader(stream))
        values = np.array([float(r["objective"]) for r in history])
        assert np.isfinite(values).all()
        smallest_step = min(smallest_step, float(np.min(np.diff(values))))
        assert np.min(np.diff(values)) >= -1e-8 * max(1, abs(values[-1])), folder
        assert np.isclose(values[-1], attempt["objective"])
    assert status["objective"] == max(a["objective"] for a in status["attempts"] if a["converged"])
    for row in outputs:
        weights = np.asarray(row["weights"])
        assert np.isfinite(weights).all() and np.min(weights) >= 0
        assert np.isclose(weights.sum(), 1, atol=1e-12)
        distribution = {}
        for points, weight in zip(row["candidate_points"], weights):
            distribution[str(len(points))] = distribution.get(str(len(points)), 0.) + weight
        assert set(distribution) == set(row["count_distribution"])
        for n, value in distribution.items():
            assert np.isclose(value, row["count_distribution"][n], atol=1e-13)
        assert np.isclose(row["expected_count"], sum(int(n)*p for n,p in distribution.items()))
        assert row["map_candidate_index"] == int(weights.argmax())
    record = {"path": str(folder.relative_to(ROOT)), "regions": len(outputs),
              "selected_initialization": selected, "objective": status["objective"],
              "minimum_objective_step": smallest_step, "all_attempts_converged": all(a["converged"] for a in status["attempts"]),
              "near_optimal_start_disagreements": status["regions_with_start_disagreement"]}
    if refresh:
        regions = load_regions(folder / "regions.json")
        settings = Settings(**read(folder / "settings.json"))
        data = prepare(regions, tuple(parameters["worker_ids"].tolist()), settings)
        value, weights = objective(data, table)
        assert abs(value-status["objective"]) < 1e-7, (value,status["objective"])
        error = max(float(np.max(abs(w-np.array(r["weights"])))) for w,r in zip(weights,outputs))
        assert error < 1e-12
        record["recomputed_final_posterior_max_error"] = error
    return record


def main():
    if not (HERE / "docs/source_layout.json").is_file():
        raise SystemExit("This historical audit requires private results and relocation provenance; they are not published. Use the repository tests for clean-clone checks.")
    DEST.mkdir(exist_ok=True)
    protected = [p for p in ROOT.rglob("*") if p.is_file() and DEST not in p.parents]
    before = {str(p): sha(p) for p in protected}
    source_preservation = {}
    for snapshot in HERE.glob("outputs/configuration_consensus_v*/source_snapshot_*.json"):
        for name, expected in read(snapshot)["preserved"].items():
            assert historical_sha256(HERE/name) == expected, name
            source_preservation[name] = expected
    old_path=HERE / "outputs/method_schematics/previous_class_model/nature_png/manifest.json"
    if not old_path.exists():
        old_path=HERE / "outputs/method_schematics/nature_png/manifest.json"
    old_manifest = read(old_path)
    historical_checks = {p: historical_sha256(Path(p)) == value for p,value in old_manifest["source_sha256"].items()}
    assert all(historical_checks.values()), [p for p,ok in historical_checks.items() if not ok]

    tests = subprocess.run([sys.executable,"-m","unittest","-v","test_configuration_model"],cwd=TEST_ROOT,capture_output=True,text=True)
    (DEST/"tests.txt").write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0
    workers,_ = select_qc_workers(DEFAULT_DATA_ROOT)
    issues, mismatches = [],[]
    views = tuple(v for image in QC_IDS for v in load_image_views(DEFAULT_DATA_ROOT,image,workers,
                  issues=issues,mismatches=mismatches,require_complete=True))
    assert not issues, issues
    assert len(views)==245 and sum(len(v.answers) for v in views)==8575
    rebuilt,_ = regions_from_views(views)
    saved = load_regions(ROOT/"real/training/regions.json")
    assert len(saved)==len(rebuilt)==2204
    for a,b in zip(saved,rebuilt):
        assert a.region_id==b.region_id and a.bounds==b.bounds and a.image_id==b.image_id
        assert a.candidate_sources==b.candidate_sources
        for x,y in zip(a.candidates,b.candidates):
            np.testing.assert_array_equal(x,y)
        assert len(a.observations)==len(b.observations)
        for x,y in zip(a.observations,b.observations):
            assert (x.worker_id,x.bounds,x.view_id)==(y.worker_id,y.bounds,y.view_id)
            np.testing.assert_array_equal(x.points,y.points)
    write_json(DEST/"real_input_audit.json",{"issues":issues,"declared_count_mismatches":mismatches,
        "point_lists_are_authoritative":True,"original_answers":8575,"regions_reproduced_exactly":len(saved)})
    records=[audit_fit(ROOT/"real/training",refresh=True)]
    provenance = {}
    for inp in sorted((ROOT/"synthetic_inputs").iterdir()):
        name=inp.name
        legacy=read(ROOT/"results"/name/"legacy/run.json")
        assert legacy["status"]=="converged"
        for phase in ("qc","evaluation"):
            payload=read(inp/f"{phase}_views.json")
            assert legacy[f"{phase}_input_sha256"]==sha(inp/f"{phase}_views.json")
            expected=Counter((v["image_id"],v["view_id"],a["worker_id"],float(p[0]),float(p[1]))
                              for v in payload["views"] for a in v["answers"] for p in a["points"])
            with (inp/f"{phase}_points.csv").open() as stream:
                actual=Counter((r["img_id"],r["view_id"],r["worker_id"],float(r["x"]),float(r["y"]))
                               for r in csv.DictReader(stream))
            assert expected==actual, name
        records.append(audit_fit(ROOT/"results"/name/"configuration",refresh=True))
        if name in ("merge_majority","split_majority"):
            provenance[name]={p:read(inp/f"{p}_reference_not_used_by_methods.json")["seed"] for p in ("training","evaluation")}
        else:
            provenance[name]={"generation":"Reused saved deterministic input; no new generation seed",
                "training_input_sha256":sha(inp/"qc_views.json"),"evaluation_input_sha256":sha(inp/"evaluation_views.json")}
    write_json(DEST/"input_provenance_corrections.json",{"note":"The original definition.json generation_seed field recorded the base seed, not the actual per-condition seeds. Numerical input and output files are unchanged; use these seeds instead.","conditions":provenance})
    root=ROOT/"heldout_image_validation"
    scores=[]
    for i in range(1,6):
        fold=root/f"fold_{i}"
        records.append(audit_fit(fold))
        evaluation=read(fold/"evaluation.json")
        assert evaluation["heldout_image"] not in evaluation["training_images"]
        assert not set(evaluation["context_workers"]) & set(evaluation["scored_workers"])
        assert len(evaluation["context_workers"])+len(evaluation["scored_workers"])==35
        with (fold/"prediction_scores.csv").open() as stream:
            scores.extend(dict(r,**{k:float(r[k]) for k in ("configuration","pooled_worker_table","simple_pooling")}) for r in csv.DictReader(stream))
    means={k:float(np.mean([r[k] for r in scores])) for k in ("configuration","pooled_worker_table","simple_pooling")}
    report=read(root/"report.json")
    assert report["completed_folds"]==5
    for k,value in means.items():
        assert np.isclose(value,report["mean_log_density"][k],atol=1e-13)
    stratified=[]
    for label,subset in (("empty",[r for r in scores if int(r["point_count"])==0]),
                         ("nonempty",[r for r in scores if int(r["point_count"])>0])):
        stratified.append({"responses":label,"n":len(subset),
            **{k:float(np.mean([r[k] for r in subset])) for k in means}})
    write_csv(DEST/"heldout_stratified_scores.csv",stratified)
    for name in ("merge_majority","split_majority"):
        records.append(audit_fit(ROOT/"end_to_end_gap_preserving"/name,refresh=True))
    legacy=read(ROOT/"legacy_provenance.json")
    assert sha(Path(legacy["archive"]))==legacy["archive_sha256"]
    for name,value in legacy["source_sha256"].items():
        assert sha(HERE.parent/"Cellcrowd_EM_2025"/name)==value
    for path,value in before.items():
        assert sha(Path(path))==value, path
    # Guard against a rerun corrupting the existing result provenance.
    process=subprocess.run([sys.executable,str(source_path("run_configuration_pilot.py")),"real","--output",str(ROOT)],capture_output=True,text=True)
    assert process.returncode!=0 and "Existing results preserved" in process.stderr
    for path,value in before.items():
        assert historical_sha256(Path(path))==value,path
    figures=HERE/"outputs/method_schematics/configuration_png"
    figure_manifest=read(figures/"manifest.json")
    assert len(figure_manifest["checks"])==len(list(figures.glob("*.png")))==8
    assert len(list(figures.glob("*.csv")))==8
    assert not list(figures.glob("*.pdf")) and not list(figures.glob("*.svg"))
    from PIL import Image
    for entry in figure_manifest["checks"]:
        path=figures/f"{entry['figure']}.png"
        with Image.open(path) as im:
            assert im.size==(4800,2700) and im.mode=="RGB"
        assert all(entry[k] for k in ("all_text_black","text_inside_canvas","positioned_text_nonoverlapping","english_only"))
        assert "Arial" in entry["embedded_intermediate_fonts"]
    for path,value in figure_manifest["source_sha256"].items():
        assert historical_sha256(Path(path))==value,path
    published=read(figures/"published_paths.json")
    for entry in published["files"]:
        assert sha(Path(entry["path"]))==sha(Path(entry["source"]))==entry["sha256"]
    code=DEST/"source"
    code.mkdir(exist_ok=True)
    sources={}
    for name in ("configuration_model.py","local_regions.py","run_configuration_pilot.py",
                 "validate_configuration_pilot.py","test_configuration_model.py","make_configuration_figures.py","audit_configuration_results.py"):
        shutil.copy2(source_path(name),code/name)
        sources[name]=sha(source_path(name))
    write_csv(DEST/"fit_checks.csv",records)
    write_json(DEST/"audit.json",{"passed":True,"fit_checks":records,"real_regions":len(saved),
        "original_answers":8575,"source_count_mismatches":len(mismatches),"heldout_scored_answers":len(scores),
        "heldout_mean_log_density":means,"original_model_source_verified_against_archive":source_preservation,
        "heldout_stratified_scores":stratified,"figure_count":8,"published_figure_paths":published["files"],
        "historical_figure_sources_unchanged":historical_checks,"result_files_unchanged_during_audit":len(before),
        "current_source_sha256":sources,"settings":asdict(Settings()),
        "packages":{name:importlib.metadata.version(name) for name in ("numpy","scipy","matplotlib","pandas","hdbscan","Pillow")}})
    print(json.dumps({"passed":True,"fits":len(records),"heldout_mean_log_density":means,"input_count_mismatches":len(mismatches)}),flush=True)


if __name__=="__main__":
    main()
