"""Parent-image-held-out prediction and an end-to-end synthetic region check.

Candidate construction on the held-out image sees only 28 context workers;
seven other workers are scored afterward. Nothing is selected using the scores.
The density is conditional on these context-defined local regions, not a density
for an entire microscope image or a test of biological correctness.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from configuration_model import (Fit, Observation, inside, infer, posterior, prepare, Settings, summarize)
from local_regions import load_regions, regions_from_views, save_regions, _area, _intersect
from run_configuration_pilot import save_fit
from data import (Answer, View, QC_IDS, DEFAULT_DATA_ROOT, load_image_views,
                  select_qc_workers, write_csv, write_json)


def project_answers(regions, views):
    """Keep fixed regions/candidates; select original responses by geometry only."""
    result, audit = [], {"partial_answers": 0, "answers": 0, "clicks": 0}
    for region in regions:
        available = {}
        for view in views:
            vb = (view.origin_x, view.origin_y, view.origin_x + view.width, view.origin_y + view.height)
            visible = _intersect(region.bounds, vb)
            if visible is None:
                continue
            b = region.bounds
            margin = min(b[0]-vb[0],b[1]-vb[1],vb[2]-b[2],vb[3]-b[3])
            rank = (-_area(visible),-margin,view.view_id)
            for a in view.answers:
                points = a.points + [view.origin_x, view.origin_y]
                observed = Observation(a.worker_id, points[inside(points,b)], vb, view.view_id)
                available.setdefault(a.worker_id,[]).append((rank, observed))
        obs = tuple(min(options,key=lambda x:x[0])[1] for _,options in sorted(available.items()))
        result.append(replace(region, observations=obs))
        audit["answers"] += len(obs)
        audit["clicks"] += sum(len(a.points) for a in obs)
        b=region.bounds
        audit["partial_answers"] += sum(not (a.bounds[0]<=b[0] and a.bounds[1]<=b[1]
                                             and a.bounds[2]>=b[2] and a.bounds[3]>=b[3]) for a in obs)
    return tuple(result),audit


def baseline_weights(region):
    from configuration_model import point_key
    lookup = {point_key(s): k for k,s in enumerate(region.candidates)}
    count = np.full(len(lookup),1/len(lookup))
    for a in region.observations:
        count[lookup[point_key(a.points)]] += 1
    return count/count.sum()


def prediction(output, selected_fold=None):
    destination = output/"heldout_image_validation"
    destination.mkdir(parents=True,exist_ok=True)
    regions = load_regions(output/"real/training/regions.json")
    workers,_ = select_qc_workers(DEFAULT_DATA_ROOT)
    scores,folds = [],[]
    for i,image in enumerate(QC_IDS):
        if selected_fold is not None and i+1 != selected_fold:
            continue
        held = workers[i*7:(i+1)*7]
        context = tuple(w for w in workers if w not in held)
        train = tuple(r for r in regions if r.image_id != image)
        fitted,status = save_fit(destination/f"fold_{i+1}",train)
        context_views=load_image_views(DEFAULT_DATA_ROOT,image,context,require_complete=True)
        held_views=load_image_views(DEFAULT_DATA_ROOT,image,held,require_complete=True)
        context_regions,audit=regions_from_views(context_views,settings=fitted.settings)
        target_regions,held_audit=project_answers(context_regions,held_views)
        folder=destination/f"fold_{i+1}"
        save_regions(folder/"context_regions.json",context_regions)
        save_regions(folder/"heldout_answers_fixed_candidates.json",target_regions)
        write_json(folder/"context_region_audit.json",audit)
        w_new,_=infer(context_regions,fitted)
        context_data=prepare(context_regions,worker_ids=workers,settings=fitted.settings)
        mean_table=fitted.error_tables.mean(axis=0)
        shared=np.tile(mean_table[None],(len(workers),1,1))
        w_shared,_=posterior(context_data,shared)
        target_data=prepare(target_regions,worker_ids=workers,settings=fitted.settings)
        fixed=np.tile((fitted.settings.pseudocounts/fitted.settings.prior_strength)[None],(len(workers),1,1))
        w_raw=tuple(baseline_weights(r) for r in context_regions)
        # Worker order must agree; do not silently index tables in another order.
        assert fitted.worker_ids == workers
        for r,(f,c,n,idx) in enumerate(zip(target_data.log_position_tail,target_data.underlying_bins,
                                           target_data.reported_bins,target_data.worker_indices)):
            predictions={}
            for method,tables,weights in (("configuration",fitted.error_tables,w_new),
                                          ("pooled_worker_table",shared,w_shared),
                                          ("simple_pooling",fixed,w_raw)):
                terms=f+np.log(tables[idx[:,None],c,n[:,None]])
                predictions[method]=logsumexp(terms+np.log(np.maximum(weights[r],np.finfo(float).tiny)),axis=1)
            for j,worker_idx in enumerate(idx):
                scores.append({"image_id":image,"region_id":context_regions[r].region_id,"worker_id":workers[worker_idx],
                               "point_count":len(target_regions[r].observations[j].points),
                               **{key:float(value[j]) for key,value in predictions.items()}})
        fold_scores=[s for s in scores if s["image_id"]==image]
        record={"heldout_image":image,"training_images":[name for name in QC_IDS if name!=image],
                "context_workers":list(context),"scored_workers":list(held),"training_regions":len(train),
                "context_regions":len(context_regions),"heldout_audit":held_audit,
                "heldout_image_used_for_worker_fit":False,"heldout_answers_used_for_regions_or_candidates":False,
                "mean_log_density":{name:float(np.mean([s[name] for s in fold_scores]))
                                    for name in ("configuration","pooled_worker_table","simple_pooling")}}
        folds.append(record)
        write_json(folder/"evaluation.json",record)
        write_csv(folder/"prediction_scores.csv",fold_scores)
        # Collect only completed evaluations. Interrupted folds are retained
        # and are never presented as converged or silently overwritten.
        import csv
        all_folds=[]
        all_scores=[]
        for completed in sorted(destination.glob("fold_*/evaluation.json")):
            score_path=completed.parent/"prediction_scores.csv"
            if not score_path.exists():
                continue
            all_folds.append(json.loads(completed.read_text()))
            with score_path.open() as stream:
                all_scores.extend(dict(row,**{k:float(row[k]) for k in ("configuration","pooled_worker_table","simple_pooling")})
                                  for row in csv.DictReader(stream))
        write_csv(destination/"prediction_scores.csv",all_scores)
        write_json(destination/"report.json",{"folds":all_folds,"completed_folds":len(all_folds),
            "interpretation":"Conditional response prediction on context-defined regions; not biological accuracy.",
            "mean_log_density":{name:float(np.mean([s[name] for s in all_scores]))
                                for name in ("configuration","pooled_worker_table","simple_pooling")}})
        print(json.dumps(record["mean_log_density"]),flush=True)


def end_to_end(output, name="end_to_end_synthetic"):
    destination=output/name
    destination.mkdir(parents=True,exist_ok=False)
    report={}
    for condition in ("merge_majority","split_majority"):
        folder=output/"synthetic_inputs"/condition
        payload=json.loads((folder/"qc_views.json").read_text())
        views=[]
        for row in payload["views"]:
            # Ten disjoint physical display fields in each synthetic parent image.
            x=int(row["view_id"])*100
            views.append(View(row["image_id"],row["view_id"],x,0,100,100,1000,100,
                              tuple(Answer(a["worker_id"],np.asarray(a["points"]).reshape(-1,2)) for a in row["answers"])))
        regions,audit=regions_from_views(views)
        write_json(destination/f"{condition}_region_audit.json",audit)
        fitted,status=save_fit(destination/condition,regions)
        payload=json.loads((folder/"evaluation_views.json").read_text())
        eval_views=tuple(View(row["image_id"],row["view_id"],int(row["view_id"])*100,0,100,100,1000,100,
                             tuple(Answer(a["worker_id"],np.asarray(a["points"]).reshape(-1,2)) for a in row["answers"]))
                         for row in payload["views"])
        eval_regions,eval_audit=regions_from_views(eval_views)
        weights,_=infer(eval_regions,fitted)
        rows=[summarize(r,w) for r,w in zip(eval_regions,weights)]
        write_json(destination/condition/"evaluation_outputs.json",rows)
        write_json(destination/condition/"evaluation_region_audit.json",eval_audit)
        report[condition]={"training_regions":len(regions),"evaluation_regions":len(eval_regions),
                           "expected_count_by_field":[sum(row["expected_count"] for row in rows
                                                          if 100*i <= (row["bounds"][0]+row["bounds"][2])/2 < 100*(i+1))
                                                       for i in range(10)],"fit":status}
        write_json(destination/"report.json",report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prediction","end-to-end"))
    p.add_argument("--results",type=Path,default=PROJECT_ROOT/"outputs/configuration_consensus_v2")
    p.add_argument("--name",default="end_to_end_synthetic",help="Fresh directory for a recorded preprocessing revision")
    p.add_argument("--fold",type=int,choices=range(1,6),help="Run one parent-image fold without restarting other folds")
    a=p.parse_args()
    if a.stage=="prediction":
        prediction(a.results,a.fold)
    else:
        end_to_end(a.results,a.name)


if __name__=="__main__":
    main()
