"""Draw estimation and output assembly from an existing converged example.

Evaluate the fixed observation density and aggregate saved arrays only.
No EM iteration, fitting, or application to a new image is performed.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import gaussian_filter

import make_method_schematics as art

OUT = METHOD_OUTPUT_ROOT
ROOT = OUT.parents[2]
import model

CASE = ROOT / "comparison_report/connected_split_case"
RESULTS = CASE / "results/connected_split/current"
HASHES = {}
VALUES = {}


def register(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(register(path).read_text())


def point_tile(fig, rect, definition, points, color=art.BLUE):
    ax = art.tile(fig, rect, bounds=(10, 90, 90, 10))
    art.draw_regions(ax, definition)
    art.dots(ax, points, color, size=145)
    return ax


def stacked_bar(fig, rect, values, labels=False):
    ax = fig.add_axes(rect)
    start = 0.
    for value, color in zip(values, (art.BLUE, art.ORANGE)):
        ax.barh([0], [value], left=start, height=1, color=color)
        if labels and value > .15:
            ax.text(start+value/2, 0, f"{value:.2f}", ha="center", va="center",
                    color="white", fontsize=17)
        start += value
    ax.set_xlim(0, 1); ax.set_ylim(-.5, .5); ax.axis("off")
    return ax


def input_data():
    definition = read_json(CASE / "definition.json")
    run = read_json(RESULTS / "run.json")
    assert run["status"] == "converged"
    training = read_json(CASE / "synthetic_inputs/connected_split/qc_views.json")
    application = read_json(CASE / "synthetic_inputs/connected_split/evaluation_views.json")
    assert run["qc_input_sha256"] == HASHES[str(CASE / "synthetic_inputs/connected_split/qc_views.json")]
    assert run["evaluation_input_sha256"] == HASHES[str(CASE / "synthetic_inputs/connected_split/evaluation_views.json")]
    output = read_json(RESULTS / "evaluation_outputs.json")[0]
    assert output["status"] == "converged"
    fitted = {}
    with np.load(register(RESULTS / "fitted_parameters.npz"), allow_pickle=False) as a:
        for name in a.files:
            fitted[name] = a[name].copy()
    for source in (source_path("model.py"), source_path("data.py")):
        register(source)
    return definition, run, training, application, fitted, output


def estimation(definition, run, training, fitted):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for x, text in ((.143, "Observed answers"), (.506, "Iterative estimation"),
                    (.865, "Fitted quantities")):
        art.label(fig, x, .94, text, size=23)
    art.label(fig, .143, .882, "Saved synthetic example", size=16, color=art.GRAY)
    for x, name in ((.03, "Region 1"), (.155, "Region 2")):
        art.label(fig, x+.055, .798, name, size=17, color=art.GRAY)
    # Both region examples and all worker counts are taken from the saved input.
    for y, wid, count, color in ((.553, "A1", 7, art.BLUE), (.272, "B1", 3, art.ORANGE)):
        for x, view in zip((.03, .155), training["views"][:2]):
            answer = next(a for a in view["answers"] if a["worker_id"] == wid)
            assert sum(a["points"] == answer["points"] for a in view["answers"]) == count
            point_tile(fig, [x, y, .110, .223], definition, answer["points"], color)
        art.label(fig, .143, y-.022, f"{count} workers", size=20, color=color)
    art.label(fig, .143, .162, r"Keep whole point sets $S_{rk}$", size=17, color=art.GRAY)
    art.arrow(fig, (.279, .568), (.328, .568))

    art.label(fig, .505, .850, "Infer hidden choices", size=23, color=art.BLUE)
    art.label(fig, .505, .802, "Group, response path, configuration", size=17, color=art.GRAY)
    art.label(fig, .505, .765, "Orange crosses: observed points", size=15, color=art.GRAY)
    candidates = [definition["click_templates"][key] for key in ("separated", "merged")]
    response = training["views"][0]["answers"][0]["points"]
    assert response == candidates[0]
    normalized = tuple(np.asarray(c)/100 for c in candidates)
    log_f = model.log_f_for_points(np.asarray(response)/100, normalized)
    assert log_f[0] > log_f[1]
    grid = np.linspace(0, 1, 201)
    xx, yy = np.meshgrid(grid, grid)
    for k, x in enumerate((.354, .536)):
        ax = art.tile(fig, [x, .576, .137, .205], bounds=(10, 90, 88, 12))
        art.draw_regions(ax, definition)
        density = model.spatial_density(np.c_[xx.ravel(), yy.ravel()], normalized[k]).reshape(xx.shape)
        ax.contourf(xx*100, yy*100, density, levels=[.1, 1, 5, 15, 25, 70],
                    colors=["#EFF4F6", "#D6E5EC", "#AFCDDC", "#79AAC1", "#397E9E"], alpha=.92)
        ax.contour(xx*100, yy*100, density, levels=[5, 15, 25], colors=[art.BLUE], linewidths=.7)
        # The same actual answer is compared with each candidate's fixed density.
        points = np.asarray(response)
        ax.scatter(points[:, 0], points[:, 1], s=75, marker="+", color=art.HOT, linewidths=2.2, zorder=7)
        art.label(fig, x+.0685, .551, rf"$F(Y\mid S_{k+1})$", size=19)
    art.label(fig, .505, .506, "Compare counts and positions", size=17, color=art.GRAY)
    art.label(fig, .505, .449, r"$q_{jg}$  +  expected assignments", size=20, color=art.BLUE)

    # These arrows depict the existing E/M cycle; no iterative values are invented.
    art.arrow(fig, (.663, .535), (.622, .326), curve=-.27)
    art.arrow(fig, (.393, .297), (.365, .559), curve=-.48)
    art.label(fig, .505, .340, "Update parameters", size=23)
    art.label(fig, .505, .276, r"$\omega,\;\rho,\;\pi$", size=29)
    art.label(fig, .505, .223, "Weighted counts + priors", size=17, color=art.GRAY)
    art.label(fig, .505, .151, "Repeat until stable", size=19)
    art.arrow(fig, (.691, .519), (.749, .519))
    art.label(fig, .72, .565, "Converged", size=16, color=art.GRAY)

    art.label(fig, .865, .851, r"Group membership $q_{jg}$", size=19)
    ax = fig.add_axes([.773, .760, .184, .048])
    for j, q in enumerate(fitted["q"]):
        ax.bar(j, q[0], width=.80, color=art.BLUE)
        ax.bar(j, q[1], bottom=q[0], width=.80, color=art.ORANGE)
    ax.set_xlim(-.6, len(fitted["q"])-.4); ax.set_ylim(0, 1); ax.axis("off")
    art.label(fig, .833, .721, "7 workers", size=16, color=art.GRAY)
    art.label(fig, .932, .721, "3 workers", size=16, color=art.GRAY)

    art.label(fig, .865, .651, r"Group proportions $\omega_g$", size=19)
    stacked_bar(fig, [.773, .582, .184, .041], fitted["omega"], labels=True)

    art.label(fig, .865, .471, "Group-consistent\nresponse weight ρ", size=18, linespacing=1.3)
    for y, indices, color in ((.362, slice(0,7), art.BLUE), (.302, slice(7,10), art.ORANGE)):
        rhos = fitted["rho"][indices]
        assert np.allclose(rhos, rhos[0])
        value = float(rhos[0])
        ax = fig.add_axes([.773, y, .133, .030])
        ax.barh([0], [1], color=art.LIGHT, height=1)
        ax.barh([0], [value], color=color, height=1)
        ax.set_xlim(0,1); ax.set_ylim(-.5,.5); ax.axis("off")
        art.label(fig, .937, y+.015, f"{value:.3f}", size=17)

    art.label(fig, .865, .234, r"Candidate weights $\pi_{rgk}$", size=19)
    for g, y in enumerate((.155, .080)):
        stacked_bar(fig, [.833, y, .124, .034], fitted["pi_qc"][0,g], labels=False)
        art.label(fig, .796, y+.017, f"Group {g+1}", size=16)
    art.save(fig, "model_inference")
    VALUES["model_inference"] = {
        "role": "Illustration of the existing EM/MAP estimation loop, using a converged saved fit at right.",
        "response_used_for_likelihood_example": response,
        "candidate_configurations": candidates, "example_log_likelihoods": log_f.tolist(),
        "omega": fitted["omega"].tolist(), "q": fitted["q"].tolist(),
        "rho": fitted["rho"].tolist(), "pi_first_training_region": fitted["pi_qc"][0].tolist(),
        "run": run,
        "note": "The likelihood example illustrates F, not a posterior probability. The E-step combines F with current parameters to infer group membership and response-path/candidate assignments. Priors enter the M-step. No intermediate trajectory was fabricated."
    }


def outputs(definition, run, application, fitted, output):
    q_by_worker = dict(zip(run["worker_ids"], fitted["q"]))
    view = application["views"][0]
    mixing = np.mean([q_by_worker[a["worker_id"]] for a in view["answers"]], axis=0)
    pi = np.asarray(output["pi"])
    weights = mixing @ pi
    assert np.allclose(mixing, output["v"], atol=1e-12)
    assert np.allclose(weights, output["weights"], atol=1e-12)
    candidates = output["candidate_points_pixels"]
    assert [len(c) for c in candidates] == [2,1]
    for v in application["views"]:
        assert {tuple(map(tuple,a["points"])) for a in v["answers"]} == {tuple(map(tuple,c)) for c in candidates}
    counts = np.array([len(c) for c in candidates])
    probabilities = [float(weights[counts==n].sum()) for n in (1,2,3)]
    mass = art.point_mass(candidates, weights, 100)
    smooth = gaussian_filter(mass, sigma=3, mode="reflect")
    assert np.isclose(mass.sum(), output["expected_count"])
    assert probabilities[2] == 0

    fig = plt.figure(figsize=(16,9), dpi=120)
    art.label(fig, .201, .94, "Group-specific estimates", size=23)
    art.label(fig, .547, .94, "Combined candidates", size=23)
    art.label(fig, .201, .894, "Saved synthetic example", size=16, color=art.GRAY)
    art.label(fig, .201, .858, r"Worker $q,\rho$ fixed", size=16, color=art.GRAY)
    art.label(fig, .547, .882, r"$w_{rk}=\sum_g v_{rg}\,\pi_{rgk}$", size=23)
    for g, y in enumerate((.603,.252)):
        art.label(fig, .201, y+.219, rf"Group {g+1}    $v_{{r{g+1}}}={mixing[g]:.2f}$", size=20)
        for k, x in enumerate((.040,.222)):
            point_tile(fig, [x,y,.137,.194], definition, candidates[k], art.BLUE if k==0 else art.ORANGE)
            art.label(fig, x+.0685, y-.025, rf"$\pi_{{r{g+1}{k+1}}}={pi[g,k]:.3f}$", size=21,
                       color=art.BLUE if k==0 else art.ORANGE)
    art.label(fig, .201, .141, r"$v_{rg}=\mathrm{mean}_{j\in J_r}\;q_{jg}$", size=23)
    art.label(fig, .201, .087, "From participating workers", size=17, color=art.GRAY)

    art.arrow(fig, (.371,.682), (.411,.540))
    art.arrow(fig, (.371,.330), (.411,.520))
    art.arrow(fig, (.413,.529), (.451,.529))
    for k, y in enumerate((.603,.252)):
        point_tile(fig, [.460,y,.175,.208], definition,candidates[k],art.BLUE if k==0 else art.ORANGE)
        n=len(candidates[k])
        art.label(fig,.547,y+.215,f"{n} point"+("s" if n>1 else ""),size=20,color=art.BLUE if k==0 else art.ORANGE)
        art.label(fig,.547,y-.032,f"{weights[k]:.3f}",size=28,color=art.BLUE if k==0 else art.ORANGE)
    art.label(fig,.547,.105,"Whole configurations retained",size=17,color=art.GRAY)

    # Both summaries use ALL candidates: first collect the two configurations,
    # then branch. A separate arrow from each candidate would be misleading.
    for xs, ys in (((.645,.684),(.698,.698)),
                   ((.645,.684),(.346,.346)),
                   ((.684,.684),(.346,.698))):
        fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure,
                                 color="#879298", linewidth=1.25))
    art.arrow(fig, (.684,.526), (.744,.702))
    art.arrow(fig, (.684,.526), (.744,.335))
    art.label(fig,.854,.900,"Point-count probability",size=22)
    ax=fig.add_axes([.771,.600,.181,.230])
    ax.bar([1,2,3],probabilities,color=[art.ORANGE,art.BLUE,art.LIGHT],width=.57)
    ax.set_ylim(0,1); ax.set_xlim(.45,3.55)
    ax.set_xticks([1,2,3]); ax.set_yticks([0,.5,1])
    ax.tick_params(labelsize=16,length=4)
    ax.set_xlabel("Points",fontsize=17,labelpad=5)
    ax.spines[["top","right"]].set_visible(False)
    for x,value in zip((1,2,3),probabilities):
        ax.text(x,value+.045,f"{value:.3f}",ha="center",fontsize=17,color=art.INK)

    art.label(fig,.854,.480,"Mean point map",size=22)
    ax=art.tile(fig,[.758,.184,.194,.259],bounds=(10,90,90,10))
    art.draw_regions(ax,definition)
    norm=Normalize(0,.015)
    rgba=plt.colormaps["inferno"](norm(smooth))
    rgba[...,3]=np.clip(smooth/.003,0,.96)
    ax.imshow(rgba,extent=(0,100,100,0),interpolation="bilinear",zorder=3)
    art.label(fig,.854,.157,f"Mean count  {mass.sum():.2f}",size=19)
    cax=fig.add_axes([.802,.092,.107,.014])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm,"inferno"),cax=cax,orientation="horizontal",ticks=[0,.015])
    cb.ax.tick_params(labelsize=12,length=2)
    cb.outline.set_visible(False)
    art.label(fig,.854,.047,"Point mass / pixel",size=15,color=art.GRAY)
    art.save(fig,"inference_to_outputs")
    VALUES["inference_to_outputs"]={
        "role":"Assembly of final outputs from saved worker memberships and saved region-specific candidate weights.",
        "saved_output":output,"reconstructed_group_mix":mixing.tolist(),"reconstructed_weights":weights.tolist(),
        "point_counts_shown":[1,2,3],"point_count_probabilities":probabilities,
        "point_mass":mass.tolist(),"display_mass":smooth.tolist(),
        "smoothing":{"method":"scipy.ndimage.gaussian_filter","sigma_pixels":3,"mode":"reflect",
                     "display_range":[0,.015],"alpha_rule":"clip(smoothed_mass/0.003,0,0.96)"},
        "note":"The group mix v is mean posterior membership among participating workers, not the fitted group prior omega. These pi values were already saved after fitting this evaluation region with worker q and rho fixed. Rho affects fitting pi but is not multiplied into the final map again. The mean map has three peaks, but the three-point candidate has probability zero."
    }


def main():
    plt.rcParams.update({"mathtext.fontset":"dejavusans","mathtext.default":"regular"})
    definition,run,training,application,fitted,output=input_data()
    estimation(definition,run,training,fitted)
    outputs(definition,run,application,fitted,output)
    for path,digest in HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (OUT/"inference_source_values.json").write_text(json.dumps(VALUES,indent=2)+"\n")
    manifest={
        "figures":{
            "model_inference":"Observed whole answers are evaluated under the fixed observation model; posterior assignments and parameters are alternately updated until convergence. The right side shows an existing converged fit.",
            "inference_to_outputs":"Saved group-specific weights and participating-worker memberships yield whole-configuration weights, a point-count distribution, and a mean point-mass map."
        },
        "outputs":["model_inference.png","model_inference.svg","inference_to_outputs.png","inference_to_outputs.svg"],
        "style":{"size":[1920,1080],"ratio":"16:9","language":"English","font":"Liberation Sans","svg_text":"editable"},
        "processing":"Read saved data and converged fits. Evaluate the fixed observation density for one illustrative comparison. Aggregate saved arrays, apply display-only smoothing, and draw. No model fitting or new image inference.",
        "parameter_learning":"EM/MAP estimates omega, rho and pi; q is a derived posterior group-membership probability. The E-step uses the full group-dependent and group-independent mixture, not F alone. The displayed F comparison is only an illustration of its count/location component.",
        "new_region_rule":"Keep learned worker q and rho fixed; estimate only region-specific pi. The second figure uses the already saved result of that operation.",
        "candidate_policy":"Configurations retain all jointly observed points; no new combinations or shifted candidate coordinates are created.",
        "synthetic_geometry":"Saved overlapping ellipse outlines are context for the example, not biological ground truth and not a model input.",
        "source_sha256":HASHES,"source_files_unchanged":True,"checks":art.CHECKS,
    }
    (OUT/"inference_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"outputs":manifest["outputs"],"checks":art.CHECKS,"source_files_unchanged":True},indent=2))


if __name__=="__main__":
    main()
