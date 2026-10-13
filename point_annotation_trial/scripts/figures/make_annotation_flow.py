"""Show actual crowded input -> recorded whole answers -> comparisons -> output.

This is a read-only reaggregation of an existing converged fit, not a new fit.
The selected examples expose different point counts; the calculations retain
every recorded answer and candidate, including empty answers.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import softmax

import make_method_schematics as art

HERE = METHOD_OUTPUT_ROOT
ROOT = HERE.parents[2]
OUT = HERE / "concept_png"
SAVED = ROOT / "point_annotation_trial/outputs/convergence_extension/qc_final"
from project_paths import FONT_ROOT
from data import DEFAULT_DATA_ROOT
FONT = FONT_ROOT / "Arial.TTF"
import model

COLORS = ("#287395", "#BC714E", "#338579")
MARKERS = ("o", "+", "D")
GRAY = "#899197"
HASHES = {}


def register(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def text(fig, x, y, label, size=19, **kwargs):
    return fig.text(x, y, label, ha="center", va="center", fontsize=size,
                    color="black", **kwargs)


def arrow(fig, start, end, curve=0, head=True, color="#7D898F"):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
        arrowstyle="-|>" if head else "-", mutation_scale=16, linewidth=1.2,
        color=color, connectionstyle=f"arc3,rad={curve}"))


def image_tile(fig, rect, raw, normalization):
    ax = art.tile(fig, rect, bounds=(0,80,80,0), frame=True)
    ax.imshow(raw, cmap="gray", norm=normalization, extent=(0,80,80,0),
              interpolation="nearest")
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAB2B6")
    return ax


def example_points(ax, points, number, on_image=False):
    """Different marker shapes expose nearby answers without moving any point."""
    p=np.asarray(points).reshape(-1,2)
    col,marker=COLORS[number],MARKERS[number]
    if marker=="o":
        ax.scatter(p[:,0],p[:,1],s=85 if on_image else 67,marker=marker,
                   facecolors="none",edgecolors=col,linewidths=1.6,zorder=6)
    elif marker=="+":
        if on_image:
            ax.scatter(p[:,0],p[:,1],s=52,marker=marker,c="white",linewidths=2.7,zorder=7)
        ax.scatter(p[:,0],p[:,1],s=52 if on_image else 67,marker=marker,c=col,
                   linewidths=1.65,zorder=8)
    else:
        ax.scatter(p[:,0],p[:,1],s=42 if on_image else 57,marker=marker,c=col,
                   edgecolors="white",linewidths=.65,zorder=9)


def save(fig, metadata):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Inspect only displayed text: ticks belonging to disabled axes are excluded.
    displayed = list(fig.texts)
    for ax in fig.axes:
        displayed.extend(ax.texts)
        if ax.axison:
            displayed += [ax.xaxis.label, ax.yaxis.label]
            displayed += ax.get_xticklabels() + ax.get_yticklabels()
    displayed = [t for t in displayed if t.get_visible() and t.get_text()]
    for label in displayed:
        value = label.get_text()
        assert not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value)
        assert "QC" not in value and "$" not in value
        assert mpl.colors.to_rgba(label.get_color())[:3] == (0.,0.,0.)
        assert Path(font_manager.findfont(label.get_fontproperties())).resolve() == FONT.resolve()
        box = label.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, value
    overlaps = []
    for i, first in enumerate(displayed):
        a = first.get_window_extent(renderer)
        for second in displayed[i+1:]:
            b = second.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                overlaps.append([first.get_text(), second.get_text()])
    assert not overlaps, overlaps
    target = OUT / "annotation_flow.png"
    fig.savefig(target, dpi=300, facecolor="white")
    with Image.open(target) as im:
        assert im.size == (4800,2700)
        rgb = im.convert("RGB")
    rgb.save(target, dpi=(300,300))
    plt.close(fig)
    for path, expected in HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    metadata.update({"source_sha256": HASHES, "sources_unchanged": True,
        "figure": target.name,
        "checks": {"size": [4800,2700], "ratio": "16:9", "format": "PNG",
                   "mode": "RGB", "dpi": 300, "font": "Arial", "text": "black",
                   "language": "English", "no_equations": True,
                   "text_inside_canvas": True, "no_text_overlaps": True},
        "displayed_text": [t.get_text() for t in displayed]})
    (OUT / "annotation_flow_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(target)


def main():
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    mpl.rcParams.update({"font.family": "Arial", "text.color": "black",
        "font.size": 18, "axes.labelcolor": "black", "axes.edgecolor": "#9DA6AB",
        "axes.linewidth": .75, "xtick.color": "black", "ytick.color": "black",
        "figure.facecolor": "white", "savefig.facecolor": "white"})
    with gzip.open(register(SAVED / "qc_inputs.json.gz"), "rt") as f:
        saved = json.load(f)
    with np.load(register(SAVED / "G2_ospa_pair_fit.npz"), allow_pickle=False) as f:
        fit_metadata = json.loads(str(f["metadata"]))
        assert fit_metadata["status"] == "converged"
        assert fit_metadata["inputs_sha256"] == HASHES[str(SAVED / "qc_inputs.json.gz")]
        q, rho, pi = f["q"].copy(), f["rho"].copy(), f["pi"].copy()
    index, view = next((i,v) for i,v in enumerate(saved["views"])
        if (v["image_id"],v["view_id"]) == ("QC3.png","280_280_80_80"))
    answers, candidates = view["answers"], view["candidates"]
    points = [np.array(c["points"]).reshape(-1,2) for c in candidates]
    q_by_worker = dict(zip(saved["worker_ids"],q))
    mix = np.mean([q_by_worker[a["worker_id"]] for a in answers],axis=0)
    weights = mix @ pi[index,:,:len(candidates)]
    assert np.isclose(weights.sum(),1)
    # These are examples of high-weight candidates, not a selected biological truth.
    order = np.argsort(-weights,kind="stable")
    selected = list(map(int, order[:3]))
    selected_workers = [candidates[k]["source_worker_ids"][0] for k in selected]
    selected_answers = [next(a for a in answers if a["worker_id"] == w) for w in selected_workers]
    worker_color = dict(zip(selected_workers,COLORS))
    # Keep exact duplicates as separate answers, but only one candidate column.
    keys = [tuple(sorted(map(tuple,p))) for p in points]
    assert len(set(keys)) == len(candidates)
    for a in answers:
        assert tuple(sorted(map(tuple,a["points"]))) in keys
    assert len(answers) == 35 and len(candidates) == 34
    assert sum(c["support_count"] for c in candidates) == len(answers)

    raw_path = register(DEFAULT_DATA_ROOT / view["image_id"] / view["image_id"])
    with Image.open(raw_path) as im:
        x0,y0 = view["origin_x"],view["origin_y"]
        raw = np.array(im.convert("L").crop((x0,y0,x0+80,y0+80)))
    lo,hi = np.percentile(raw,[1,99.5])
    image_norm = PowerNorm(.72,vmin=lo,vmax=hi)
    source_model = register(source_path("model.py"))
    register(source_path("data.py"))
    normalized = tuple(p / 80 for p in points)
    log_f = np.vstack([model.log_f_for_points(np.array(a["points"]).reshape(-1,2)/80,
                                             normalized) for a in answers])
    relative_match = softmax(log_f,axis=1)
    row_order = [next(i for i,a in enumerate(answers) if a["worker_id"] == w) for w in selected_workers]
    row_order += [i for i in range(len(answers)) if i not in row_order]
    shown_match = relative_match[np.ix_(row_order,order)]
    mass = art.point_mass(points,weights,80)
    smooth = gaussian_filter(mass,1.25,mode="reflect")
    assert np.isclose(mass.sum(),sum(len(p)*w for p,w in zip(points,weights)))

    fig = plt.figure(figsize=(16,9),dpi=120)
    for x,label in ((.132,"All annotations"),(.398,"Whole answers"),
                    (.646,"Compare and fit"),(.891,"Weighted outputs")):
        text(fig,x,.936,label,size=22)
    text(fig,.132,.850,"One image region",size=18)
    text(fig,.398,.850,"Keep annotator IDs",size=18)
    text(fig,.646,.850,"Counts and positions",size=18)

    # Start with all 309 original points; three examples are highlighted by ID.
    ax=image_tile(fig,[.025,.378,.214,.380],raw,image_norm)
    for a in answers:
        if a["worker_id"] in worker_color:
            continue
        p=np.array(a["points"]).reshape(-1,2)
        if len(p):
            ax.scatter(p[:,0],p[:,1],s=11,color="#D5DBDE",edgecolors="#3E484E",
                       linewidths=.3,alpha=.75,zorder=4)
    for number,a in enumerate(selected_answers):
        example_points(ax,a["points"],number,on_image=True)
    text(fig,.132,.332,"35 annotators",size=21)
    text(fig,.132,.286,"309 points",size=18)
    # A short key explicitly identifies the three colours as annotators, not classes.
    for x,col,marker,number in zip((.079,.132,.185),COLORS,MARKERS,(1,2,3)):
        fig.add_artist(plt.Line2D([x-.010],[.205],transform=fig.transFigure,marker=marker,
                                 markersize=7,color=col,markeredgewidth=1.4,
                                 markerfacecolor="none" if marker=="o" else col,linestyle="none"))
        text(fig,x+.010,.205,str(number),size=17)
    text(fig,.132,.161,"Highlighted annotators",size=16)

    # Fan out using recorded identities, never by inferring identity from geometry.
    arrow(fig,(.244,.567),(.274,.567),head=False)
    for y in (.693,.463,.233):
        arrow(fig,(.274,.567),(.311,y))
    for row,(answer,col,y) in enumerate(zip(selected_answers,COLORS,(.600,.370,.140)),1):
        ax=art.tile(fig,[.316,y,.107,.190],bounds=(-3,83,83,-3),frame=True)
        p=np.array(answer["points"]).reshape(-1,2)
        example_points(ax,p,row-1)
        text(fig,.465,y+.121,f"Annotator {row}",size=16)
        text(fig,.465,y+.068,f"{len(p)} point"+("s" if len(p)!=1 else ""),size=16)
    text(fig,.399,.088,"32 more answers retained",size=16)

    # A whole-answer / whole-candidate comparison matrix, not spatial clustering.
    ax=fig.add_axes([.562,.484,.175,.311])
    ax.imshow(shown_match,cmap="Greys",vmin=0,vmax=1,aspect="auto",interpolation="nearest")
    ax.set_xticks([0,9,19,33],["1","10","20","34"])
    ax.set_yticks([0,9,19,34],["1","10","20","35"])
    ax.tick_params(labelsize=12,length=3,pad=2)
    ax.set_xlabel("Candidate sets",fontsize=15,labelpad=4)
    ax.set_ylabel("Answers",fontsize=15,labelpad=3)
    for i,col in enumerate(COLORS):
        ax.add_patch(Rectangle((-.095,(34-i)/35),.035,1/35,transform=ax.transAxes,
                              facecolor=col,edgecolor="none",clip_on=False))
        ax.add_patch(Rectangle((i/34,1.02),1/34,.025,transform=ax.transAxes,
                              facecolor=col,edgecolor="none",clip_on=False))
    text(fig,.646,.405,"Exact copies share one candidate",size=14)
    cax=fig.add_axes([.582,.352,.128,.012])
    cb=fig.colorbar(plt.cm.ScalarMappable(Normalize(0,1),"Greys"),cax=cax,
                    orientation="horizontal",ticks=[0,1])
    cb.ax.tick_params(labelsize=11,length=2,pad=2)
    cb.outline.set_visible(False)
    text(fig,.646,.302,"Relative match per answer",size=14)
    arrow(fig,(.646,.277),(.646,.234))
    text(fig,.646,.202,"Fit response patterns",size=18)
    text(fig,.646,.148,"Across image regions",size=16)
    arrow(fig,(.545,.200),(.571,.119),curve=.65)
    arrow(fig,(.739,.119),(.762,.200),curve=.65)
    text(fig,.646,.083,"Update configuration weights",size=15)
    arrow(fig,(.496,.562),(.544,.562))
    arrow(fig,(.754,.562),(.798,.562))

    # Every candidate contributes. Colour ties the three example answers to bars.
    text(fig,.891,.850,"Configuration weights",size=18)
    ax=fig.add_axes([.827,.620,.142,.173])
    bar_colors=list(COLORS)+["#C2C9CD"]*(len(candidates)-3)
    ax.bar(np.arange(1,len(candidates)+1),weights[order],width=.82,color=bar_colors)
    ax.set_xlim(.3,34.7); ax.set_ylim(0,.10)
    ax.set_xticks([1,10,20,34]); ax.set_yticks([0,.05,.10],["0","5%","10%"])
    ax.tick_params(labelsize=12,length=3,pad=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel("All 34 candidates",fontsize=15,labelpad=5)
    text(fig,.891,.478,"Mean point map",size=20)
    ax=image_tile(fig,[.806,.126,.168,.299],raw,image_norm)
    norm=Normalize(0,.05)
    rgba=plt.colormaps["inferno"](norm(smooth))
    rgba[...,3]=np.clip(smooth/.018,0,.94)
    ax.imshow(rgba,extent=(0,80,80,0),interpolation="bilinear",zorder=4)
    ax.contour(np.arange(80)+.5,np.arange(80)+.5,smooth,
               levels=[.014],colors=["#A53336"],linewidths=.75,zorder=5)
    cax=fig.add_axes([.833,.076,.114,.012])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm,"inferno"),cax=cax,
                   orientation="horizontal",ticks=[0,.05])
    cb.ax.set_xticklabels(["0","0.05"])
    cb.ax.tick_params(labelsize=12,length=2,pad=2)
    cb.outline.set_visible(False)
    text(fig,.891,.028,"Point mass per pixel",size=14)

    save(fig,{
        "purpose":"Show an actual densely annotated region being unpacked by recorded annotator identity, compared as whole point sets, and summarized using saved fitted weights.",
        "region":{"image_id":view["image_id"],"view_id":view["view_id"],
                  "origin":[x0,y0],"size":[80,80]},
        "input":{"answers":answers,"number_of_answers":len(answers),
                 "number_of_points":sum(len(a["points"]) for a in answers)},
        "candidate_policy":"All unique observed whole point sets, with point order ignored. Only exact duplicates merge; all 35 original answers remain in the likelihood. There are 34 candidates, including the shared empty candidate. No clustering of nearby points and no invented candidate combinations.",
        "candidate_points": [p.tolist() for p in points],
        "candidate_supports": [c["support_count"] for c in candidates],
        "selected_examples":{"rule":"The three highest-weight candidates in a previously displayed image region; no selection by biological correctness or comparative method advantage.",
                             "indices":selected,"worker_ids":selected_workers,
                             "display_aliases":[1,2,3],"counts":[len(a["points"]) for a in selected_answers],
                             "colours":"Blue, rust and teal identify three example annotators, not inferred response groups.",
                             "markers":"Hollow circle, plus and diamond distinguish overlapping answers without positional jitter. The separated point-set frames have a 3-pixel display margin so boundary marks remain whole."},
        "comparison":{"log_density":log_f.tolist(),"relative_match":relative_match.tolist(),
                      "definition":"For each observed answer, exp(log F) normalized to sum to one across all candidates. This is a display-only relative likelihood under uniform candidate weights, NOT an EM assignment posterior, learned configuration probability, or accuracy.",
                      "row_order":row_order,"column_order":order.tolist(),
                      "colour_range":[0,1],"per_map_maximum_normalization":False,
                      "role":"The full model combines these fixed count/location densities with learned group preferences and response mixture weights. The loop is conceptual; the displayed comparison matrix is fixed and is not updated during parameter fitting."},
        "saved_output":{"weights":weights.tolist(),"group_mix":mix.tolist(),
                        "group_specific_weights":pi[index,:,:len(candidates)].tolist(),
                        "worker_ids":saved["worker_ids"],"q":q.tolist(),"rho":rho.tolist(),
                        "expected_count":float(mass.sum())},
        "display":{"raw_image_limits":[float(lo),float(hi)],"raw_image_gamma":.72,
                   "smoothing":"Gaussian, reflect boundary","sigma_pixels":1.25,
                   "mass_range":[0,.05],"alpha":"clip(smoothed_mass/0.018,0,0.94)",
                   "contour_level":.014,"per_map_maximum_normalization":False},
        "unsmoothed_point_mass":mass.tolist(),"smoothed_point_mass":smooth.tolist(),
        "processing":"Read saved input and converged parameters, evaluate the existing fixed observation density, reconstruct saved output weights, and render. No fitting, parameter changes, or new image inference.",
        "interpretation":"All shown point coordinates are original recorded values; no positional jitter. Identity is recorded, not inferred from an unlabeled point cloud. Image pixels are visual context only. This is an illustration from fitted data, not an accuracy comparison. The map is an average over whole configurations, not a newly generated simultaneous point set.",
    })


if __name__=="__main__":
    main()
