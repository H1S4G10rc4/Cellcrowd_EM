"""Five compact PNG figures from existing results. No model fitting."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter

import make_configuration_figures as base

ROOT = base.HERE / "outputs/configuration_consensus_v2"
OUT = base.HERE / "outputs/method_schematics/compact_png"
BLUE, RUST, GREY = base.BLUE, base.RUST, base.GREY
VALUES = {}
LAYOUT = []


def label(fig, x, y, text, size=21, **kw):
    return fig.text(x, y, text, ha="center", va="center", fontsize=size, **kw)


def tile(fig, rect, bounds=(12, 88, 20, 80), reference=None, frame=False):
    ax = fig.add_axes(rect)
    x0, x1, y0, y1 = bounds
    ax.set(xlim=(x0, x1), ylim=(y1, y0), aspect="equal", xticks=[], yticks=[])
    ax.apply_aspect()
    if frame:
        ax.set_facecolor("#F6F7F7")
        for s in ax.spines.values():
            s.set_color("#AEB7BC")
            s.set_linewidth(.7)
    else:
        ax.set_axis_off()
    if reference:
        for r in reference["regions"]:
            ax.add_patch(Ellipse(r["center"], 2*r["radii"][0], 2*r["radii"][1],
                                 facecolor="#E5E8E9", edgecolor="#7C858A", lw=.9))
    return ax


def nearby(ax, text, where="top", size=20, gap=.012):
    """Place labels against the actual equal-aspect axes, not the slot."""
    ax.apply_aspect()
    b = ax.get_position()
    y = b.y1 + gap if where == "top" else b.y0 - gap
    va = "bottom" if where == "top" else "top"
    t = ax.figure.text((b.x0+b.x1)/2, y, text, ha="center", va=va, fontsize=size)
    LAYOUT.append({"text": text, "gap_fraction_of_height": gap})
    return t


def bars(ax, probabilities, xlabel="Points"):
    ax.bar(range(3), probabilities, color=BLUE, width=.58)
    ax.set(ylim=(0, 1.04), xticks=[0, 1, 2], yticks=[0, .5, 1], xlabel=xlabel)
    ax.xaxis.label.set_fontsize(18)
    base.clean(ax)


def scale(fig, rect, cmap, norm, title, ticks, extend="neither"):
    ax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax,
                     orientation="horizontal", ticks=ticks, extend=extend)
    cb.ax.tick_params(labelsize=14, length=3, pad=2)
    cb.set_label(title, fontsize=15, labelpad=1)


def save(fig, stem):
    base.FigureCanvasPgf(fig)
    base.export.save_png(fig, stem)


def input_regions(root):
    records = base.read_json(root / "real/training/regions.json")
    x, y, w, h = 280, 280, 80, 80
    selected = [r for r in records if r["image_id"] == base.QC_IDS[2]
                and x <= r["bounds"][0] and y <= r["bounds"][1]
                and r["bounds"][2] <= x+w and r["bounds"][3] <= y+h]
    selected.sort(key=lambda r: (r["bounds"][1], r["bounds"][0]))
    focus = [r for r in selected if {0,1,2}.issubset({len(s) for s in r["candidates"]})][:2]
    fig = base.canvas()
    ax = tile(fig, [.025, .08, .45, .8], (x,x+w,y,y+h), frame=True)
    ax.set_facecolor("white")
    observed = np.concatenate([np.asarray(a["points"]).reshape(-1,2)
                               for r in selected for a in r["observations"]])
    base.points(ax, observed, color=RUST, marker="+", size=36, alpha=.75)
    for r in selected:
        b = r["bounds"]
        focal = r["region_id"] in [f["region_id"] for f in focus]
        ax.add_patch(Rectangle(b[:2], b[2]-b[0], b[3]-b[1], fill=False,
                               edgecolor=BLUE if focal else "#AFB7BB", lw=1.6 if focal else .8))
        if focal:
            number = next(i+1 for i,f in enumerate(focus) if f["region_id"] == r["region_id"])
            ax.text(b[0]+.6, b[1]+1, str(number), ha="left", va="top", fontsize=19,
                    bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
    nearby(ax, "Crowd annotations and local regions", size=23, gap=.028)
    nearby(ax, "35 workers; one response per worker per region", "bottom", size=18, gap=.016)
    label(fig, .753, .947, "Candidate point configurations", 23)
    rows = []
    for row, r in enumerate(focus):
        bottom = .595 if row == 0 else .175
        b = r["bounds"]
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        side = max(b[2]-b[0], b[3]-b[1])+4
        label(fig, .753, bottom+.279, f"Region {row+1}", 20)
        base.arrow(fig, (.486,bottom+.12), (.522,bottom+.12))
        for n, left in enumerate((.531,.682,.833)):
            k = next(i for i,s in enumerate(r["candidates"]) if len(s) == n)
            ax = tile(fig, [left,bottom,.139,.2472],
                      (cx-side/2,cx+side/2,cy-side/2,cy+side/2), frame=True)
            base.points(ax, r["candidates"][k], size=95)
            nearby(ax, f"{n} point"+("s" if n != 1 else ""), "bottom", 19)
            rows.append({"region_id":r["region_id"], "candidate_index":k,
                         "points":r["candidates"][k], "source":r["candidate_sources"][k]})
    label(fig, .753, .072, "Observed + proposed point configurations", 18)
    VALUES["01_point_configurations"] = rows
    save(fig, "01_point_configurations")


def generative(root):
    fig = base.canvas()
    for x, y, text in ((.137,.922,"Point configuration"),(.478,.925,"Worker count probabilities"),(.81,.841,"Example responses")):
        label(fig, x, y, text, 22)
    rows = []
    for row, (condition, m) in enumerate((("merge_majority",2),("split_majority",1))):
        _, workers, tables = base.fitted_values(root, condition)
        bottom = .566 if row == 0 else .142
        centers = [[35,50],[65,50]] if m == 2 else [[50,50]]
        reference = {"regions":[{"center":p,"radii":[18 if m==2 else 20,24]} for p in centers]}
        ax = tile(fig, [.025,bottom,.223,.325], reference=reference)
        base.points(ax, centers, size=105)
        nearby(ax, f"{m} underlying point"+("s" if m != 1 else ""), "bottom", 19, .007)
        ax = fig.add_axes([.351,bottom+.002,.238,.318])
        xx = np.arange(3)
        for offset, name, color, display in ((-.18,"frequent_1",RUST,"Worker 1"),(.18,"careful_1",BLUE,"Worker 8")):
            probs = tables[workers.index(name),m]
            ax.bar(xx+offset, probs, width=.33, color=color, label=display)
            rows.append({"condition":condition,"worker":name,"underlying_bin":m,"probabilities":probs.tolist()})
        ax.set(ylim=(0,1.04),xticks=xx,xticklabels=["0","1","2+"],yticks=[0,.5,1],xlabel="Reported count")
        ax.set_ylabel("Probability", fontsize=18)
        ax.xaxis.label.set_fontsize(18)
        base.clean(ax)
        base.arrow(fig, (.256,bottom+.17), (.296,bottom+.17))
        base.arrow(fig, (.601,bottom+.17), (.635,bottom+.17))
        for n,left in enumerate((.642,.756,.870)):
            ax = tile(fig, [left,bottom+.074,.110,.171], reference=reference)
            if n == 1:
                base.points(ax, [[49.2,50.8]], color=RUST, marker="+", size=95)
            elif n == 2:
                base.points(ax, [[35.7,49.8],[64.7,50.5]] if m==2 else [[44.4,49.6],[55.8,50.7]],
                            color=RUST, marker="+", size=95)
            nearby(ax, f"{n} point"+("s" if n != 1 else ""), "bottom", 18, .005)
        if row == 0:
            handles, legends = fig.axes[1].get_legend_handles_labels()
            fig.legend(handles, legends, loc="center",bbox_to_anchor=(.805,.506),
                       ncol=2,frameon=False,fontsize=17,handlelength=1.2,columnspacing=1.2)
    label(fig,.803,.101,r"Position spread: $\sigma=4$ pixels (fixed)",17)
    label(fig,.47,.026,r"$C_j(m,n)=P(\mathrm{reported\ count\ bin}=n\mid\mathrm{underlying\ count\ bin}=m)$",20)
    VALUES["02_generative_model"] = rows
    save(fig, "02_generative_model")


def inference(root):
    fig = base.canvas()
    status, workers, tables = base.fitted_values(root,"merge_majority")
    raw = base.read_json(root/"synthetic_inputs/merge_majority/qc_views.json")["views"][2]
    result = base.read_json(root/f"results/merge_majority/configuration/posteriors_{status['selected_initialization']}.json")[2]
    label(fig,.170,.950,"Observed responses",23)
    for j,a in enumerate(raw["answers"]):
        left = .025+(j%5)*.060
        bottom = .777-(j//5)*.157
        ax = tile(fig,[left,bottom,.052,.111],(12,88,12,88),frame=True)
        base.points(ax,a["points"],color=RUST if j<7 else BLUE,marker="+",size=47)
    label(fig,.17,.590,"10 workers; one region",17)
    label(fig,.515,.950,"Point configuration probabilities",23)
    order = np.argsort(result["weights"])[::-1]
    for rank,k in enumerate(order[:3]):
        ax = tile(fig,[.354+rank*.115,.705,.106,.1885],(12,88,12,88),frame=True)
        base.points(ax,result["candidate_points"][k],size=80)
        nearby(ax, f"{result['weights'][k]:.3f}","bottom",20,.008)
    label(fig,.521,.622,"Other point configurations: <0.001",16)
    base.arrow(fig,(.326,.794),(.345,.794))
    label(fig,.850,.950,"Selected point configuration",21)
    ax=tile(fig,[.764,.705,.182,.1885],(12,88,12,88),frame=True)
    base.points(ax,result["candidate_points"][result["map_candidate_index"]],size=105)
    nearby(ax,"Highest probability","bottom",18,.008)
    base.arrow(fig,(.704,.794),(.790,.794))
    label(fig,.50,.535,r"$w_{rk}\propto a_{rk}\prod_j p(Y_{rj}\mid S_{rk},C_j)$",23)
    label(fig,.17,.477,r"Worker count probabilities, $C_j$",20)
    order_workers=[workers.index(f"frequent_{i+1}") for i in range(7)]+[workers.index(f"careful_{i+1}") for i in range(3)]
    ax=fig.add_axes([.064,.185,.232,.228])
    ax.bar(range(1,11),tables[order_workers,2,1],color=[RUST]*7+[BLUE]*3,width=.65)
    ax.set(ylim=(0,1.04),xticks=[1,7,10],yticks=[0,.5,1],xlabel="Worker")
    ax.set_ylabel(r"$P(1\mid 2+)$",fontsize=19)
    ax.xaxis.label.set_fontsize(18)
    base.clean(ax)
    base.arrow(fig,(.306,.418),(.340,.594))
    base.arrow(fig,(.362,.594),(.333,.418))
    label(fig,.343,.385,"Update",16)
    label(fig,.505,.447,"Point-count distribution",20)
    ax=fig.add_axes([.412,.185,.184,.222])
    probability=[result["count_distribution"].get(str(n),0) for n in range(3)]
    bars(ax,probability)
    base.arrow(fig,(.505,.507),(.505,.475))
    label(fig,.842,.507,"Expected point mass",21)
    mass=base.mass_map(result)
    shown=gaussian_filter(mass,1.5,mode="reflect")
    ax=tile(fig,[.735,.195,.23,.279],(12,88,12,88),frame=True)
    ax.imshow(shown,cmap=base.MASS_CMAP,vmin=0,vmax=.08,extent=(0,100,100,0),interpolation="nearest")
    base.arrow(fig,(.666,.534),(.711,.382))
    label(fig,.707,.548,"Mean",16)
    scale(fig,[.754,.167,.19,.011],base.MASS_CMAP,Normalize(0,.08),"Expected points / pixel",[0,.04,.08])
    label(fig,.505,.063,r"$\displaystyle \max_C\left[\sum_r\log\!\left(\sum_k a_{rk}\prod_j p(Y_{rj}\mid S_{rk},C_j)\right)+\log p(C)\right]$",21)
    VALUES["03_inference_and_outputs"]={"condition":"merge_majority","region_index":2,
        "observations":raw["answers"],"posterior":result,"worker_order":order_workers,
        "worker_under_count_probabilities":tables[order_workers,2,1].tolist(),
        "candidate_prior":"Fixed, count-balanced", "error_table_prior":"Fixed Dirichlet"}
    np.savez_compressed(OUT/"inference_map_values.npz",mass_unsmoothed=mass,mass_display=shown)
    save(fig,"03_inference_and_outputs")


def comparison(root):
    fig=base.canvas()
    xs=[.022,.265,.510,.781]
    for x,t in ((.125,"Synthetic input"),(.367,"Legacy Algorithm"),(.613,"Current model"),(.873,r"$P(\mathrm{defined\ count})$")):
        label(fig,x,.955,t,23)
    rows=[]
    maps={}
    for row,condition in enumerate(("merge_majority","split_majority")):
        bottom=.571 if row==0 else .181
        inp=root/"synthetic_inputs"/condition
        raw=base.read_json(inp/"evaluation_views.json")["views"][0]
        ref=base.read_json(inp/"evaluation_reference_not_used_by_methods.json")["regions_for_evaluation_only"][0]
        results=base.read_json(root/"results"/condition/"configuration/evaluation_outputs.json")
        fixed=base.read_json(root/"results"/condition/"configuration/fixed_error_table_ablation.json")
        result=results[0]
        path=root/"results"/condition/"legacy/native_heatmaps.npz"
        base.HASHES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        legacy=np.load(path)["evaluation"][0]
        mass=base.mass_map(result)
        shown=gaussian_filter(mass,1.5,mode="reflect")
        maps[condition+"_legacy_native"]=legacy
        maps[condition+"_mass_unsmoothed"]=mass
        maps[condition+"_mass_display"]=shown
        for col in range(3):
            ax=tile(fig,[xs[col],bottom,.213,.321],reference=ref)
            if col==0:
                rp=np.asarray(ref["points"])
                ax.scatter(rp[:,0],rp[:,1],s=165,facecolors="none",edgecolors="black",lw=1.25,zorder=4)
                for a in raw["answers"]:
                    base.points(ax,a["points"],color=RUST if a["worker_id"].startswith("frequent") else BLUE,
                                marker="+",size=72,alpha=.90)
            elif col==1:
                rgba=base.SCORE_CMAP(Normalize(-1,1,clip=True)(legacy))
                rgba[:,:,3]=np.clip(abs(legacy)*.85,0,.9)
                ax.imshow(rgba,extent=(0,100,100,0),interpolation="nearest",zorder=1)
            else:
                rgba=base.MASS_CMAP(Normalize(0,.08,clip=True)(shown))
                rgba[:,:,3]=np.clip(shown/.045,0,.93)
                ax.imshow(rgba,extent=(0,100,100,0),interpolation="nearest",zorder=1)
                p=np.asarray(result["candidate_points"][result["map_candidate_index"]]).reshape(-1,2)
                ax.scatter(p[:,0],p[:,1],s=105,facecolors="none",edgecolors="black",lw=1.2,zorder=4)
                nearby(ax,"Selected point configuration","bottom",16,.005)
        counts={n:sum(len(a["points"])==n for a in raw["answers"]) for n in range(3)}
        label(fig,.128,bottom-.018,f"0 / 1 / 2 points: {counts[0]} / {counts[1]} / {counts[2]} workers",16)
        if row==0:
            label(fig,.128,.907,"2 defined points",17)
        else:
            label(fig,.128,.484,"1 defined point",17)
        ax=fig.add_axes([xs[3],bottom+.002,.192,.276])
        truth=ref["count"]
        series={}
        for name,vals,color,marker in (("Fixed table",fixed,GREY,"s"),("Estimated table",results,BLUE,"o")):
            ps=[r["count_distribution"].get(str(truth),0.) for r in vals]
            ax.scatter(range(1,11),ps,s=65 if marker=="s" else 33,
                       facecolors="none" if marker=="s" else color,edgecolors=color,
                       linewidths=1.1,marker=marker,label=name,zorder=3 if marker=="s" else 4)
            series[name]=ps
        ax.set(xlim=(.4,10.6),ylim=(-.035,1.04),xticks=[1,5,10],yticks=[0,.5,1],xlabel="Synthetic example")
        ax.xaxis.label.set_fontsize(16)
        base.clean(ax)
        nearby(ax,"Under-counting" if row==0 else "Over-counting",size=19,gap=.008)
        rows.append({"condition":condition,"illustrated_evaluation_index":0,"input_counts":counts,
                     "reference_count":truth,"posterior":result,"calibration_probabilities":series,
                     "legacy_range":[float(legacy.min()),float(legacy.max())]})
    scale(fig,[.285,.094,.173,.012],base.SCORE_CMAP,Normalize(-1,1),"Signed score",[-1,0,1],"min")
    scale(fig,[.536,.094,.173,.012],base.MASS_CMAP,Normalize(0,.08),"Expected points / pixel",[0,.04,.08])
    handles=[plt.Line2D([],[],marker=m,color=c,linestyle="none",markersize=8,markerfacecolor="none" if m=="o" else c,
                       label=t) for m,c,t in (("o","black","Defined points"),("+",RUST,"Workers 1-7"),("+",BLUE,"Workers 8-10"))]
    fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.018,.067),frameon=False,fontsize=15,handlelength=1.1,labelspacing=.25)
    legend_handles=[plt.Line2D([],[],marker=m,color=c,markerfacecolor="none" if m=="s" else c,
                              linestyle="none",markersize=7,label=t)
                    for m,c,t in (("o",BLUE,"Estimated table"),("s",GREY,"Fixed table"))]
    fig.legend(handles=legend_handles,loc="center",bbox_to_anchor=(.873,.063),frameon=False,fontsize=15,labelspacing=.25)
    VALUES["04_legacy_comparison"]=rows
    np.savez_compressed(OUT/"comparison_map_values.npz",**maps)
    save(fig,"04_legacy_comparison")


def real_output(root):
    # Reuse the audited native plotting objects, then change their layout only.
    captured=[]
    original=base.export.save_png
    base.export.save_png=lambda fig,stem: captured.append(fig)
    try:
        base.real_overlay(root,OUT)
    finally:
        base.export.save_png=original
    fig=captured[0]
    for t in list(fig.texts):
        t.remove()
    xs=[.015,.262,.509,.756]
    for i,ax in enumerate(fig.axes[:8]):
        row,col=divmod(i,4)
        ax.set_position([xs[col],.514 if row==0 else .082,.23,.408889])
        if row==0:
            nearby(ax,["Raw image","Crowd annotations","Expected point mass","Point configurations"][col],size=21,gap=.011)
    # Compact, adjacent legend and scale labels.
    fig.axes[8].set_position([.541,.061,.17,.010])
    label(fig,.13,.054,"20 pixels",16)
    VALUES["05_real_annotation_outputs"]=base.SOURCES["real_annotation_outputs"]
    save(fig,"05_real_annotation_outputs")


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--only",choices=["input","generative","inference","comparison","real"])
    args=parser.parse_args()
    OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    base.export.OUT=OUT
    base.export.CHECKS.clear()
    base.export.configure()
    functions={"input":input_regions,"generative":generative,"inference":inference,"comparison":comparison,"real":real_output}
    for name,function in functions.items():
        if args.only is None or args.only==name:
            function(args.results)
    for path,digest in base.HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    suffix="" if args.only is None else "_"+args.only
    base.write_json(OUT/f"source_values{suffix}.json",VALUES)
    base.write_json(OUT/f"layout_checks{suffix}.json",{"checks":base.export.CHECKS,"source_sha256":base.HASHES,
        "source_files_unchanged":True,"label_gaps":LAYOUT,"size_pixels":[4800,2700],
        "models_rerun":False,"old_figures_modified":False})


if __name__=="__main__":
    main()
