"""PNG-only figures from saved configuration-model results; no fitting."""
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
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from data import DEFAULT_DATA_ROOT, QC_IDS, load_image_views, select_qc_workers, write_json, write_csv

HERE = PROJECT_ROOT
sys.path.insert(0, str(FIGURE_SOURCE_ROOT))
import make_nature_png as export

BLUE = "#256781"
RUST = "#AD6647"
GREY = "#90969A"
MASS_CMAP = LinearSegmentedColormap.from_list("point_mass", ["#FFFFFF", "#9FCCD4", BLUE])
SCORE_CMAP = LinearSegmentedColormap.from_list("legacy_signed_score", [RUST, "#FFFFFF", BLUE])
HASHES = {}
SOURCES = {}


def read_json(path):
    HASHES[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def mass_map(summary, shape=(100, 100)):
    grid = np.zeros(shape)
    for points, probability in zip(summary["candidate_points"], summary["weights"]):
        p = np.asarray(points, float).reshape(-1, 2)
        if not len(p):
            continue
        xy = np.clip(p, [0, 0], [shape[1] - 1, shape[0] - 1])
        x0, y0 = np.floor(xy).astype(int).T
        x1, y1 = np.minimum(x0+1, shape[1]-1), np.minimum(y0+1, shape[0]-1)
        fx, fy = (xy - np.floor(xy)).T
        for x, y, w in ((x0,y0,(1-fx)*(1-fy)), (x1,y0,fx*(1-fy)), (x0,y1,(1-fx)*fy), (x1,y1,fx*fy)):
            np.add.at(grid, (y,x), probability*w)
    assert np.isclose(grid.sum(), summary["expected_count"], atol=1e-10)
    return grid


def canvas():
    return plt.figure(figsize=(16, 9), facecolor="white")


def diagram(fig, rect, reference=None):
    ax = fig.add_axes(rect)
    ax.set(xlim=(12,88), ylim=(88,12), aspect="equal")
    ax.set_axis_off()
    if reference:
        for r in reference["regions"]:
            ax.add_patch(Ellipse(r["center"], 2*r["radii"][0], 2*r["radii"][1],
                                 facecolor="#E5E8E9", edgecolor="#7C858A", linewidth=.9, zorder=0))
    return ax


def points(ax, xy, color=BLUE, marker="o", size=50, alpha=1):
    p = np.asarray(xy).reshape(-1,2)
    if len(p):
        ax.scatter(p[:,0],p[:,1],s=size,c=color,marker=marker,alpha=alpha,
                   linewidths=1.3,edgecolors="white" if marker=="o" else None,zorder=3)


def clean(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=17, width=.8, length=4)


def arrow(fig,start,end,rad=0):
    fig.add_artist(FancyArrowPatch(start,end,transform=fig.transFigure,arrowstyle="-|>",
                                  mutation_scale=15,linewidth=1.1,color="#647078",
                                  connectionstyle=f"arc3,rad={rad}"))


def fitted_values(root,condition):
    target=root/"results"/condition/"configuration"
    status=read_json(target/"status.json")
    path=target/f"parameters_{status['selected_initialization']}.npz"
    HASHES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    data=np.load(path)
    return status,data["worker_ids"].tolist(),data["error_tables"]


def generative(root,out):
    fig=canvas()
    _,workers,tables=fitted_values(root,"merge_majority")
    worker=workers.index("frequent_1")
    reference={"regions":[{"center":[35,50],"radii":[18,24]},
                           {"center":[65,50],"radii":[18,24]}]}
    for x,label in ((.155,"Underlying configuration"),(.475,"Worker count errors"),(.805,"Example responses")):
        fig.text(x,.948,label,ha="center",fontsize=23)
    for row,m in enumerate((2,1)):
        bottom=.55 if row==0 else .14
        ref=reference if m==2 else {"regions":[{"center":[50,50],"radii":[20,24]}]}
        ax=diagram(fig,[.045,bottom,.215,.325],ref)
        points(ax,[[35,50],[65,50]] if m==2 else [[50,50]],size=85)
        fig.text(.153,bottom-.023,f"{m} underlying point"+("s" if m>1 else ""),ha="center",fontsize=19)
        ax=fig.add_axes([.375,bottom+.045,.20,.265])
        probabilities=tables[worker,m]
        ax.bar([0,1,2],probabilities,width=.58,color=BLUE)
        ax.set(ylim=(0,1.02),xticks=[0,1,2],xticklabels=["0","1","2+"],yticks=[0,.5,1])
        ax.set_ylabel("Probability",fontsize=18)
        clean(ax)
        fig.text(.475,bottom-.023,"Reported count",ha="center",fontsize=19)
        arrow(fig,(.27,bottom+.175),(.319,bottom+.175))
        arrow(fig,(.59,bottom+.175),(.65,bottom+.175))
        for n,left in enumerate((.658,.771,.884)):
            ax=diagram(fig,[left,bottom+.065,.103,.205],ref)
            if n==1:
                points(ax,[[49.2,50.8]],color=RUST,size=60,marker="+")
            elif n==2:
                points(ax,[[35.7,49.8],[64.7,50.5]] if m==2 else [[44.4,49.6],[55.8,50.7]],color=RUST,size=60,marker="+")
            fig.text(left+.0515,bottom+.032,f"{n} point"+("s" if n!=1 else ""),ha="center",fontsize=17)
    fig.text(.804,.066,"Position spread: 4 pixels (fixed)",ha="center",fontsize=18)
    fig.text(.48,.064,r"$C_j(m,n)=P(\mathrm{reported\ count\ bin}=n\mid\mathrm{underlying\ count\ bin}=m)$",ha="center",fontsize=17)
    # The two bottom labels occupy different rows to keep the equation readable.
    fig.texts[-1].set_position((.37,.027))
    FigureCanvasPgf(fig)
    SOURCES["generative_model"]={"worker_id":"frequent_1","condition":"merge_majority",
                                  "worker_count_error_table":tables[worker].tolist(),
                                  "point_configurations":"Illustrative draws, not a new simulation or biological reference"}
    write_csv(out/"generative_model_values.csv",[{"worker":"frequent_1","underlying_bin":m,"reported_bin":n,
        "probability":float(tables[worker,m,n])} for m in range(3) for n in range(3)])
    export.save_png(fig,"generative_model")


def inference_schematic(root,out):
    fig=canvas()
    target=root/"results/merge_majority/configuration"
    status,workers,tables=fitted_values(root,"merge_majority")
    raw=read_json(root/"synthetic_inputs/merge_majority/qc_views.json")["views"][2]
    result=read_json(target/f"posteriors_{status['selected_initialization']}.json")[2]
    fig.text(.165,.944,"Observed point sets",ha="center",fontsize=23)
    for j,a in enumerate(raw["answers"]):
        left=.037+(j%5)*.059
        bottom=.703-(j//5)*.162
        ax=diagram(fig,[left,bottom,.054,.126])
        ax.add_patch(Rectangle((12,12),76,76,facecolor="#F6F7F7",edgecolor="#B2BBC0",linewidth=.7,zorder=0))
        points(ax,a["points"],color=RUST if j<7 else BLUE,marker="+",size=34)
    fig.text(.168,.509,"10 workers; one example region",ha="center",fontsize=17)
    fig.text(.487,.944,"Candidate examples",ha="center",fontsize=23)
    best={}
    for k,(s,p) in enumerate(zip(result["candidate_points"],result["weights"])):
        if len(s) not in best or p>result["weights"][best[len(s)]]:
            best[len(s)]=k
    for n,left in enumerate((.376,.468,.560)):
        ax=diagram(fig,[left,.66,.085,.205])
        ax.add_patch(Rectangle((12,12),76,76,facecolor="#F6F7F7",edgecolor="#B2BBC0",linewidth=.8,zorder=0))
        points(ax,result["candidate_points"][best[n]],size=60)
        fig.text(left+.0425,.626,f"{n} point"+("s" if n!=1 else ""),ha="center",fontsize=18)
    fig.text(.493,.582,r"Fixed prior, $a_{rk}$",ha="center",fontsize=17)
    fig.text(.795,.944,"Point-count posterior",ha="center",fontsize=23)
    ax=fig.add_axes([.723,.625,.215,.235])
    p=[result["count_distribution"].get(str(n),0) for n in range(3)]
    ax.bar(range(3),p,color=BLUE,width=.58)
    ax.set(ylim=(0,1.02),xticks=[0,1,2],yticks=[0,.5,1],xlabel="Points")
    ax.xaxis.label.set_fontsize(18)
    clean(ax)
    fig.text(.795,.493,r"$w_{rk}\propto a_{rk}\prod_j p(Y_{rj}\mid S_{rk},C_j)$",ha="center",fontsize=22)
    arrow(fig,(.332,.76),(.365,.76))
    arrow(fig,(.652,.76),(.695,.76))
    fig.text(.26,.426,r"Worker-specific error tables, $C_j$",ha="center",fontsize=21)
    # Show an actual learned under-count rate for each worker, rather than a
    # large numeric matrix. The same C also contains the other response counts.
    order=[workers.index(f"frequent_{i+1}") for i in range(7)]+[workers.index(f"careful_{i+1}") for i in range(3)]
    ax=fig.add_axes([.081,.185,.35,.181])
    ax.bar(range(10),tables[order,2,1],color=[RUST]*7+[BLUE]*3,width=.62)
    ax.set(ylim=(0,1.02),xticks=[0,6,9],xticklabels=["1","7","10"],yticks=[0,1],xlabel="Worker")
    ax.set_ylabel(r"$P(1\mid 2+)$",fontsize=20)
    ax.xaxis.label.set_fontsize(18)
    clean(ax)
    arrow(fig,(.83,.435),(.83,.293))
    arrow(fig,(.83,.286),(.455,.286))
    fig.text(.650,.339,"Expected counts + fixed prior",ha="center",fontsize=18)
    arrow(fig,(.46,.40),(.689,.622))
    fig.text(.51,.53,"Update",ha="center",fontsize=18)
    fig.text(.53,.055,r"$\displaystyle \max_C\left[\sum_r\log\!\left(\sum_k a_{rk}\prod_j p(Y_{rj}\mid S_{rk},C_j)\right)+\log p(C)\right]$",ha="center",fontsize=22)
    FigureCanvasPgf(fig)
    SOURCES["model_inference"]={"training_region_index":2,"selected_initialization":status["selected_initialization"],
        "posterior_source":str(target/f"posteriors_{status['selected_initialization']}.json"),
        "count_probabilities":p,"worker_plot_values":tables[order,2,1].tolist(),
        "objective":"Observed-data log likelihood plus fixed Dirichlet log prior"}
    write_csv(out/"model_inference_values.csv",
        [{"quantity":"point_count_posterior","index":n,"value":value} for n,value in enumerate(p)]+
        [{"quantity":"P(1|2+)","index":workers[j],"value":float(tables[j,2,1])} for j in order])
    export.save_png(fig,"model_inference")


def worker_patterns(root,out):
    fig=canvas()
    rows=[]
    for row,condition in enumerate(("merge_majority","split_majority")):
        status,workers,tables=fitted_values(root,condition)
        bottom=.56 if row==0 else .14
        m,n=(2,1) if row==0 else (1,2)
        reference={"regions":[{"center":p,"radii":[18,24]} for p in ([[35,50],[65,50]] if m==2 else [[50,50]])]}
        ax=diagram(fig,[.025,bottom,.18,.30],reference)
        points(ax,[[35,50],[65,50]] if m==2 else [[50,50]],size=70)
        arrow(fig,(.212,bottom+.15),(.277,bottom+.15))
        ax=diagram(fig,[.285,bottom,.18,.30],reference)
        points(ax,[[50,50]] if n==1 else [[44,50],[56,50]],color=RUST,marker="+",size=85)
        fig.text(.115,bottom-.028,f"{m} underlying point"+("s" if m>1 else ""),ha="center",fontsize=18)
        fig.text(.375,bottom-.028,f"{n} reported point"+("s" if n>1 else ""),ha="center",fontsize=18)
        ax=fig.add_axes([.565,bottom+.01,.365,.295])
        for group,color,ys in (("Workers 1-7",RUST,1), ("Workers 8-10",BLUE,0)):
            prefix="frequent" if ys==1 else "careful"
            indices=[i for i,w in enumerate(workers) if w.startswith(prefix)]
            rates=tables[indices,m,min(n,2)]
            offsets=np.linspace(-.085,.085,len(rates))
            ax.scatter(rates,ys+offsets,s=65,color=color)
            for i,value in zip(indices,rates):
                rows.append({"condition":condition,"worker_id":workers[i],"underlying_bin":m,"reported_bin":min(n,2),"probability":float(value)})
        ax.set(xlim=(0,1),ylim=(-.35,1.35),xticks=[0,.5,1],yticks=[0,1],yticklabels=["8-10","1-7"])
        ax.set_xlabel(r"$P(1\mid 2+)$" if row==0 else r"$P(2+\mid 1)$",fontsize=22)
        ax.set_ylabel("Workers",fontsize=19)
        clean(ax)
    for x,label in ((.115,"Underlying configuration"),(.375,"Count error"),(.75,"Estimated response tendency")):
        fig.text(x,.943,label,ha="center",fontsize=22)
    write_csv(out/"worker_response_pattern_values.csv",rows)
    SOURCES["worker_response_patterns"]={"rows":rows,"groups_used_for_drawing_only":True}
    FigureCanvasPgf(fig)
    export.save_png(fig,"worker_response_patterns")


def comparison(root, out):
    fig = canvas()
    x = [.018,.217,.416,.615,.823]
    labels = ["Synthetic regions", "Crowd annotations", "Legacy Algorithm", "Configuration model", "Count probability"]
    for left,label in zip(x,labels):
        fig.text(left+.082,.947,label,ha="center",fontsize=20)
    rows = []
    maps = {}
    for row,condition in enumerate(("merge_majority","split_majority")):
        inp = root / "synthetic_inputs" / condition
        payload = read_json(inp / "evaluation_views.json")
        reference = read_json(inp / "evaluation_reference_not_used_by_methods.json")["regions_for_evaluation_only"][0]
        result = read_json(root / "results" / condition / "configuration/evaluation_outputs.json")[0]
        path = root / "results" / condition / "legacy/native_heatmaps.npz"
        HASHES[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        legacy = np.load(path)["evaluation"][0]
        raw_mass = mass_map(result)
        smoothed = gaussian_filter(raw_mass,1.5,mode="reflect")
        maps[condition+"_mass_unsmoothed"] = raw_mass
        maps[condition+"_mass_display"] = smoothed
        maps[condition+"_legacy_native"] = legacy
        bottom = .555 if row==0 else .155
        for col in range(4):
            ax = diagram(fig,[x[col],bottom,.176,.325],reference)
            if col==0:
                points(ax,reference["points"],color="black",size=65)
            elif col==1:
                for a in payload["views"][0]["answers"]:
                    points(ax,a["points"],color=RUST if a["worker_id"].startswith("frequent") else BLUE,
                           marker="+",size=65,alpha=.9)
            elif col==2:
                # Keep the signed native values, including the negative tail.
                # The common display scale is clipped below -1; the colorbar
                # explicitly extends there. No per-map maximum normalization.
                rgba = SCORE_CMAP(Normalize(-1,1,clip=True)(legacy))
                rgba[:,:,3] = np.clip(abs(legacy)*.85,0,.9)
                ax.imshow(rgba,extent=(0,100,100,0),interpolation="nearest",zorder=1)
            else:
                rgba = MASS_CMAP(Normalize(0,.08,clip=True)(smoothed))
                rgba[:,:,3] = np.clip(smoothed/.045,0,.93)
                ax.imshow(rgba,extent=(0,100,100,0),interpolation="nearest",zorder=1)
                p=np.array(result["candidate_points"][result["map_candidate_index"]]).reshape(-1,2)
                ax.scatter(p[:,0],p[:,1],s=90,facecolors="none",edgecolors="black",linewidths=1.1,zorder=4)
        count_frequency = {n: sum(len(a["points"])==n for a in payload["views"][0]["answers"]) for n in range(3)}
        fig.text(x[0]+.088,bottom-.025,f"{reference['count']} defined point"+("s" if reference['count']!=1 else ""),ha="center",fontsize=18)
        count_text = ", ".join(f"{n} point"+("s" if n!=1 else "")+f": {count_frequency[n]}" for n in (1,2))
        fig.text(x[1]+.088,bottom-.025,count_text,ha="center",fontsize=16)
        if count_frequency[0]:
            fig.text(x[1]+.088,bottom-.055,f"No points: {count_frequency[0]}",ha="center",fontsize=15)
        fig.text(x[3]+.088,bottom-.025,"Whole MAP configuration",ha="center",fontsize=16)
        ax = fig.add_axes([x[4]+.01,bottom+.025,.135,.27])
        probability = [result["count_distribution"].get(str(n),0.) for n in range(3)]
        ax.bar(range(3),probability,color=BLUE,width=.6)
        ax.set(ylim=(0,1.04),xticks=[0,1,2],yticks=[0,.5,1],xlabel="Points")
        clean(ax)
        ax.xaxis.label.set_fontsize(17)
        rows.append({"condition":condition,"view_index":0,"defined_count":reference["count"],
                     "input_counts":count_frequency,"count_probabilities":probability,
                     "legacy_native_range":[float(legacy.min()),float(legacy.max())],
                     "current_map_candidate":result["candidate_points"][result["map_candidate_index"]]})
    for left,cmap,norm,label,extend,ticks in ((x[2]+.01,SCORE_CMAP,Normalize(-1,1),"Signed score","min",[-1,0,1]),
                                             (x[3]+.01,MASS_CMAP,Normalize(0,.08),"Expected points / pixel","neither",[0,.04,.08])):
        ax = fig.add_axes([left,.069,.155,.011])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ax,orientation="horizontal",extend=extend,ticks=ticks)
        cb.ax.tick_params(labelsize=13,length=3)
        cb.set_label(label,fontsize=14,labelpad=1)
    np.savez_compressed(out/"comparison_map_values.npz",**maps)
    handles=[plt.Line2D([],[],marker="+",color=color,linestyle="none",markersize=9,
                        markeredgewidth=1.5,label=label)
             for color,label in ((RUST,"Workers 1-7"),(BLUE,"Workers 8-10"))]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.034,.018),
               ncol=2,frameon=False,fontsize=15,handletextpad=.35,columnspacing=1.1)
    SOURCES["legacy_configuration_comparison"] = rows
    write_csv(out/"legacy_configuration_values.csv",[{k:json.dumps(v) if isinstance(v,(list,dict)) else v
                                                      for k,v in row.items()} for row in rows])
    export.save_png(fig,"legacy_configuration_comparison")


def calibration(root,out):
    fig=canvas()
    rows=[]
    for column,condition in enumerate(("merge_majority","split_majority")):
        target=root/"results"/condition/"configuration"
        result=read_json(target/"evaluation_outputs.json")
        fixed=read_json(target/"fixed_error_table_ablation.json")
        true_count=2 if column==0 else 1
        ax=fig.add_axes([.095+column*.49,.21,.38,.60])
        for label,values,color,marker in (("Fixed error table",fixed,GREY,"s"),("Estimated error table",result,BLUE,"o")):
            probabilities=np.array([r["count_distribution"].get(str(true_count),0.) for r in values])
            ax.scatter(np.arange(1,11),probabilities,s=70,color=color,marker=marker,label=label,zorder=3)
            for i,p in enumerate(probabilities):
                rows.append({"condition":condition,"view_index":i,"method":label,"defined_count":true_count,"probability":float(p)})
        ax.set(xlim=(.4,10.6),ylim=(-.025,1.045),xticks=[1,5,10],yticks=[0,.5,1],xlabel="Synthetic example")
        if column==0:
            ax.set_ylabel("P(defined point count)",fontsize=21,labelpad=12)
        ax.xaxis.label.set_fontsize(21)
        clean(ax)
        ax.set_title("Under-counting" if column==0 else "Over-counting",fontsize=23,pad=18)
    fig.legend(*fig.axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,.055),ncol=2,frameon=False,fontsize=20)
    write_csv(out/"count_calibration_values.csv",rows)
    SOURCES["count_calibration"]={"all_evaluation_examples_included":True,"rows":rows}
    export.save_png(fig,"count_calibration")


def real_overlay(root,out):
    target=root/"real/training"
    status=read_json(target/"status.json")
    results=read_json(target/f"posteriors_{status['selected_initialization']}.json")
    workers,_=select_qc_workers(DEFAULT_DATA_ROOT)
    # Preserve the previously used image/crop locations; do not select on a
    # favorable current output. The image pixels are display only, not a model input.
    locations=[(QC_IDS[2],(280,280,80,80)),(QC_IDS[0],(0,350,80,80))]
    fig=canvas()
    xs=[.025,.273,.521,.769]
    for x,label in zip(xs,["Raw image","Crowd annotations","Expected point mass","Whole configurations"]):
        fig.text(x+.108,.949,label,ha="center",fontsize=21)
    source=[]
    map_arrays={}
    for row,(image_id,(x,y,w,h)) in enumerate(locations):
        image_path=DEFAULT_DATA_ROOT/image_id/image_id
        HASHES[str(image_path.resolve())]=hashlib.sha256(image_path.read_bytes()).hexdigest()
        raw=np.asarray(Image.open(image_path).convert("L"))
        view=next(v for v in load_image_views(DEFAULT_DATA_ROOT,image_id,workers) if (v.origin_x,v.origin_y,v.width,v.height)==(x,y,w,h))
        local=raw[y:y+h,x:x+w]
        image_results=[r for r in results if r["image_id"]==image_id]
        mass=sum((mass_map(r,raw.shape) for r in image_results),np.zeros(raw.shape))
        smoothed=gaussian_filter(mass,1.5,mode="reflect")[y:y+h,x:x+w]
        map_arrays[f"row_{row}_mass_unsmoothed"]=mass[y:y+h,x:x+w]
        map_arrays[f"row_{row}_mass_display"]=smoothed
        selected=[]
        for r in image_results:
            selected.extend(r["candidate_points"][r["map_candidate_index"]])
        selected=np.asarray(selected).reshape(-1,2)
        selected=selected[(selected[:,0]>=x)&(selected[:,0]<x+w)&(selected[:,1]>=y)&(selected[:,1]<y+h)]-[x,y]
        lo,hi=np.percentile(raw,[1,99.7])
        bottom=.545 if row==0 else .095
        for col in range(4):
            ax=fig.add_axes([xs[col],bottom,.215,.38])
            ax.imshow(local,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest",extent=(0,w,h,0))
            if col==1:
                p=np.concatenate([a.points for a in view.answers])
                ax.scatter(p[:,0],p[:,1],s=12,c="#F0A35F",alpha=.58,linewidths=0)
            elif col==2:
                rgba=plt.get_cmap("inferno")(Normalize(0,.08,clip=True)(smoothed))
                rgba[:,:,3]=np.clip(smoothed/.04,0,.85)
                ax.imshow(rgba,extent=(0,w,h,0),interpolation="nearest")
            elif col==3 and len(selected):
                ax.scatter(selected[:,0],selected[:,1],s=55,facecolors="none",edgecolors="#FFC76E",linewidths=1.6)
            ax.set(xlim=(0,w),ylim=(h,0),xticks=[],yticks=[])
            for spine in ax.spines.values():
                spine.set_linewidth(.55)
            ax.plot([5,25],[h-5,h-5],color="white",linewidth=2)
        source.append({"image_id":image_id,"crop":[x,y,w,h],"display_limits":[float(lo),float(hi)],
                       "scale_bar_pixels":20,"workers":len(workers),"selected_configuration_points_local":selected.tolist(),
                       "expected_count_in_crop":float(mass[y:y+h,x:x+w].sum())})
    # Label the pixel scale outside the busy image; no invented physical scale.
    fig.text(.13,.037,"20 pixels",ha="center",fontsize=16)
    cbax=fig.add_axes([.546,.067,.165,.012])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,.08),cmap="inferno"),cax=cbax,orientation="horizontal",ticks=[0,.04,.08])
    cb.ax.tick_params(labelsize=13,length=3)
    cb.set_label("Expected points / pixel",fontsize=14,labelpad=0)
    np.savez_compressed(out/"real_overlay_values.npz",**map_arrays)
    SOURCES["real_annotation_outputs"] = source
    write_csv(out/"real_annotation_output_values.csv",[{k:json.dumps(v) if isinstance(v,list) else v
                                                        for k,v in row.items()} for row in source])
    export.save_png(fig,"real_annotation_outputs")


def annotation_configurations(root,out):
    """Show the actual region extraction and whole candidates, without refitting."""
    records=read_json(root/"real/training/regions.json")
    x,y,w,h=280,280,80,80
    selected=[r for r in records if r["image_id"]==QC_IDS[2]
              and x<=r["bounds"][0] and y<=r["bounds"][1]
              and r["bounds"][2]<=x+w and r["bounds"][3]<=y+h]
    # Geometry-only selection, not posterior confidence or visual success.
    selected.sort(key=lambda r:(r["bounds"][1],r["bounds"][0]))
    focus=[r for r in selected if {0,1,2}.issubset({len(s) for s in r["candidates"]})][:2]
    assert len(focus)==2
    observed=np.concatenate([np.asarray(a["points"]).reshape(-1,2)
                             for r in selected for a in r["observations"]])
    fig=canvas()
    for cx,label in ((.155,"Crowd annotations"),(.476,"Local regions"),(.821,"Candidate examples")):
        fig.text(cx,.946,label,ha="center",fontsize=23)
    for col,left in enumerate((.022,.347)):
        ax=fig.add_axes([left,.195,.273,.63])
        ax.set(xlim=(x,x+w),ylim=(y+h,y),aspect="equal",xticks=[],yticks=[])
        for spine in ax.spines.values():
            spine.set_color("#B2BBC0");spine.set_linewidth(.7)
        points(ax,observed,color=RUST,marker="+",size=32,alpha=.65)
        if col:
            for r in selected:
                b=r["bounds"]
                focal=r["region_id"] in [f["region_id"] for f in focus]
                ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,
                                       edgecolor=BLUE if focal else "#AFB7BB",linewidth=1.5 if focal else .7))
                if focal:
                    n=next(i for i,f in enumerate(focus) if f["region_id"]==r["region_id"])
                    ax.text(b[0]+.5,b[1]+1,str(n+1),ha="left",va="top",fontsize=15,
                            bbox={"facecolor":"white","edgecolor":"none","pad":1})
    arrow(fig,(.300,.51),(.337,.51))
    arrow(fig,(.625,.51),(.659,.51))
    sources=[]
    for row,r in enumerate(focus):
        bottom=.563 if row==0 else .255
        b=r["bounds"]
        cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        side=max(b[2]-b[0],b[3]-b[1])+4
        examples=[]
        for n,left in enumerate((.674,.776,.878)):
            k=next(i for i,s in enumerate(r["candidates"]) if len(s)==n)
            ax=fig.add_axes([left,bottom,.094,.21])
            ax.set(xlim=(cx-side/2,cx+side/2),ylim=(cy+side/2,cy-side/2),aspect="equal",xticks=[],yticks=[],facecolor="#F6F7F7")
            for spine in ax.spines.values():
                spine.set_color("#B2BBC0");spine.set_linewidth(.7)
            points(ax,r["candidates"][k],size=65)
            if row==0:
                fig.text(left+.047,.815,f"{n} point"+("s" if n!=1 else ""),ha="center",fontsize=17)
            examples.append({"index":k,"points":r["candidates"][k],"source":r["candidate_sources"][k]})
        fig.text(.82,bottom-.022,f"Region {row+1}",ha="center",fontsize=18)
        sources.append({"region_id":r["region_id"],"bounds":b,"shown_candidates":examples})
    fig.text(.155,.112,"35 workers",ha="center",fontsize=19)
    fig.text(.476,.112,"One response per worker",ha="center",fontsize=19)
    fig.text(.821,.112,"Observed + proposed point sets",ha="center",fontsize=18)
    SOURCES["annotation_configurations"]={"image_id":QC_IDS[2],"crop":[x,y,w,h],
        "region_ids":[r["region_id"] for r in selected],"focal_regions":sources,
        "selection":"First two fully enclosed regions with 0/1/2-point candidates, ordered by upper and left bounds; no posterior used",
        "observations":"Original selected responses from the fitted region file; outside/crossing regions omitted from this schematic"}
    write_csv(out/"annotation_configuration_values.csv",[{"region_id":r["region_id"],"candidate_index":c["index"],
        "candidate_points":json.dumps(c["points"]),"point_count":len(c["points"]),"candidate_source":c["source"]}
        for r in sources for c in r["shown_candidates"]])
    export.save_png(fig,"annotation_configurations")


def inference_to_outputs(root,out):
    # Retain an imperfect case. It has genuine uncertainty between one and two
    # points and is not replaced by an artificially favorable example.
    result=read_json(root/"results/split_majority/configuration/evaluation_outputs.json")[3]
    fig=canvas()
    order=np.argsort(result["weights"])[::-1]
    fig.text(.173,.947,"Candidate configurations",ha="center",fontsize=22)
    fig.text(.485,.947,"Whole MAP configuration",ha="center",fontsize=22)
    fig.text(.817,.947,"Expected point mass",ha="center",fontsize=22)
    for rank,k in enumerate(order[:3]):
        bottom=.679-rank*.235
        ax=diagram(fig,[.037,bottom,.13,.215])
        ax.add_patch(Rectangle((12,12),76,76,facecolor="#F6F7F7",edgecolor="#B2BBC0",linewidth=.7,zorder=0))
        points(ax,result["candidate_points"][k],size=75)
        fig.text(.219,bottom+.10,f"{result['weights'][k]:.3f}",ha="center",fontsize=23)
    fig.text(.165,.11,f"Other configurations: {sum(result['weights'][k] for k in order[3:]):.3f}",ha="center",fontsize=17)
    arrow(fig,(.282,.773),(.379,.773))
    arrow(fig,(.288,.483),(.37,.375))
    arrow(fig,(.287,.513),(.663,.513))
    fig.text(.476,.548,"Weighted mean",ha="center",fontsize=19)
    ax=diagram(fig,[.399,.657,.16,.24])
    ax.add_patch(Rectangle((12,12),76,76,facecolor="#F6F7F7",edgecolor="#B2BBC0",linewidth=.7,zorder=0))
    points(ax,result["candidate_points"][result["map_candidate_index"]],size=100)
    ax=fig.add_axes([.412,.188,.187,.225])
    probability=[result["count_distribution"].get(str(n),0.) for n in range(3)]
    ax.bar(range(3),probability,color=BLUE,width=.58)
    ax.set(ylim=(0,1.02),xticks=[0,1,2],yticks=[0,.5,1],xlabel="Points")
    ax.set_ylabel("Probability",fontsize=18)
    ax.xaxis.label.set_fontsize(18)
    clean(ax)
    mass=mass_map(result)
    shown=gaussian_filter(mass,1.5,mode="reflect")
    ax=diagram(fig,[.682,.267,.28,.49])
    ax.imshow(shown,cmap=MASS_CMAP,vmin=0,vmax=.08,extent=(0,100,100,0),interpolation="nearest")
    ax.add_patch(Rectangle((12,12),76,76,fill=False,edgecolor="#B2BBC0",linewidth=.7))
    cbax=fig.add_axes([.724,.177,.20,.013])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,.08),cmap=MASS_CMAP),cax=cbax,orientation="horizontal",ticks=[0,.04,.08])
    cb.ax.tick_params(labelsize=15,length=3)
    cb.set_label("Expected points / pixel",fontsize=17,labelpad=4)
    fig.text(.503,.440,"Point-count distribution",ha="center",fontsize=18)
    SOURCES["inference_to_outputs"]={"condition":"split_majority","evaluation_index":3,
        "top_candidate_indices":order[:3].tolist(),"top_weights":[result["weights"][k] for k in order[:3]],
        "count_distribution":result["count_distribution"],"expected_count":result["expected_count"],
        "defined_count_for_evaluation_only":1,"map_count":len(result["candidate_points"][result["map_candidate_index"]]),
        "note":"This is the retained failed synthetic count case, not a claim of successful reconstruction"}
    np.savez_compressed(out/"inference_output_map_values.npz",mass_unsmoothed=mass,mass_display=shown)
    write_csv(out/"inference_output_values.csv",[{"candidate_index":k,"candidate_points":json.dumps(s),
        "point_count":len(s),"posterior_probability":p,"is_map":k==result["map_candidate_index"]}
        for k,(s,p) in enumerate(zip(result["candidate_points"],result["weights"]))])
    export.save_png(fig,"inference_to_outputs")


def publish_figures(out):
    """Refresh the latest existing PNG paths, retaining their previous versions."""
    base=HERE/"outputs/method_schematics"
    backup=base/"previous_class_model"
    backup.mkdir(exist_ok=True)
    names=("annotation_configurations","generative_model","model_inference",
           "inference_to_outputs","worker_response_patterns","real_annotation_outputs")
    mappings={"nature_png":{n:n for n in names},
              "estimation_png":{"generative_model":"generative_model","estimation_and_outputs":"model_inference"},
              "concept_png":{"annotation_flow":"annotation_configurations","model_concept":"inference_to_outputs"}}
    manifest=json.loads((out/"manifest.json").read_text())
    published=[]
    for directory,mapping in mappings.items():
        active=base/directory
        archive=backup/directory
        if not archive.exists():
            active.rename(archive)
        active.mkdir(exist_ok=True)
        for target,source in mapping.items():
            path=active/f"{target}.png"
            shutil.copy2(out/f"{source}.png",path)
            assert export.digest(path)==export.digest(out/f"{source}.png")
            published.append({"path":str(path),"source":str(out/f"{source}.png"),"sha256":export.digest(path)})
        write_json(active/"manifest.json",{"model":"latent configuration with worker count errors",
            "files":{f"{k}.png":f"../configuration_png/{v}.png" for k,v in mapping.items()},
            "complete_manifest":"../configuration_png/manifest.md","previous_version":str(archive),
            "checks":[c for c in manifest["checks"] if c["figure"] in mapping.values()]})
    write_json(out/"published_paths.json",{"files":published,"previous_version_directory":str(backup)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,default=HERE/"outputs/configuration_consensus_v2")
    p.add_argument("--output",type=Path,default=HERE/"outputs/method_schematics/configuration_png")
    p.add_argument("--publish",action="store_true",help="Update existing PNG paths, retaining a prior-version backup")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    export.OUT=a.output
    export.configure()
    comparison(a.results,a.output)
    calibration(a.results,a.output)
    generative(a.results,a.output)
    inference_schematic(a.results,a.output)
    worker_patterns(a.results,a.output)
    real_overlay(a.results,a.output)
    annotation_configurations(a.results,a.output)
    inference_to_outputs(a.results,a.output)
    write_json(a.output/"source_values.json",SOURCES)
    for path,digest in HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    write_json(a.output/"manifest.json",{"source_sha256":HASHES,"checks":export.CHECKS,
        "size_pixels":[4800,2700],"smoothing":{"sigma_pixels":1.5,"method":"normalized Gaussian, reflect boundary"},
        "legacy_map_display":{"range":[-1,1],"clip_negative_tail":True,"colorbar_extend":"min","native_values_retained":True},
        "current_map_display":{"range":[0,.08],"per_map_maximum_normalization":False},
        "interpretation":"Development synthetic checks, not a blinded test or biological accuracy claim."})
    if a.publish:
        publish_figures(a.output)


if __name__=="__main__":
    main()
