"""Draw the conditional generative model, not biological truth or a new fit."""
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
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np

import make_method_schematics as art

OUT = METHOD_OUTPUT_ROOT
ROOT = OUT.parents[2]
MODEL_DIR = ROOT / "point_annotation_trial"
import model


def human(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((.5, .73), .15, facecolor="#DFE6E8",
                        edgecolor=art.GRAY, linewidth=1.3))
    verts = [(.19,.13), (.19,.49), (.36,.55), (.50,.48),
             (.64,.55), (.81,.49), (.81,.13), (.19,.13)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes), facecolor="#D5E4EC",
                           edgecolor=art.BLUE, linewidth=1.3))


def main():
    sources = [source_path("model.py"), source_path("data.py"),
               ROOT / "comparison_report/connected_split_case/definition.json"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    definition = json.loads(sources[2].read_text())
    candidates = [definition["click_templates"]["separated"],
                   definition["click_templates"]["merged"]]
    # A transparent, hand-specified possible response, not a new measurement or fit.
    response = [[33.,53.], [68.,48.]]
    chosen = np.asarray(candidates[0])/100
    m = len(chosen)
    bins = np.arange(5)
    count_prob = np.exp(model.log_count_probability(m, bins))
    count_prob = np.r_[count_prob, 1-count_prob.sum()]
    assert np.isclose(count_prob.sum(),1)
    grid = np.linspace(0,1,301)
    xx, yy = np.meshgrid(grid,grid)
    density = model.spatial_density(np.column_stack([xx.ravel(),yy.ravel()]),chosen).reshape(xx.shape)
    f_value = float(np.exp(model.log_f_for_points(np.array(response)/100,(chosen,))[0]))

    plt.rcParams.update({"mathtext.fontset":"dejavusans", "mathtext.default":"regular"})
    fig = plt.figure(figsize=(16,9),dpi=120)
    for x, txt in ((.091,"Annotator"),(.378,"Configuration choice"),
                   (.665,"Response variation"),(.885,"Generated answer")):
        art.label(fig,x,.940,txt,size=22)

    ax=fig.add_axes([.025,.550,.13,.235]); human(ax)
    art.label(fig,.092,.800,r"Worker $j$",size=21)
    art.label(fig,.092,.513,r"$P(z_j=g)=\omega_g$",size=21)
    art.label(fig,.092,.463,"Group proportion",size=17,color=art.GRAY)
    art.label(fig,.092,.422,"Shared across regions",size=16,color=art.GRAY)
    art.arrow(fig,(.154,.673),(.222,.747),color=art.BLUE)
    art.arrow(fig,(.154,.633),(.222,.404))
    art.label(fig,.181,.768,r"$\rho_j$",size=23,color=art.BLUE)
    art.label(fig,.204,.558,r"$1-\rho_j$",size=20,color=art.GRAY)

    art.label(fig,.378,.851,"Class-consistent",size=20,color=art.BLUE)
    art.label(fig,.378,.559,"Group-independent",size=20,color=art.GRAY)
    xslots = (.240,.400)
    for y, row in ((.625,0),(.334,1)):
        for k, x in enumerate(xslots):
            ax=art.tile(fig,[x,y,.130,.210],bounds=(10,90,88,12))
            art.draw_regions(ax,definition)
            art.dots(ax,candidates[k],art.BLUE if k==0 else art.ORANGE,size=155)
            value = rf"$\pi_{{rg{k+1}}}$" if row==0 else r"$1/K_r$"
            art.label(fig,x+.065,y-.025,value,size=23,color=art.BLUE if row==0 else art.GRAY)
    art.label(fig,.378,.265,r"Fixed configurations $S_{r1},\;S_{r2}$",size=18,color=art.GRAY)
    art.arrow(fig,(.538,.726),(.576,.654),color=art.BLUE)
    art.arrow(fig,(.538,.433),(.576,.634))

    # Illustrate F for a selected two-point candidate. All contours come directly
    # from the same fixed spatial-density function used in the current code.
    art.label(fig,.665,.855,r"Example: $S_{r1}$",size=18,color=art.GRAY)
    ax=art.tile(fig,[.589,.593,.153,.236],bounds=(10,90,86,14))
    art.draw_regions(ax,definition)
    ax.contourf(xx*100,yy*100,density,levels=[.1,1,5,15,25,35],
                 colors=["#EFF4F6","#D6E5EC","#AFCDDC","#79AAC1","#397E9E"],alpha=.9)
    ax.contour(xx*100,yy*100,density,levels=[5,15,25],colors=[art.BLUE],linewidths=.7)
    ax.scatter(np.array(candidates[0])[:,0],np.array(candidates[0])[:,1],
                 s=27,color=art.BLUE,edgecolors="white",linewidth=.6,zorder=10)
    art.label(fig,.665,.563,"Position density",size=18)
    art.label(fig,.665,.521,r"$\sigma=0.05,\quad\varepsilon=0.01$",size=17,color=art.GRAY)

    ax=fig.add_axes([.607,.293,.123,.138])
    ax.bar(np.arange(6),count_prob,width=.65,color=["#B6C8D0"]*2+[art.BLUE]+["#B6C8D0"]*3)
    ax.set_xlim(-.6,5.6); ax.set_ylim(0,.42)
    ax.set_xticks(range(6),["0","1","2","3","4","5+"])
    ax.set_yticks([0,.4],["0","0.4"])
    ax.tick_params(axis="both",labelsize=13,length=3,pad=3)
    ax.spines[["top","right"]].set_visible(False)
    art.label(fig,.665,.460,r"Point count $N$  ($m=2$)",size=18)
    art.label(fig,.665,.244,r"$P(N\mid m)\propto0.5^{|N-m|}$",size=16,color=art.GRAY)

    ax=art.tile(fig,[.804,.566,.160,.285],bounds=(10,90,90,10))
    art.draw_regions(ax,definition)
    art.dots(ax,response,art.HOT,size=155)
    art.label(fig,.885,.855,"Illustrative response",size=17,color=art.GRAY)
    art.label(fig,.885,.526,r"$Y_{rj}=\{y_1,y_2\}$",size=22)
    art.label(fig,.885,.470,r"$Y_{rj}\sim F_r(\cdot\mid S_{rk})$",size=20)
    art.arrow(fig,(.748,.709),(.798,.709))
    # Count and location both determine the generated response.
    art.arrow(fig,(.745,.363),(.803,.600),curve=.20)
    art.label(fig,.877,.355,r"$F_r(Y\mid S)$",size=21)
    art.label(fig,.877,.306,"Count × point positions",size=17,color=art.GRAY)

    # This is the model equation, not a claim that the candidate weights are cell probabilities.
    equation=(r"$p(Y_{rj}\mid z_j=g)=\rho_j\sum_k\pi_{rgk}F_r(Y_{rj}\mid S_{rk})"
              r"+(1-\rho_j)\,\frac{1}{K_r}\sum_k F_r(Y_{rj}\mid S_{rk})$")
    art.label(fig,.510,.143,equation,size=22)
    art.label(fig,.500,.065,r"Estimate by EM (MAP):  $\theta=\{\omega,\rho,\pi\}$",size=23)
    art.save(fig,"generative_model")

    metadata={
        "outputs":["generative_model.png","generative_model.svg"],
        "canvas":[1920,1080],"aspect_ratio":"16:9","language":"English",
        "model":"Conditional annotation generative model with fixed observed candidate configurations",
        "learned_parameters":{
            "omega_g":"Group prior for each worker; one group shared across image regions.",
            "rho_j":"Probability of selecting the class-consistent response component.",
            "pi_rgk":"Candidate choice probability in region r for group g."
        },
        "latent_sampling_steps":[
            "For each worker: z_j ~ Categorical(omega).",
            "For each worker and region: u_rj ~ Bernoulli(rho_j).",
            "If u_rj=1: k ~ Categorical(pi_r,z_j). Otherwise: k ~ Uniform({1,...,K_r}).",
            "Let m=|S_rk|. Draw N with P(N=n|m)=0.5^abs(n-m)/Z_m, n>=0, Z_m=(1+0.5-0.5^(m+1))/(1-0.5).",
            "Draw N independent locations from the mean of candidate-centered, boundary-truncated Gaussian densities (weight 0.99) plus the uniform spatial density (weight 0.01). Empty candidates use uniform spatial density."
        ],
        "fixed_settings":{"count_decay":model.Q_COUNT,"normalized_position_sigma":model.BANDWIDTH,
                           "uniform_spatial_fraction":model.UNIFORM_FRACTION},
        "priors":{"omega":"Dirichlet(2,...,2)","rho_j":"Beta(2,2)",
                    "pi_rg":"Dirichlet(1+1/K_r,...,1+1/K_r)"},
        "inference":"MAP updates by EM. q_jg is a posterior group probability derived during inference, not an additional generative parameter.",
        "example":{"candidates_pixels":candidates,"selected_candidate_index":0,
                     "response_pixels":response,"response_kind":"Hand-specified possible response for illustration, not a fitted result or random sample.",
                     "seed":None,"count_bins":["0","1","2","3","4","5+"],
                     "count_probabilities":count_prob.tolist(),"response_density":f_value,
                     "contour_levels":[.1,1,5,15,25,35]},
        "interpretation":"Candidate configurations are fixed from observed answers. Region outlines are visual context, not image evidence. This model does not generate microscopy images, biological cell states, or one-to-one point matches. Final output candidates remain whole observed configurations; F describes potentially perturbed responses to those candidates.",
        "processing":"Density and count-probability evaluations plus drawing only. No fitting, new experimental run, or change to saved results.",
        "source_sha256":hashes,
        "checks":art.CHECKS,
    }
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    metadata["source_files_unchanged"]=True
    (OUT/"generative_model_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"outputs":metadata["outputs"],"checks":art.CHECKS,
                      "count_probability_sum":float(count_prob.sum()),"source_files_unchanged":True}))


if __name__=="__main__":
    main()
