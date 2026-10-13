"""A single code-native schematic of generation, MAP fitting and output weights.

Only fixed-density evaluation and aggregation of a saved converged fit are done.
PNG is the sole figure deliverable; LaTeX intermediates are temporary.
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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
import numpy as np
from scipy.special import logsumexp

import make_inference_schematics as source
import make_nature_png as export

HERE = METHOD_OUTPUT_ROOT
OUT = HERE / "estimation_png"
STEM = "generation_estimation_overview"
BLUE = "#327B97"
RUST = "#BC7956"
GRAY = "#9AA5AC"
LIGHT = "#E8ECEE"
INK = "#333C41"
COLORS = (BLUE, RUST)


def label(fig, x, y, text, size=19, ha="center", **kwargs):
    return fig.text(x, y, text, fontsize=size, color="black", ha=ha,
                    va="center", **kwargs)


def line(fig, xs, ys, color=GRAY, width=1.0, style="-"):
    fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure,
                             color=color, linewidth=width, linestyle=style))


def arrow(fig, start, end, color=GRAY, curve=0., width=1.25, style="-"):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=15,
                                  linewidth=width, color=color, linestyle=style,
                                  connectionstyle=f"arc3,rad={curve}"))


def person(fig, cx, cy, width=.038, color=BLUE):
    ax = fig.add_axes([cx-width/2, cy-width*.83, width, width*1.66])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Circle((.5, .77), .18, facecolor=color, edgecolor="none"))
    ax.add_patch(Ellipse((.5, .31), .75, .65, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((0, -.1), 1, .15, facecolor="white", edgecolor="none"))
    return ax


def tile(fig, rect, definition, points, crosses=False, density=None):
    ax = fig.add_axes(rect)
    ax.set_xlim(10, 90); ax.set_ylim(86, 14)
    ax.set_aspect("equal"); ax.axis("off")
    for region in definition["regions"].values():
        ax.add_patch(Ellipse(region["center"], 2*region["radii"][0],
                             2*region["radii"][1], facecolor="#EFF1F2",
                             edgecolor="#B8C1C5", linewidth=.8))
    if density is not None:
        xx, yy, zz = density
        ax.contourf(xx*100, yy*100, zz,
                    levels=[.1, 1, 5, 15, 25, 35, 70],
                    colors=["#F0F5F7", "#DEEAF0", "#BDD7E3", "#8FBACC", "#568DA6", "#316F8B"],
                    alpha=.92)
        ax.contour(xx*100, yy*100, zz, levels=[5, 15, 25],
                   colors=[BLUE], linewidths=.65)
    p = np.asarray(points)
    marker_style = {} if crosses else {"edgecolors": "white"}
    ax.scatter(p[:, 0], p[:, 1], s=96 if not crosses else 118,
               marker="o" if not crosses else "+", color=INK if not crosses else RUST,
               linewidths=1.1 if not crosses else 2.0,
               zorder=8, **marker_style)
    return ax


def probability_bars(fig, y, values, color):
    for x, value in zip((.461, .581), values):
        ax = fig.add_axes([x-.037, y-.017, .074, .014])
        ax.barh([0], [1], color=LIGHT, height=1)
        ax.barh([0], [value], color=color, height=1)
        ax.set_xlim(0, 1); ax.set_ylim(-.5, .5); ax.axis("off")
        label(fig, x, y+.019, f"{value:.2f}", size=19)


def inspect_saved_fit(training, fitted, run):
    """Re-evaluate the saved objective and posterior; do not update parameters."""
    worker_ids = run["worker_ids"]
    pi = fitted["pi_qc"]
    omega, rho = fitted["omega"], fitted["rho"]
    all_log_m, candidates_by_region = [], []
    for r, view in enumerate(training["views"]):
        candidates = []
        for ans in view["answers"]:
            point_set = sorted(map(tuple, ans["points"]))
            if point_set not in candidates:
                candidates.append(point_set)
        candidates = tuple(np.asarray(c, dtype=float)/100 for c in candidates)
        assert len(candidates) == 2
        responses = {a["worker_id"]: np.asarray(a["points"])/100 for a in view["answers"]}
        log_f = np.stack([source.model.log_f_for_points(responses[w], candidates) for w in worker_ids])
        log_group = logsumexp(log_f[:, None, :] + np.log(pi[r])[None, :, :], axis=-1)
        log_equal = logsumexp(log_f, axis=1)-np.log(len(candidates))
        log_m = np.logaddexp(np.log(rho)[:, None] + log_group,
                            np.log1p(-rho)[:, None] + log_equal[:, None])
        all_log_m.append(log_m)
        candidates_by_region.append([c.tolist() for c in candidates])
    logits = np.log(omega)[None, :] + np.sum(all_log_m, axis=0)
    q = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
    prior = np.log(omega).sum() + np.log(rho).sum() + np.log1p(-rho).sum() + np.log(pi).sum()/2
    data = logsumexp(logits, axis=1).sum()
    objective = float(data + prior)
    assert np.allclose(q, fitted["q"], atol=1e-8)
    assert np.isclose(objective, run["objective"], atol=1e-7, rtol=0)
    return {"objective_recomputed": objective, "log_likelihood": float(data),
            "log_prior_without_constants": float(prior), "posterior_q": q.tolist(),
            "maximum_q_difference": float(np.max(abs(q-fitted["q"]))),
            "normalized_candidates_by_region": candidates_by_region}


def main():
    definition, run, training, application, fitted, output = source.input_data()
    audit = inspect_saved_fit(training, fitted, run)
    existing = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in HERE.rglob("*.png") if p.parent != OUT}
    candidates = [definition["click_templates"][key] for key in ("separated", "merged")]
    response = training["views"][0]["answers"][0]["points"]
    assert response == candidates[0]
    pi = fitted["pi_qc"][0]
    q = fitted["q"]
    v = q.mean(axis=0)
    weights = v @ pi
    assert np.allclose(weights, output["weights"], atol=1e-6)
    grid = np.linspace(0, 1, 251)
    xx, yy = np.meshgrid(grid, grid)
    zz = source.model.spatial_density(np.c_[xx.ravel(), yy.ravel()],
                                     np.asarray(candidates[0])/100).reshape(xx.shape)
    count_prob = np.exp(source.model.log_count_probability(2, np.arange(5)))
    count_prob = np.r_[count_prob, 1-count_prob.sum()]

    OUT.mkdir(exist_ok=True)
    export.configure()
    export.OUT = OUT
    fig = plt.figure(figsize=(16, 9), dpi=120)

    # Generation: the parameter is placed at the particular choice it controls.
    for x, title in ((.090, "Group prior"), (.261, "Response path"),
                     (.493, "Configuration choice"), (.732, "Count + positions"),
                     (.917, "Observed answer")):
        label(fig, x, .960, title, size=20)
    label(fig, .090, .909, r"$P(z_j=g)=\omega_g$", size=21)
    label(fig, .261, .909, r"$\rho_j$", size=25)
    label(fig, .493, .907, r"$\pi_{rgk}$", size=25)
    label(fig, .732, .909, r"Fixed $F_r(Y\mid S)$", size=20)
    label(fig, .917, .909, r"$Y_{rj}$", size=24)

    for g, x in enumerate((.050, .133)):
        label(fig, x, .848, f"Group {g+1}", size=17)
        person(fig, x, .768, width=.047, color=COLORS[g])
        label(fig, x, .699, rf"$\omega_{g+1}={fitted['omega'][g]:.2f}$", size=19)
    label(fig, .090, .617, "One group per worker", size=16)
    label(fig, .090, .577, "Shared across regions", size=16)
    # Both groups enter the same per-answer mixture. The group selects a row of pi.
    line(fig, [.177, .181, .181, .177], [.820, .820, .681, .681], width=.9)
    arrow(fig, (.181, .750), (.215, .714))
    arrow(fig, (.181, .750), (.215, .582), curve=.08)

    label(fig, .266, .760, r"$\rho_j$", size=22)
    label(fig, .266, .716, "Use group", size=18)
    label(fig, .266, .626, r"$1-\rho_j$", size=22)
    label(fig, .266, .583, "Equal choice", size=18)
    arrow(fig, (.320, .712), (.342, .712))
    arrow(fig, (.320, .582), (.342, .582))
    line(fig, [.347, .347], [.646, .758], width=.9)
    line(fig, [.347, .354], [.758, .758], width=.9)
    line(fig, [.347, .354], [.646, .646], width=.9)

    for k, x in enumerate((.461, .581)):
        tile(fig, [x-.043, .793, .086, .118], definition, candidates[k])
        label(fig, x, .781, "2 points" if k == 0 else "1 point", size=16)
    for g, y in enumerate((.730, .657)):
        label(fig, .383, y, f"Group {g+1}", size=16)
        probability_bars(fig, y, pi[g], COLORS[g])
    label(fig, .383, .584, "Either", size=16)
    probability_bars(fig, .584, [.5, .5], GRAY)
    # The display uses one fitted region; all rows share the same candidate set.
    line(fig, [.625, .637, .637, .625], [.750, .750, .574, .574], width=1.0)
    arrow(fig, (.637, .729), (.659, .729))

    tile(fig, [.663, .704, .138, .185], definition, candidates[0], density=(xx, yy, zz))
    label(fig, .732, .700, "2-point example", size=15)
    ax = fig.add_axes([.678, .596, .107, .075])
    ax.bar(np.arange(6), count_prob, color=["#BDCDD4"]*2+[BLUE]+["#BDCDD4"]*3, width=.68)
    ax.set_ylim(0, .41); ax.set_xlim(-.55, 5.55)
    ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
    ax.set_yticks([]); ax.spines[["top", "left", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=12, length=2, pad=2)
    label(fig, .732, .548, "Point count", size=15)
    arrow(fig, (.805, .787), (.841, .787))
    tile(fig, [.841, .707, .151, .194], definition, response, crosses=True)
    label(fig, .917, .691, "Point-set response", size=17)
    label(fig, .917, .619, r"$p(Y_{rj}\mid z_j=g)$", size=22)
    label(fig, .917, .570, "Response density", size=17)

    # This density is evaluated, not estimated as an independent parameter.
    label(fig, .505, .491,
          r"$\displaystyle p(Y_{rj}\mid z_j=g)"
          r"=\rho_j\sum_k\pi_{rgk}F_r(Y_{rj}\mid S_{rk})"
          r"+(1-\rho_j)\frac{1}{K_r}\sum_kF_r(Y_{rj}\mid S_{rk})$", size=23)
    line(fig, [.020, .980], [.444, .444], color="#D4DADD", width=.8)

    # Inference: expected assignments and parameter updates, not hard sorting.
    label(fig, .181, .404, "Infer hidden choices", size=22)
    label(fig, .522, .404, "Update parameters", size=22)
    label(fig, .181, .355, "Posterior group membership", size=17)
    for j in range(10):
        person(fig, .037 + j*.0317, .297, width=.023,
               color=COLORS[int(np.argmax(q[j]))])
    label(fig, .132, .252, "7 workers", size=16)
    label(fig, .291, .252, "3 workers", size=16)
    label(fig, .181, .202,
          r"$\displaystyle q_{jg}\propto\omega_g\prod_r p(Y_{rj}\mid z_j=g)$", size=20)
    label(fig, .181, .150, "Also infer path + configuration", size=16)

    for y, symbol, text in ((.340, r"$\omega$", "Group proportions"),
                            (.274, r"$\rho$", "Follow-group share"),
                            (.208, r"$\pi$", "Configuration choices")):
        label(fig, .414, y, symbol, size=25)
        label(fig, .447, y, text, size=18, ha="left")
    label(fig, .534, .150, "Expected counts + priors", size=17)
    arrow(fig, (.348, .307), (.389, .307), color=BLUE)
    arrow(fig, (.390, .175), (.341, .175), color=BLUE, curve=-.55)
    label(fig, .367, .232, "Repeat", size=14)
    label(fig, .360, .069,
          r"$\displaystyle\max_{\omega,\rho,\pi}\;"
          r"\sum_j\log\!\left[\sum_g\omega_g"
          r"\prod_r p(Y_{rj}\mid z_j=g)\right]"
          r"+\log p(\omega,\rho,\pi)$", size=21)

    line(fig, [.714, .714], [.026, .423], color="#D4DADD", width=.8)
    arrow(fig, (.669, .308), (.727, .308), color=BLUE)
    label(fig, .853, .404, "Final configuration weights", size=21)
    label(fig, .853, .351, r"$\textstyle w_{rk}=\sum_g v_{rg}\,\pi_{rgk}$", size=23)
    label(fig, .853, .288,
          rf"Mean membership:  $v_r=({v[0]:.2f},\,{v[1]:.2f})$", size=16)
    for k, x in enumerate((.775, .925)):
        label(fig, x, .243, "2 points" if k == 0 else "1 point", size=18)
        tile(fig, [x-.055, .097, .110, .146], definition, candidates[k])
        label(fig, x, .064, f"{weights[k]:.3f}", size=27)

    FigureCanvasPgf(fig)
    export.save_png(fig, STEM)
    for path, sha in {**source.HASHES, **existing}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    metadata = {
        "file": f"{STEM}.png",
        "description": "Conditional answer generation, posterior assignments, EM/MAP parameter updates, and final whole-configuration weights in one figure.",
        "style": {"size": [4800, 2700], "aspect_ratio": "16:9", "text": "Black Arial",
                  "math": "XeLaTeX, STIX Two Math", "language": "English", "figure_formats": ["PNG"]},
        "sources": source.HASHES,
        "saved_example": {"fit_status": run["status"], "whole_candidates_pixels": candidates,
                          "response_pixels": response, "omega": fitted["omega"].tolist(),
                          "rho": fitted["rho"].tolist(), "pi_first_region": pi.tolist(),
                          "q": q.tolist(), "v": v.tolist(), "weights": weights.tolist(),
                          "number_of_regions": len(training["views"]),
                          "worker_ids": run["worker_ids"]},
        "numeric_checks": audit,
        "fixed_response_density": {"count_decay": source.model.Q_COUNT,
                                   "position_sigma": source.model.BANDWIDTH,
                                   "uniform_fraction": source.model.UNIFORM_FRACTION,
                                   "example_candidate_index": 0,
                                   "count_bins": ["0", "1", "2", "3", "4", "5+"],
                                   "count_probabilities": count_prob.tolist()},
        "equation_notes": [
            "The product over r includes only regions answered by worker j. One latent group is shared across these regions.",
            "q is normalized across groups; the proportionality form omits only that normalization.",
            "Path and configuration posteriors are inferred as well as q. Updates use expected assignments, not hard group labels.",
            "v_rg is the mean q_jg among participating workers, not omega_g. The displayed synthetic region contains all ten workers.",
            "rho controls the mixture during fitting and is not multiplied into the final map a second time.",
            "F and candidate sets are fixed. Neither image pixels nor the illustrative outlines contribute to fitting.",
            "Response density, group membership, and configuration weights are distinct quantities; none is biological accuracy.",
            "The figures show saved converged values, not a fabricated intermediate optimization sequence.",
            "Dirichlet(2) for omega, Beta(2,2) for rho, and Dirichlet(1+1/K) for each pi row are the fixed priors."
        ],
        "processing": "Re-evaluated fixed observation densities, saved objective and group posterior; formed weights by mean(q) @ pi. No fitting or new image inference.",
        "source_results_unchanged": True, "existing_figures_unchanged": True,
        "render_checks": export.CHECKS,
    }
    (OUT / f"{STEM}_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"output": str(OUT / f"{STEM}.png"), "source_results_unchanged": True,
                      "objective_verified": audit["objective_recomputed"],
                      "weights": weights.tolist()}), flush=True)


if __name__ == "__main__":
    main()
