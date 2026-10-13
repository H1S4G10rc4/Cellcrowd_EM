"""Split the combined schematic into two reflowed 16:9 PNG figures.

Preserve the combined figure and every stored estimate. All values and point
configurations are inherited from the same saved, converged synthetic example.
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
import numpy as np

import make_estimation_overview as base

HERE = METHOD_OUTPUT_ROOT
OUT = HERE / "estimation_png"
STEMS = ("generative_model", "estimation_and_outputs")
label, arrow, line, person, tile = base.label, base.arrow, base.line, base.person, base.tile
BLUE, RUST, GRAY, LIGHT = base.BLUE, base.RUST, base.GRAY, base.LIGHT


def save(fig, stem):
    FigureCanvasPgf(fig)
    base.export.save_png(fig, stem)


def enlarge_markers(ax, factor=1.45):
    # Only scale marker area, never point positions or the underlying regions.
    points = ax.collections[-1]
    points.set_sizes(points.get_sizes() * factor)


def bars(fig, y, values, color):
    for x, value in zip((.465, .591), values):
        ax = fig.add_axes([x-.041, y-.023, .082, .019])
        ax.barh([0], [1], height=1, color=LIGHT)
        ax.barh([0], [value], height=1, color=color)
        ax.set_xlim(0, 1); ax.set_ylim(-.5, .5); ax.axis("off")
        label(fig, x, y+.036, f"{value:.2f}", size=24)


def generation(definition, fitted, candidates, response, densities, count_prob):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for x, text in ((.092, "Group prior"), (.271, "Response path"),
                    (.490, "Configuration choice"), (.735, "Count + positions"),
                    (.918, "Observed answer")):
        label(fig, x, .952, text, size=22)
    label(fig, .092, .876, r"$P(z_j=g)=\omega_g$", size=23)
    label(fig, .271, .876, r"$\rho_j$", size=29)
    label(fig, .490, .876, r"$\pi_{rgk}$", size=29)
    label(fig, .735, .876, r"Fixed $F_r(Y\mid S)$", size=22)
    label(fig, .918, .876, r"$Y_{rj}$", size=28)

    for g, x in enumerate((.052, .147)):
        label(fig, x, .791, f"Group {g+1}", size=19)
        person(fig, x, .687, width=.063, color=base.COLORS[g])
        label(fig, x, .585, rf"$\omega_{g+1}={fitted['omega'][g]:.2f}$", size=21)
    label(fig, .098, .391, "One group per worker", size=18)
    label(fig, .098, .329, "Shared across regions", size=18)
    line(fig, [.194, .198, .198, .194], [.753, .753, .561, .561])
    arrow(fig, (.198, .653), (.220, .557))
    arrow(fig, (.198, .653), (.211, .280), curve=.04)
    label(fig, .273, .622, r"$\rho_j$", size=26)
    label(fig, .273, .560, "Use group", size=21)
    label(fig, .273, .351, r"$1-\rho_j$", size=26)
    label(fig, .273, .283, "Equal choice", size=21)
    arrow(fig, (.332, .555), (.349, .555))
    arrow(fig, (.332, .280), (.349, .280))
    line(fig, [.354, .354], [.402, .605])
    line(fig, [.354, .360], [.605, .605])
    line(fig, [.354, .360], [.402, .402])

    for k, x in enumerate((.465, .591)):
        ax = tile(fig, [x-.051, .698, .102, .175], definition, candidates[k])
        enlarge_markers(ax)
        label(fig, x, .661, "2 points" if k == 0 else "1 point", size=20)
    for g, y in enumerate((.554, .423)):
        label(fig, .386, y, f"Group {g+1}", size=18)
        bars(fig, y, fitted["pi_qc"][0, g], base.COLORS[g])
    label(fig, .386, .280, "Either", size=18)
    bars(fig, .280, [.5, .5], GRAY)
    line(fig, [.638, .648, .648, .638], [.730, .730, .257, .257])
    arrow(fig, (.648, .718), (.665, .718))

    ax = tile(fig, [.663, .583, .145, .270], definition, candidates[0], density=densities[0])
    enlarge_markers(ax)
    label(fig, .735, .567, "2-point example", size=19)
    ax = fig.add_axes([.678, .326, .115, .168])
    ax.bar(np.arange(6), count_prob, color=["#BDCDD4"]*2+[BLUE]+["#BDCDD4"]*3, width=.68)
    ax.set_xlim(-.55, 5.55); ax.set_ylim(0, .41)
    ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
    ax.set_yticks([]); ax.spines[["top", "left", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=16, length=3, pad=4)
    label(fig, .735, .261, "Point count", size=19)
    arrow(fig, (.811, .718), (.841, .718))
    ax = tile(fig, [.841, .583, .151, .270], definition, response, crosses=True)
    enlarge_markers(ax)
    label(fig, .918, .536, "Point-set response", size=19)
    label(fig, .918, .405, r"$p(Y_{rj}\mid z_j=g)$", size=25)
    label(fig, .918, .330, "Response density", size=20)
    label(fig, .506, .105,
          r"$\displaystyle p(Y_{rj}\mid z_j=g)"
          r"=\rho_j\sum_k\pi_{rgk}F_r(Y_{rj}\mid S_{rk})"
          r"+(1-\rho_j)\frac{1}{K_r}\sum_k F_r(Y_{rj}\mid S_{rk})$", size=27)
    save(fig, STEMS[0])


def estimation(definition, fitted, candidates, response, densities, training):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    q = fitted["q"]
    v = q.mean(axis=0)
    weights = v @ fitted["pi_qc"][0]
    label(fig, .120, .951, "Observed answers", size=24)
    label(fig, .493, .951, "Infer hidden choices", size=24)
    label(fig, .890, .951, "Final configuration weights", size=21)
    for x, text in ((.066, "Region 1"), (.182, "Region 2")):
        label(fig, x, .851, text, size=19)
    for y, worker_id, count in ((.652, "A1", 7), (.362, "B1", 3)):
        for x, view in zip((.066, .182), training["views"][:2]):
            answer = next(a for a in view["answers"] if a["worker_id"] == worker_id)
            assert sum(a["points"] == answer["points"] for a in view["answers"]) == count
            ax = tile(fig, [x-.049, y, .098, .174], definition, answer["points"])
            enlarge_markers(ax)
        label(fig, .123, y-.035, f"{count} workers", size=23)
    label(fig, .123, .229, "Same workers\nacross regions", size=19, linespacing=1.3)
    arrow(fig, (.238, .723), (.296, .723))

    label(fig, .493, .878, "Compare counts + positions", size=20)
    for k, x in enumerate((.380, .606)):
        ax = tile(fig, [x-.071, .678, .142, .195], definition, response,
                  crosses=True, density=densities[k])
        enlarge_markers(ax)
        label(fig, x, .659, rf"$F_r(Y_{{rj}}\mid S_{{r{k+1}}})$", size=22)
    label(fig, .493, .585,
          r"$\displaystyle q_{jg}\propto\omega_g\prod_r p(Y_{rj}\mid z_j=g)$", size=25)
    label(fig, .493, .510, "Posterior group membership", size=19)
    for j in range(10):
        person(fig, .317 + .0384*j, .450, width=.028, color=base.COLORS[int(q[j].argmax())])
    label(fig, .430, .401, "7 workers", size=18)
    label(fig, .625, .401, "3 workers", size=18)
    label(fig, .493, .345, "Also infer path + configuration", size=18)

    label(fig, .493, .290, "Update parameters", size=22)
    label(fig, .493, .247, "Expected counts + priors", size=16)
    for x, symbol, text in ((.346, r"$\omega$", "Group proportions"),
                            (.508, r"$\rho$", "Follow-group share"),
                            (.670, r"$\pi$", "Choice weights")):
        label(fig, x, .202, symbol, size=28)
        label(fig, x, .153, text, size=17)
    arrow(fig, (.711, .717), (.711, .278), curve=-.12, color=BLUE)
    arrow(fig, (.285, .276), (.285, .717), curve=-.12, color=BLUE)
    label(fig, .734, .480, "Repeat", size=15, rotation=90)
    label(fig, .376, .064,
          r"$\displaystyle\max_{\omega,\rho,\pi}\;"
          r"\sum_j\log\!\left[\sum_g\omega_g\prod_r p(Y_{rj}\mid z_j=g)\right]"
          r"+\log p(\omega,\rho,\pi)$", size=22)

    line(fig, [.778, .778], [.020, .916], color="#D4DADD", width=.8)
    arrow(fig, (.752, .574), (.800, .574), color=BLUE)
    for k, y in enumerate((.651, .351)):
        label(fig, .892, y+(.225 if k == 0 else .204), "2 points" if k == 0 else "1 point", size=23)
        ax = tile(fig, [.811, y, .162, .217], definition, candidates[k])
        enlarge_markers(ax, 1.65)
        label(fig, .892, y-(.012 if k == 0 else .034), f"{weights[k]:.3f}", size=31)
    label(fig, .892, .226, r"$\textstyle w_{rk}=\sum_g v_{rg}\,\pi_{rgk}$", size=25)
    label(fig, .892, .148, "Mean worker membership", size=17)
    label(fig, .892, .078, rf"$v_r=({v[0]:.2f},\,{v[1]:.2f})$", size=24)
    save(fig, STEMS[1])


def main():
    definition, run, training, application, fitted, output = base.source.input_data()
    audit = base.inspect_saved_fit(training, fitted, run)
    new_pngs = {OUT / f"{stem}.png" for stem in STEMS}
    old = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in HERE.rglob("*.png") if p not in new_pngs}
    original = OUT / "generation_estimation_overview_manifest.json"
    prior_manifest = json.loads(original.read_text())
    old[str(original)] = hashlib.sha256(original.read_bytes()).hexdigest()
    candidates = prior_manifest["saved_example"]["whole_candidates_pixels"]
    response = prior_manifest["saved_example"]["response_pixels"]
    assert np.array_equal(fitted["omega"], prior_manifest["saved_example"]["omega"])
    assert np.array_equal(fitted["pi_qc"][0], prior_manifest["saved_example"]["pi_first_region"])
    assert np.allclose(fitted["q"].mean(axis=0) @ fitted["pi_qc"][0],
                       prior_manifest["saved_example"]["weights"], atol=1e-12)
    grid = np.linspace(0, 1, 251)
    xx, yy = np.meshgrid(grid, grid)
    densities = []
    for candidate in candidates:
        zz = base.source.model.spatial_density(np.c_[xx.ravel(), yy.ravel()],
                                              np.asarray(candidate)/100).reshape(xx.shape)
        densities.append((xx, yy, zz))
    count_prob = np.asarray(prior_manifest["fixed_response_density"]["count_probabilities"])
    base.export.configure()
    base.export.OUT = OUT
    generation(definition, fitted, candidates, response, densities, count_prob)
    estimation(definition, fitted, candidates, response, densities, training)
    for path, expected in {**base.source.HASHES, **old}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    manifest = {
        "files": {f"{STEMS[0]}.png": "Generation: group prior, response path, whole-configuration choice, and fixed count/location variation.",
                  f"{STEMS[1]}.png": "Estimation: repeated observed answers, inferred assignments, parameter updates, MAP objective, and output weights."},
        "change": "Reflowed the combined diagram into two full-width 16:9 figures with larger point sets and text. Original combined figure retained.",
        "style": prior_manifest["style"],
        "saved_example": prior_manifest["saved_example"],
        "fixed_response_density": prior_manifest["fixed_response_density"],
        "equation_notes": prior_manifest["equation_notes"],
        "additional_display": "The second diagram compares the same saved two-point answer with both fixed candidate densities; no extra response or result is generated.",
        "numeric_checks": audit,
        "processing": "Fixed-density evaluation and saved-value readback only. No optimization, refitting, or new image inference.",
        "source_sha256": base.source.HASHES,
        "original_figure_sha256": old[str(OUT / "generation_estimation_overview.png")],
        "all_previous_figures_unchanged": True,
        "all_source_results_unchanged": True,
        "checks": base.export.CHECKS,
    }
    (OUT / "split_figures_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"files": [str(p) for p in sorted(new_pngs)],
                      "original_figure_unchanged": True, "results_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
