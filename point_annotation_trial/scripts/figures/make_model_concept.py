"""Concept-only introduction: assumptions, estimation and output.

The left column illustrates assumptions, not measured worker behaviour.
The middle/right columns reuse a saved converged synthetic example. No fitting.
Only a new PNG and its provenance JSON are written; existing figures are kept.
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
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import make_method_schematics as art

HERE = METHOD_OUTPUT_ROOT
ROOT = HERE.parents[2]
CASE = ROOT / "comparison_report/connected_split_case"
OUT = HERE / "concept_png"
from project_paths import FONT_ROOT
FONT = FONT_ROOT / "Arial.TTF"
SOURCES = {}


def source(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def label(fig, x, y, text, size=20, **kwargs):
    return art.label(fig, x, y, text, size=size, color="black", **kwargs)


def configuration(fig, rect, definition, points, color, size=140):
    ax = art.tile(fig, rect, bounds=(10, 90, 85, 15))
    art.draw_regions(ax, definition)
    art.dots(ax, points, color=color, size=size)
    return ax


def arrow(fig, start, end, curve=0):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=17, linewidth=1.2,
                   color="#7C878D", connectionstyle=f"arc3,rad={curve}"))


def save(fig, metadata):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    for text in texts:
        value = text.get_text()
        assert not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value)
        assert "QC" not in value and "$" not in value
        assert mpl.colors.to_rgba(text.get_color())[:3] == (0, 0, 0)
        resolved = Path(font_manager.findfont(text.get_fontproperties())).resolve()
        assert resolved == FONT.resolve(), resolved
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, value
    positioned = list(fig.texts) + [t for ax in fig.axes for t in ax.texts]
    for i, text in enumerate(positioned):
        a = text.get_window_extent(renderer)
        for other in positioned[i + 1:]:
            b = other.get_window_extent(renderer)
            assert not (min(a.x1, b.x1) - max(a.x0, b.x0) > 1
                        and min(a.y1, b.y1) - max(a.y0, b.y0) > 1), (text.get_text(), other.get_text())
    target = OUT / "model_concept.png"
    fig.savefig(target, dpi=300, facecolor="white")
    with Image.open(target) as im:
        assert im.size == (4800, 2700)
        rgb = im.convert("RGB")
    rgb.save(target, dpi=(300, 300))
    for path, expected in SOURCES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    metadata.update({
        "output": target.name, "source_sha256": SOURCES,
        "source_files_unchanged": True,
        "checks": {"canvas": [4800, 2700], "ratio": "16:9", "format": "PNG",
                   "mode": "RGB", "dpi": 300, "text_font": "Arial", "text_color": "black",
                   "language": "English", "no_equations": True,
                   "text_inside_canvas": True, "positioned_text_nonoverlapping": True,
                   "actual_arial_font_verified": True},
        "text": [t.get_text() for t in texts],
    })
    (OUT / "model_concept_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    print(target)


def main():
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    assert font_manager.FontProperties(fname=FONT).get_name() == "Arial"
    mpl.rcParams.update({"font.family": "Arial", "font.size": 20,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "figure.facecolor": "white", "savefig.facecolor": "white"})
    definition = read_json(CASE / "definition.json")
    results = CASE / "results/connected_split/current"
    run = read_json(results / "run.json")
    assert run["status"] == "converged"
    training = read_json(CASE / "synthetic_inputs/connected_split/qc_views.json")
    applied = read_json(CASE / "synthetic_inputs/connected_split/evaluation_views.json")
    output = read_json(results / "evaluation_outputs.json")[0]
    assert output["status"] == "converged"
    with np.load(source(results / "fitted_parameters.npz"), allow_pickle=False) as fit:
        q, rho = fit["q"].copy(), fit["rho"].copy()
        first_pi = fit["pi_qc"][0].copy()
    source(source_path("model.py"))
    source(source_path("data.py"))
    candidates = output["candidate_points_pixels"]
    assert [len(c) for c in candidates] == [2, 1]
    weights = np.asarray(output["weights"])
    by_worker = dict(zip(run["worker_ids"], q))
    mixing = np.mean([by_worker[a["worker_id"]] for a in applied["views"][0]["answers"]], axis=0)
    assert np.allclose(mixing @ np.asarray(output["pi"]), weights)
    mass = art.point_mass(candidates, weights, 100)
    smoothed = gaussian_filter(mass, 3, mode="reflect")
    assert np.isclose(mass.sum(), output["expected_count"])

    fig = plt.figure(figsize=(16, 9), dpi=120)
    for x, text in ((.171, "Assumptions"), (.500, "Estimation"), (.836, "Outputs")):
        label(fig, x, .934, text, size=26)

    # Left: a conceptual account of heterogeneity, not labelled training data.
    label(fig, .171, .807, "Different preferences", size=21)
    for k, x in enumerate((.046, .184)):
        configuration(fig, [x, .563, .112, .199], definition, candidates[k],
                      art.BLUE if k == 0 else art.ORANGE, size=145)
        label(fig, x + .056, .532, "2 points" if k == 0 else "1 point", size=19)
    label(fig, .171, .374, "Answers can vary", size=21)
    for x, k in zip((.044, .132, .220), (0, 1, 0)):
        configuration(fig, [x, .180, .078, .139], definition, candidates[k],
                      "#58666E", size=95)
    label(fig, .171, .137, "Possible answers from one annotator", size=15)

    # Middle: all points stay neutral until the response groups are inferred.
    label(fig, .500, .807, "Observed answers", size=21)
    for x, text in ((.464, "Region 1"), (.584, "Region 2")):
        label(fig, x, .748, text, size=16)
    shown_answers = []
    for y, worker, count in ((.536, "A1", 7), (.341, "B1", 3)):
        label(fig, .380, y + .083, f"{count}\nworkers", size=17, linespacing=1.15)
        for x, view in zip((.416, .536), training["views"][:2]):
            answer = next(a for a in view["answers"] if a["worker_id"] == worker)
            assert sum(a["points"] == answer["points"] for a in view["answers"]) == count
            configuration(fig, [x, y, .096, .171], definition, answer["points"],
                          "#58666E", size=105)
            shown_answers.append({"worker": worker, "count": count,
                                  "view_id": view["view_id"], "points": answer["points"]})
    # The loop explains fitting without introducing posterior symbols or updates.
    arrow(fig, (.646, .497), (.596, .247), curve=-.27)
    arrow(fig, (.407, .243), (.351, .480), curve=-.35)
    label(fig, .500, .301, "Compare and update", size=16)
    label(fig, .500, .241, "Learn preferences", size=21)
    label(fig, .500, .194, "and consistency", size=21)
    # Group membership is a fitted quantity; it is never given as a training label.
    left_class = int(np.argmax(first_pi[:, 0]))
    group_colors = [art.BLUE if np.argmax(row) == left_class else art.ORANGE for row in q]
    ax = fig.add_axes([.423, .114, .154, .032])
    ax.scatter(np.arange(len(q)), np.zeros(len(q)), s=105, c=group_colors,
               edgecolors="white", linewidths=.8)
    ax.set_xlim(-.5, len(q)-.5); ax.set_ylim(-.5, .5); ax.axis("off")
    label(fig, .500, .079, "Inferred annotator groups", size=16)

    # Right: use exact saved weights. The joint configurations stay intact.
    label(fig, .836, .807, "Configuration weights", size=21)
    for k, x in enumerate((.713, .848)):
        configuration(fig, [x, .563, .112, .199], definition, candidates[k],
                      art.BLUE if k == 0 else art.ORANGE, size=145)
        label(fig, x + .056, .532, f"{100*weights[k]:.1f}%", size=25)
    # Collect both candidates before constructing their weighted average.
    for xs, ys in (((.769,.769),(.478,.441)), ((.904,.904),(.478,.441)),
                   ((.769,.904),(.441,.441))):
        fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure,
                                 color="#7C878D", linewidth=1.2))
    arrow(fig, (.836, .441), (.836, .395))
    label(fig, .836, .365, "Average over configurations", size=18)
    ax = art.tile(fig, [.748, .134, .176, .199], bounds=(10,90,85,15))
    art.draw_regions(ax, definition)
    norm = Normalize(0, .015)
    rgba = plt.colormaps["inferno"](norm(smoothed))
    rgba[..., 3] = np.clip(smoothed / .003, 0, .96)
    ax.imshow(rgba, extent=(0,100,100,0), interpolation="bilinear", zorder=3)
    label(fig, .836, .103, "Mean point map", size=20)
    cax = fig.add_axes([.770, .050, .132, .012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, "inferno"), cax=cax,
                      orientation="horizontal", ticks=[0, .015])
    cb.ax.set_xticklabels(["0", "0.015"])
    cb.ax.tick_params(labelsize=12, length=2, pad=2)
    cb.outline.set_visible(False)
    label(fig, .945, .045, "Point mass\nper pixel", size=12, linespacing=1.1)
    arrow(fig, (.305, .511), (.339, .511))
    arrow(fig, (.656, .511), (.696, .511))

    save(fig, {
        "purpose": "A pre-algorithm conceptual introduction: assumed variation, parameter estimation from repeated whole answers, and weighted-configuration output.",
        "assumptions": "People can have different stable response preferences. Any person's answer can depart from their group preference. The left-hand two/one/two sequence is a hand-specified possible sequence, not an observed or fitted worker trajectory.",
        "left_sequence": {"kind": "illustration only", "candidate_indices": [0,1,0], "seed": None},
        "observed_examples": shown_answers,
        "fitted_memberships": q.tolist(), "fitted_response_weights": rho.tolist(),
        "display_group_relabeling": "Blue denotes the fitted group preferring the two-point candidate in this example; rust denotes the other fitted group.",
        "output_candidates": candidates, "output_weights": weights.tolist(),
        "output_expected_count": output["expected_count"],
        "interpretation": "The grey outlines provide visual context, not biological truth and not image input. Candidate sets are observed whole answers. All candidates contribute to the mean map; its three peaks do not represent a simultaneous three-point candidate. Only the model parameters are learned; the model family is predefined. The simplified centre omits the distinction between worker-parameter learning and subsequent region-only fitting; both values come from the saved converged run.",
        "processing": "Read saved answers and parameters, verify candidate weights, compute the same weighted point mass, and draw. No fitting, parameter search, image inference, or modification of existing results.",
        "display": {"smoothing": "Gaussian, reflect boundary", "sigma_pixels": 3,
                    "range": [0,.015], "alpha": "clip(smoothed_mass/0.003,0,0.96)",
                    "per_map_normalization": False},
        "unsmoothed_mass": mass.tolist(), "smoothed_display_mass": smoothed.tolist(),
    })


if __name__ == "__main__":
    main()
