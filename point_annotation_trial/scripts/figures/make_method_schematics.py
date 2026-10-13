"""Illustrate the current point-set model using saved inputs and fitted values.

No model is fitted. Only this output directory is written. SVG text stays editable.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, to_rgb
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


OUT = METHOD_OUTPUT_ROOT
ROOT = OUT.parents[2]
REPORT = ROOT / "comparison_report"
SAVED = ROOT / "point_annotation_trial/outputs/convergence_extension"
BLUE = "#287395"
ORANGE = "#BC714E"
HOT = "#E36B39"
INK = "#26333B"
GRAY = "#66747C"
LIGHT = "#E3E8EA"
SOURCE_HASHES = {}
VALUES = {}
CHECKS = []

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 19,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#C3CBCF", "axes.linewidth": .8,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "current-point-set-method",
})


def register(path):
    path = Path(path)
    SOURCE_HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(register(path).read_text())


def label(fig, x, y, text, size=21, color=INK, ha="center", **kwargs):
    return fig.text(x, y, text, ha=ha, va="center", fontsize=size,
                    color=color, **kwargs)


def arrow(fig, start, end, color="#879298", curve=0):
    patch = FancyArrowPatch(start, end, transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=17,
                            linewidth=1.25, color=color,
                            connectionstyle=f"arc3,rad={curve}")
    fig.add_artist(patch)


def tile(fig, rect, bounds=(0, 100, 100, 0), frame=False):
    ax = fig.add_axes(rect)
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame)
    return ax


def draw_regions(ax, definition):
    # The saved generating regions are explanatory geometry, never image evidence.
    for region in definition["regions"].values():
        ax.add_patch(Ellipse(region["center"], 2*region["radii"][0],
                             2*region["radii"][1], facecolor="#DDE2E3",
                             edgecolor="#8E9AA0", linewidth=1.15, alpha=.7))


def dots(ax, points, color=BLUE, size=160, real=False):
    p = np.asarray(points).reshape(-1, 2)
    if len(p):
        if real:
            ax.scatter(p[:, 0], p[:, 1], s=size*1.8, color="#19242B",
                       linewidths=0, alpha=.9, zorder=6)
        ax.scatter(p[:, 0], p[:, 1], s=size, color=color,
                   edgecolors="white" if not real else "#FFE5C1",
                   linewidths=1.2 if not real else .6, zorder=7)


def point_mass(candidates, weights, size):
    mass = np.zeros((size, size), dtype=float)
    for candidate, weight in zip(candidates, weights):
        p = np.asarray(candidate).reshape(-1, 2)
        if not len(p):
            continue
        x, y = np.clip(p[:, 0], 0, size-1), np.clip(p[:, 1], 0, size-1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0+1, size-1), np.minimum(y0+1, size-1)
        fx, fy = x-x0, y-y0
        for xx, yy, w in ((x0, y0, (1-fx)*(1-fy)), (x1, y0, fx*(1-fy)),
                          (x0, y1, (1-fx)*fy), (x1, y1, fx*fy)):
            np.add.at(mass, (yy, xx), weight*w)
    expected = sum(len(p)*w for p, w in zip(candidates, weights))
    assert np.isclose(mass.sum(), expected)
    return mass


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    escaped = []
    for txt in fig.findobj(matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text():
            continue
        value = txt.get_text()
        assert not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value), value
        assert "QC" not in value, value
        box = txt.get_window_extent(renderer)
        if box.width and box.height and not (
                box.x0 >= 0 and box.y0 >= 0
                and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height):
            escaped.append(value)
    assert not escaped, escaped
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=120, facecolor="white")
    with Image.open(OUT / f"{stem}.png") as im:
        assert im.size == (1920, 1080)
    CHECKS.append({"figure": stem, "size": [1920, 1080],
                   "text_inside_canvas": True, "english_only": True})
    plt.close(fig)


def configurations():
    case = REPORT / "connected_split_case"
    definition = read_json(case / "definition.json")
    run = read_json(case / "results/connected_split/current/run.json")
    assert run["status"] == "converged"
    row = read_json(case / "results/connected_split/current/evaluation_outputs.json")[0]
    candidates, weights = row["candidate_points_pixels"], row["weights"]
    order = sorted(range(len(candidates)), key=lambda k: -len(candidates[k]))
    assert [len(candidates[k]) for k in order] == [2, 1]
    mass = point_mass(candidates, weights, 100)
    smooth = gaussian_filter(mass, 3, mode="reflect")
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for x, txt in ((.135, "Illustrative regions"), (.375, "Observed answers"),
                   (.610, "Candidate weights"), (.857, "Mean point map")):
        label(fig, x, .90, txt, size=21)
    label(fig, .135, .852, "Synthetic example", size=16, color=GRAY)
    label(fig, .857, .852, "Weighted average", size=16, color=GRAY)
    ax = tile(fig, [.025, .335, .22, .39], bounds=(10, 90, 90, 10))
    draw_regions(ax, definition)
    for y, k, count, col in zip((.56, .14), order, (7, 3), (BLUE, ORANGE)):
        for x in (.280, .515):
            ax = tile(fig, [x, y, .19, .285], bounds=(10, 90, 84, 16))
            draw_regions(ax, definition)
            dots(ax, candidates[k], col, size=165)
        label(fig, .375, y+.283, f"{count} workers", size=19, color=col)
        label(fig, .610, y+.283, f"{len(candidates[k])} point" + ("s" if len(candidates[k])>1 else ""), size=19, color=col)
        label(fig, .610, y-.017, f"{weights[k]:.3f}", size=26, color=col)
        arrow(fig, (.249, .53), (.278, y+.14))
        arrow(fig, (.475, y+.14), (.511, y+.14))
        arrow(fig, (.712, y+.14), (.758, .53))
    ax = tile(fig, [.760, .335, .195, .39], bounds=(10, 90, 90, 10))
    draw_regions(ax, definition)
    norm = Normalize(0, .015)
    rgba = plt.colormaps["inferno"](norm(smooth))
    rgba[..., 3] = np.clip(smooth/.003, 0, .96)
    ax.imshow(rgba, extent=(0, 100, 100, 0), interpolation="bilinear", zorder=3)
    cax = fig.add_axes([.788, .285, .140, .017])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, "inferno"), cax=cax,
                      orientation="horizontal", ticks=[0, .015])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=13, length=3)
    label(fig, .857, .217, "Point mass / pixel", size=16, color=GRAY)
    label(fig, .857, .137, f"Mean count  {row['expected_count']:.2f}", size=19)
    VALUES["annotation_configurations"] = {
        "definition": definition, "saved_output": row,
        "display_sigma_pixels": 3, "display_range": [0, .015],
        "point_mass": mass.tolist(), "display_mass": smooth.tolist(),
        "meaning": "Two observed whole configurations; the mean map is not a three-point candidate.",
    }
    save(fig, "annotation_configurations")


def worker_patterns():
    path = REPORT / "results/C_switching/current/worker_parameters.csv"
    with register(path).open(newline="") as f:
        workers = list(csv.DictReader(f))
    rows = {r["worker_id"]: r for r in workers}
    run = read_json(REPORT / "results/C_switching/current/run.json")
    assert run["status"] == "converged"
    # Show four consecutive saved examples, not fabricated alternating responses.
    views = read_json(REPORT / "synthetic_inputs/C_switching/qc_views.json")["views"][:4]
    left_class = int(np.argmax([float(rows["A1"][f"q_class_{g}"]) for g in (1, 2)]))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    # Compact spacing; preserve the 16:9 canvas, typography and all saved values.
    label(fig, .397, .953, "Repeated answers", size=23)
    label(fig, .397, .910, "Synthetic example", size=16, color=GRAY)
    label(fig, .760, .953, "Group membership", size=20)
    label(fig, .760, .910, "Left-like / Right-like", size=15, color=GRAY)
    label(fig, .925, .940, "Class-consistent\nresponse weight, ρ", size=18, linespacing=1.35)
    xpos = [.147, .275, .403, .531]
    tile_width = .118
    tile_height = tile_width * 16 / 9
    for index, x in enumerate(xpos):
        label(fig, x+tile_width/2, .858, f"View {index+1}", size=16, color=GRAY)
    output_rows = []
    for worker, y, title, n in (("A1", .718, "Stable left", 7),
                               ("B1", .441, "Stable right", 3),
                               ("unstable", .164, "Switching", 1)):
        color = BLUE if worker == "A1" else ORANGE if worker == "B1" else INK
        label(fig, .020, y+.02, title, size=20, color=color, ha="left")
        label(fig, .020, y-.025, f"{n} worker"+("s" if n>1 else ""), size=17, color=GRAY, ha="left")
        answers = []
        for x, view in zip(xpos, views):
            answer = next(a for a in view["answers"] if a["worker_id"] == worker)
            answers.append(answer["points"])
            ax = tile(fig, [x, y-tile_height/2, tile_width, tile_height], frame=True)
            for px in (20, 80):
                ax.scatter([px], [50], s=110, facecolors="none", edgecolors=LIGHT, linewidths=1)
            dots(ax, answer["points"], BLUE if answer["points"][0][0] == 20 else ORANGE, size=140)
        arrow(fig, (.655, y), (.680, y))
        qleft = float(rows[worker][f"q_class_{left_class+1}"])
        rho = float(rows[worker]["rho"])
        ax = fig.add_axes([.690, y-.026, .140, .052])
        ax.barh([0], [qleft], color=BLUE, height=1)
        ax.barh([0], [1-qleft], left=[qleft], color=ORANGE, height=1)
        ax.set_xlim(0, 1); ax.set_ylim(-.5, .5); ax.axis("off")
        label(fig, .760, y-.061, f"{qleft:.2f} / {1-qleft:.2f}", size=16, color=GRAY)
        ax = fig.add_axes([.873, y-.023, .105, .046])
        ax.plot([0, 1], [0, 0], color=LIGHT, linewidth=5, solid_capstyle="round")
        ax.plot([0, rho], [0, 0], color=GRAY, linewidth=5, solid_capstyle="round")
        ax.scatter([rho], [0], s=80, color=INK, zorder=3)
        ax.set_xlim(-.04, 1.04); ax.set_ylim(-1, 1); ax.axis("off")
        label(fig, .925, y-.061, f"{rho:.3f}", size=18)
        output_rows.append({"worker": worker, "represented_workers": n, "answers": answers,
                            "left_like_membership": qleft, "rho": rho})
    # Explicit fixed 0--1 range; the bars are not worker accuracy estimates.
    label(fig, .877, .066, "0", size=14, color=GRAY)
    label(fig, .974, .066, "1", size=14, color=GRAY)
    VALUES["worker_response_patterns"] = {"workers": workers, "shown_rows": output_rows,
        "selected_view_indices": [0, 1, 2, 3], "left_class_index": left_class,
        "group_names": "Fitted classes relabeled by their response preference, for display only.",
        "rho_definition": "Class-consistent response mixture weight; not accuracy."}
    save(fig, "worker_response_patterns")


def real_outputs():
    parent = SAVED / "qc_final"
    inputs_path = register(parent / "qc_inputs.json.gz")
    with gzip.open(inputs_path, "rt") as f:
        inputs = json.load(f)
    fit_path = register(parent / "G2_ospa_pair_fit.npz")
    with np.load(fit_path, allow_pickle=False) as fitted:
        metadata = json.loads(str(fitted["metadata"]))
        assert metadata["status"] == "converged"
        assert metadata["inputs_sha256"] == SOURCE_HASHES[str(inputs_path)]
        q, pi = fitted["q"].copy(), fitted["pi"].copy()
    worker_q = dict(zip(inputs["worker_ids"], q))
    choices = [("QC3.png", "280_280_80_80"), ("QC1.png", "0_350_80_80")]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    starts = [.035, .280, .525, .770]
    width, height = .194, .345
    for x, title in zip(starts, ("Raw image", "Observed points", "Mean point map", "Top-weight candidate")):
        label(fig, x+width/2, .939, title, size=21)
    records = []
    for (image_id, view_id), y in zip(choices, (.533, .112)):
        index, view = next((i,v) for i,v in enumerate(inputs["views"])
                           if (v["image_id"], v["view_id"]) == (image_id, view_id))
        candidates = [c["points"] for c in view["candidates"]]
        mixing = np.mean([worker_q[a["worker_id"]] for a in view["answers"]], axis=0)
        weights = mixing @ pi[index, :, :len(candidates)]
        assert np.isclose(weights.sum(), 1)
        best = int(np.argmax(weights))
        from data import DEFAULT_DATA_ROOT
        image_path = register(DEFAULT_DATA_ROOT / image_id / image_id)
        with Image.open(image_path) as im:
            x0, y0 = view["origin_x"], view["origin_y"]
            raw = np.asarray(im.convert("L").crop((x0, y0, x0+80, y0+80)))
        low, high = np.percentile(raw, [1, 99.5])
        norm = PowerNorm(.72, vmin=low, vmax=high)
        axes = [tile(fig, [x, y, width, height], bounds=(0,80,80,0)) for x in starts]
        for ax in axes:
            ax.imshow(raw, cmap="gray", norm=norm, extent=(0,80,80,0), interpolation="nearest")
        # Small, outlined marks remain visible over bright objects and dark background.
        clicks = np.asarray([p for a in view["answers"] for p in a["points"]]).reshape(-1, 2)
        axes[1].scatter(clicks[:,0], clicks[:,1], s=15, facecolors=HOT,
                         edgecolors="#381D19", linewidths=.38, alpha=.63, zorder=5)
        mass = point_mass(candidates, weights, 80)
        smooth = gaussian_filter(mass, 1.25, mode="reflect")
        heat_norm = Normalize(0, .05)
        rgba = plt.colormaps["inferno"](heat_norm(smooth))
        rgba[..., 3] = np.clip(smooth/.018, 0, .94)
        axes[2].imshow(rgba, extent=(0,80,80,0), interpolation="bilinear", zorder=4)
        axes[2].contour(np.arange(80)+.5, np.arange(80)+.5, smooth,
                         levels=[.014], colors=["#A53336"], linewidths=.75, zorder=5)
        dots(axes[3], candidates[best], HOT, size=42, real=True)
        label(fig, starts[1]+width/2, y-.027, f"{len(view['answers'])} answers", size=17, color=GRAY)
        label(fig, starts[3]+width/2, y-.027, f"Weight {weights[best]:.3f} · {len(candidates[best])} points", size=17, color=GRAY)
        for left, right in ((0,1),(1,2)):
            arrow(fig, (starts[left]+width+.009, y+height/2), (starts[right]-.01, y+height/2))
        records.append({"source_image": str(image_path), "view": view,
                         "weights": weights.tolist(), "top_candidate_index": best,
                         "group_mix": mixing.tolist(), "point_mass": mass.tolist(),
                         "display_mass": smooth.tolist(),
                         "grayscale_limits": [float(low), float(high)], "grayscale_gamma": .72})
    cax = fig.add_axes([.553, .052, .139, .012])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0,.05), "inferno"), cax=cax,
                      orientation="horizontal", ticks=[0, .025, .05])
    cb.ax.tick_params(labelsize=12, length=2)
    cb.outline.set_visible(False)
    label(fig, .623, .084, "Point mass / pixel", size=15, color=GRAY)
    VALUES["real_annotation_outputs"] = {"records": records,
        "display_sigma_pixels": 1.25, "shared_display_range": [0,.05],
        "alpha_rule": "clip(display_mass/0.018, 0, 0.94)",
        "contour_level": .014,
        "selection": "Two previously used examples: ambiguous and dense. No selection by model superiority.",
        "candidate_display": "The highest-weight whole observed candidate, not a thresholded map."}
    save(fig, "real_annotation_outputs")


def main():
    configurations()
    worker_patterns()
    real_outputs()
    for path, expected in SOURCE_HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    (OUT / "source_values.json").write_text(json.dumps(VALUES, indent=2, ensure_ascii=False)+"\n")
    manifest = {
        "figures": {
            "annotation_configurations": "A saved two-point versus one-point example links input geometry, observed whole answers, candidate weights and the mean point map.",
            "worker_response_patterns": "Four consecutive saved response examples link stable and switching patterns to fitted group membership and class-consistent response weight.",
            "real_annotation_outputs": "Two real image regions show the image, all observed points, the mean point map and the highest-weight whole candidate.",
        },
        "formats": ["PNG", "SVG"], "dimensions": [1920,1080], "aspect_ratio": "16:9",
        "font": "Liberation Sans", "svg_text": "editable", "figure_language": "English",
        "source_sha256": SOURCE_HASHES,
        "processing": "Read saved inputs and converged fits; calculate weighted point masses; smooth maps for display; draw figures. No fitting or new inference was run.",
        "display_smoothing": "scipy.ndimage.gaussian_filter, reflect boundary. Synthetic: sigma 3 pixels, fixed range 0--0.015. Real: sigma 1.25 pixels, shared range 0--0.05. Unsmoothed masses and displayed arrays are both retained in source_values.json. No per-map maximum normalization.",
        "interpretation": "Synthetic geometry is illustrative and was not used for fitting. Candidate weights describe annotation configurations, not biological truth. The mean map does not define a simultaneous multi-point configuration. The real examples are fitted-data illustrations, not held-out accuracy results.",
        "unchanged_source_hash_check": True, "checks": CHECKS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"output": str(OUT), "figures": list(manifest["figures"]), "checks": CHECKS}, indent=2))


if __name__ == "__main__":
    main()
