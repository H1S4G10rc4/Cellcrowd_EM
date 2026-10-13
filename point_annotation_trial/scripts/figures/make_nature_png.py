"""Re-render the six saved schematics: black Arial, XeLaTeX math, PNG only.

Run with the existing plotting environment. No estimation is performed.
The earlier PNG/SVG files and all fitted results remain unchanged.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import contextlib
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from PIL import Image

import make_method_schematics as art
import make_generative_model as generative
import make_inference_schematics as inference

HERE = METHOD_OUTPUT_ROOT
OUT = HERE / "nature_png"
from project_paths import FONT_ROOT
FONT_DIR = FONT_ROOT
MATH_FONT = Path("/usr/share/fonts/stix-fonts/STIXTwoMath-Regular.otf")
DPI = 300
SIZE = (4800, 2700)
CHECKS = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    for filename in ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF", "Arialbi.TTF"):
        path = FONT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Actual Arial font required: {path}")
        font_manager.fontManager.addfont(path)
    assert font_manager.FontProperties(fname=FONT_DIR / "Arial.TTF").get_name() == "Arial"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": "#505050", "axes.linewidth": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "mathtext.fontset": "stix", "mathtext.default": "it",
        "pgf.texsystem": "xelatex", "pgf.rcfonts": False,
        "pgf.preamble": (
            r"\usepackage{fontspec}" "\n"
            rf"\setmainfont{{Arial.TTF}}[Path={FONT_DIR}/,"
            r"BoldFont=Arialbd.TTF,ItalicFont=Ariali.TTF,BoldItalicFont=Arialbi.TTF]" "\n"
            rf"\setsansfont{{Arial.TTF}}[Path={FONT_DIR}/,"
            r"BoldFont=Arialbd.TTF,ItalicFont=Ariali.TTF,BoldItalicFont=Arialbi.TTF]" "\n"
            r"\usepackage{unicode-math}" "\n"
            rf"\setmathfont{{{MATH_FONT.name}}}[Path={MATH_FONT.parent}/]" "\n"
            r"\renewcommand{\familydefault}{\sfdefault}"
        ),
    })
    assert Path(font_manager.findfont("Arial")).resolve() == (FONT_DIR / "Arial.TTF").resolve()


def typeset_label(value):
    # Keep the model unchanged; write an arithmetic mean as an explicit sum.
    value = value.replace(
        r"$v_{rg}=\mathrm{mean}_{j\in J_r}\;q_{jg}$",
        r"$\displaystyle v_{rg}=\frac{1}{|J_r|}\sum_{j\in J_r}q_{jg}$",
    )
    value = value.replace("ρ", r"$\rho_j$")
    value = value.replace("Class-consistent", "Group-consistent")
    value = value.replace("Point mass / pixel", "Point mass (per pixel)")
    value = value.replace("Count × point positions", r"Count $\times$ point positions")
    if value.startswith("$p(Y_{rj}"):
        value = value.replace("$p(", r"$\displaystyle p(", 1)
    if value.startswith("$w_{rk}"):
        value = value.replace("$w", r"$\displaystyle w", 1)
    return value


def save_png(fig, stem):
    fig.canvas.draw()  # Instantiate all tick labels before applying the style.
    for text in fig.findobj(mpl.text.Text):
        old_color = mpl.colors.to_rgba(text.get_color())
        text.set_color("black")
        text.set_fontfamily("sans-serif")
        text.set_text(typeset_label(text.get_text()))
        # Put black numeric labels below the bar, not on a coloured background.
        if old_color[:3] == (1., 1., 1.) and text.get_text():
            x, y = fig.transFigure.inverted().transform(text.get_transform().transform(text.get_position()))
            text.set_transform(fig.transFigure)
            text.set_position((x, y - .05))
        if stem == "inference_to_outputs":
            if text.get_text().startswith(r"$\displaystyle w_{rk}"):
                text.set_fontsize(21)
                text.set_position((.547, .870))
            elif text.get_text().startswith(r"$\displaystyle v_{rg}"):
                text.set_fontsize(21)
                text.set_position((.201, .144))
            elif text.get_text() == "From participating workers":
                text.set_position((.201, .073))
        if stem == "model_inference" and text.get_text() == "Orange crosses: observed points":
            text.set_text("Observed points")
            text.set_position((.519, .765))
            fig.add_artist(plt.Line2D([.442], [.765], transform=fig.transFigure,
                                     marker="+", markersize=9, markeredgewidth=1.7,
                                     linestyle="none", color=art.HOT))
    # XeLaTeX also measures the text, so the geometry checks use the final font.
    FigureCanvasPgf(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    escaped, texts = [], []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        value = text.get_text()
        assert not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value), value
        assert "QC" not in value
        assert mpl.colors.to_rgba(text.get_color())[:3] == (0., 0., 0.)
        box = text.get_window_extent(renderer)
        if box.width and box.height and not (
            box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
        ):
            escaped.append(value)
        texts.append({"text": value, "size_pt_on_16_inch_canvas": text.get_fontsize()})
    assert not escaped, (stem, escaped)
    # Check the positioned explanatory labels, including labels moved off bars.
    labels = list(fig.texts) + [t for ax in fig.axes for t in ax.texts]
    overlaps = []
    for i, first in enumerate(labels):
        a = first.get_window_extent(renderer)
        for second in labels[i + 1:]:
            b = second.get_window_extent(renderer)
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                overlaps.append((first.get_text(), second.get_text()))
    assert not overlaps, (stem, overlaps)

    # Vector/TeX intermediates stay in a private temporary directory, not output.
    with tempfile.TemporaryDirectory(prefix="schematic-tex-") as work:
        pdf = Path(work) / "figure.pdf"
        fig.savefig(pdf, backend="pgf", dpi=DPI, facecolor="white")
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        assert "Arial" in fonts, fonts
        if any("$" in x["text"] for x in texts):
            assert "STIXTwoMath" in fonts, fonts
        assert "Liberation" not in fonts and "DejaVu" not in fonts, fonts
        prefix = Path(work) / "render"
        subprocess.run(["pdftoppm", "-singlefile", "-r", str(DPI),
                        "-png", str(pdf), str(prefix)], check=True, capture_output=True)
        with Image.open(prefix.with_suffix(".png")) as rendered:
            assert rendered.size == SIZE, (stem, rendered.size)
            rendered.convert("RGB").save(OUT / f"{stem}.png", dpi=(DPI, DPI))
    CHECKS.append({"figure": stem, "size": SIZE, "mode": "RGB",
                   "all_text_black": True, "text_inside_canvas": True,
                   "positioned_text_nonoverlapping": True,
                   "english_only": True, "embedded_intermediate_fonts": fonts,
                   "text": texts})
    plt.close(fig)
    print(f"Rendered {stem}.png", flush=True)


def main(worker_patterns_only=False):
    OUT.mkdir(exist_ok=True)
    configure()
    art.save = save_png
    if worker_patterns_only:
        # Refresh only the requested figure and its existing provenance entry.
        others = {str(p): digest(p) for p in OUT.glob("*.png")
                  if p.name != "worker_response_patterns.png"}
        original_values = json.loads((HERE / "source_values.json").read_text())
        art.worker_patterns()
        assert art.VALUES["worker_response_patterns"] == original_values["worker_response_patterns"]
        for path, expected in {**others, **art.SOURCE_HASHES}.items():
            assert digest(path) == expected, path
        manifest_path = OUT / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["checks"] = [CHECKS[0] if c["figure"] == "worker_response_patterns" else c
                              for c in manifest["checks"]]
        manifest["source_sha256"].update(art.SOURCE_HASHES)
        manifest.setdefault("layout_revisions", {})["worker_response_patterns"] = {
            "change": "Reduced outer, inter-row and inter-column whitespace; enlarged point-set frames without shrinking text.",
            "point_frame_width_before_after": [.090, .118],
            "horizontal_frame_gap_before_after": [.015, .010],
            "vertical_frame_gap_before_after": [.092, .277 - .118 * 16 / 9],
            "data_unchanged": True, "other_figures_unchanged": True,
            "font_sizes_unchanged": True, "canvas_unchanged": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Verified unchanged source values and all other figures.", flush=True)
        return
    art.configurations()
    art.worker_patterns()
    art.real_outputs()
    # The original entrypoint records useful model provenance. Keep its temporary
    # manifest out of the final directory because it names the old export format.
    with tempfile.TemporaryDirectory(prefix="schematic-provenance-") as work:
        generative.OUT = Path(work)
        with contextlib.redirect_stdout(io.StringIO()):
            generative.main()
        gen_metadata = json.loads((Path(work) / "generative_model_manifest.json").read_text())
    definition, run, training, application, fitted, output = inference.input_data()
    inference.estimation(definition, run, training, fitted)
    inference.outputs(definition, run, application, fitted, output)

    hashes = {**art.SOURCE_HASHES, **gen_metadata["source_sha256"], **inference.HASHES}
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    descriptions = {
        "annotation_configurations": "Observed whole point sets, candidate weights and the mean point map.",
        "worker_response_patterns": "Repeated stable and switching answers, inferred group membership and response weight.",
        "real_annotation_outputs": "Two real image regions: original image, observed points, mean map and top-weight candidate.",
        "generative_model": "Conditional generation of answers through group membership, candidate choice and count/location variation.",
        "model_inference": "Alternating hidden-assignment and parameter updates, with a saved converged fit.",
        "inference_to_outputs": "Group-specific candidate weights combined into a count distribution and a mean point map.",
    }
    manifest = {
        "files": {f"{k}.png": v for k, v in descriptions.items()},
        "style": {"size": SIZE, "aspect_ratio": "16:9", "dpi": DPI, "mode": "RGB",
                  "text_font": "Arial", "text_color": "#000000", "language": "English",
                  "math_engine": "XeLaTeX via Matplotlib PGF", "math_font": "STIX Two Math",
                  "font_source": "https://downloads.sourceforge.net/project/corefonts/the%20fonts/final/arial32.exe",
                  "font_files_sha256": {p.name: digest(p) for p in FONT_DIR.glob("*.TTF")}},
        "nature_guidance": {
            "accessed": "2026-10-10",
            "sources": ["https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/",
                        "https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/"],
            "applied": "Standard Arial labels, black high-contrast text, white background, consistent thin rules, aligned layouts, RGB and non-rainbow maps.",
            "slide_adaptations": "Larger slide-readable text on a 16:9 canvas; no panel letters, overall titles or footnotes. STIX Two Math is used for LaTeX equations. Print-specific 5-7 pt sizing and 89/183 mm widths are not imposed on slides.",
            "format_limit": "PNG-only slide deliverables as requested. Nature final main figures require editable vector/layered artwork; these PNGs are not represented as submission-ready files.",
        },
        "processing": "Style-only re-rendering and the same saved-data aggregates used by the earlier figures. No fitting or new image inference. Existing scripts, figures, source data and fitted values preserved.",
        "presentation_changes": "Class-consistent changed to Group-consistent for consistent naming; the definition is unchanged. The participating-worker mean is written explicitly as a sum. All rho glyphs are LaTeX variables. Unit-like labels use parentheses.",
        "data_notes": "Point coordinates, candidate combinations, numeric values, map scales, smoothing and image intensity transforms are unchanged. Real image samples are not sharpened or upsampled as new evidence; only the full figure canvas is rendered at higher resolution.",
        "source_values": ["../source_values.json", "../generative_model_manifest.json", "../inference_source_values.json"],
        "source_sha256": hashes, "sources_unchanged": True, "checks": CHECKS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert len(list(OUT.glob("*.png"))) == 6
    assert not list(OUT.glob("*.pdf")) and not list(OUT.glob("*.svg"))
    print(f"Verified six PNGs; source values unchanged. Output: {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-patterns-only", action="store_true")
    args = parser.parse_args()
    main(worker_patterns_only=args.worker_patterns_only)
