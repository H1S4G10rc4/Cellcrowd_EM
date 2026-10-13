"""Create a static OpenType font for the Japanese Methods PDF build."""

from pathlib import Path
import sys

from fontTools.cffLib.CFF2ToCFF import convertCFF2ToCFF
from fontTools.subset import Options, Subsetter
from fontTools.ttLib import TTCollection
from fontTools.varLib.instancer import instantiateVariableFont


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: build_japanese_font.py input.ttc output.otf document.tex")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    document = Path(sys.argv[3])
    collection = TTCollection(source)
    font = collection.fonts[0]
    subsetter = Subsetter(options=Options())
    subsetter.populate(
        unicodes={ord(c) for c in document.read_text(encoding="utf-8")}
        | set(range(0x20, 0x7F))
    )
    subsetter.subset(font)
    font = instantiateVariableFont(font, {"wght": 400}, inplace=True)
    convertCFF2ToCFF(font)
    font.recalcBBoxes = False
    output.parent.mkdir(parents=True, exist_ok=True)
    font.save(output)


if __name__ == "__main__":
    main()
