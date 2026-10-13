PYTHON ?= python3
LATEX ?= xelatex
LATEX_JA ?= xelatex
FONT_PYTHON ?= $(PYTHON)
JP_FONT_SOURCE ?= /usr/share/fonts/google-noto-serif-cjk-vf-fonts/NotoSerifCJK-VF.ttc

.PHONY: check methods methods-ja

check:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v

methods:
	mkdir -p .cache/latex
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=.cache/latex docs/methods.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -output-directory=.cache/latex docs/methods.tex
	cp .cache/latex/methods.pdf docs/methods.pdf

methods-ja:
	mkdir -p .cache/latex-ja
	$(FONT_PYTHON) scripts/build_japanese_font.py $(JP_FONT_SOURCE) .cache/latex-ja/NotoSerifCJKJP-Regular.otf docs/methods_ja.tex
	$(LATEX_JA) -interaction=nonstopmode -halt-on-error -output-directory=.cache/latex-ja docs/methods_ja.tex
	$(LATEX_JA) -interaction=nonstopmode -halt-on-error -output-directory=.cache/latex-ja docs/methods_ja.tex
	cp .cache/latex-ja/methods_ja.pdf docs/methods_ja.pdf
