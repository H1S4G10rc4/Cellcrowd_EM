PYTHON ?= python3

.PHONY: check demo publication-check install-hooks

check:
	PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg $(PYTHON) -B -m unittest discover -s point_annotation_trial/tests -p 'test_*.py' -v
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B -m unittest discover -s tools -p 'test_*.py' -v

demo:
	PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg $(PYTHON) -B point_annotation_trial/scripts/experiments/run_configuration_pilot.py synthetic --skip-legacy --output outputs/synthetic_demo

publication-check:
	$(PYTHON) tools/check_publication.py --history

install-hooks:
	git config --local core.hooksPath .githooks
