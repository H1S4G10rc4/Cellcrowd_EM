# Point-configuration inference

Research code for combining multiple point annotations while preserving complete
point configurations. The current model estimates annotator-specific count-error
tables and posterior weights for candidate configurations. It does not use
reference labels or image intensities during inference.

This is a **source-only publication**. Participant data, identifiers, fitted
participant parameters, microscopy images, saved outputs, notebooks and archives
are intentionally excluded. The previous Git history is not part of this release.

## Quick start

Python 3.11 or newer is required. Run these commands from the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make check
make demo
```

The demo generates two small artificial conditions with fixed seeds: a tendency
to merge two points and a tendency to split one point. It fits the current model
and applies the fitted tables to separate generated responses. Generated reference
configurations are used only for evaluation, never for fitting. Outputs are saved
locally under `outputs/synthetic_demo/`, which is ignored by Git. The runner refuses
to overwrite an existing experiment; use a different output directory for reruns.

To include the Legacy Algorithm on the same synthetic inputs:

```sh
python -m pip install -r requirements-legacy.txt
python point_annotation_trial/scripts/experiments/run_configuration_pilot.py \
  synthetic --output outputs/synthetic_comparison
```

These are synthetic implementation checks, not evidence of biological accuracy.

## Layout

| Directory | Contents |
| --- | --- |
| `point_annotation_trial/src/` | Current model, region construction and input handling |
| `point_annotation_trial/scripts/experiments/` | Pilot, validation and historical audit entry points |
| `point_annotation_trial/scripts/figures/` | Current and historical figure-generation sources |
| `point_annotation_trial/tests/` | Numerical and source-layout tests |
| `point_annotation_trial/docs/` | Concise English and Japanese mathematical Methods sources |
| `point_annotation_trial/src/historical/` | Superseded response-class model; not the current method |
| `comparison_report/scripts/` | Historical comparison and report helpers |
| `legacy_algorithm/` | Legacy Algorithm source snapshot and checksums |
| `tools/` | Publication checks and their tests |

The current model fits a 3-by-3 count-error table per annotator, with count bins
0, 1 and 2+. It alternates posterior candidate weighting and regularized table
updates. Region definitions, candidate locations and spatial-kernel settings remain
fixed. Outputs include configuration weights, count distributions and expected
point mass. An expected mass map is not itself a sampled point configuration.

See [the Methods source](point_annotation_trial/docs/methods.tex) for assumptions,
likelihoods, the objective, convergence and limitations. Posterior support depends
on the proposed candidates and model assumptions; it is not an independently
verified probability of biological correctness.

## Local data and optional tools

Set `POINT_ANNOTATION_DATA_ROOT` to a private annotation directory when using your
own data. The default is the ignored `private_data/by_image_tag/` directory.
Do not upload inputs or derived participant tables. Hashing a name does not make
the resulting record anonymous. See [PRIVACY.md](PRIVACY.md).

Real-data validation, historical audits and most figure scripts require saved
inputs/results that are not distributed. Their presence here preserves source
code, not an assertion that all historical reports can be reproduced from a clean
clone. Older comparison scripts use the superseded class model; use the current
pilot above for comparisons with the current configuration model.

The Methods document can be built locally with XeLaTeX:

```sh
make -C point_annotation_trial methods
```

Japanese typesetting additionally requires FontTools and a suitable Noto CJK font
(`JP_FONT_SOURCE` can be overridden). Figure export may require Matplotlib,
XeLaTeX, Poppler, STIX Two Math and separately licensed Arial fonts. Set
`ANNOTATION_FONT_DIR` to the Arial directory. No font binaries or compiled documents
are distributed. Optional Python packages are listed in `requirements-docs.txt`.

## Before publishing changes

```sh
make install-hooks
make publication-check
```

Only files explicitly listed in `PUBLIC_FILES.txt` may be committed. Adding a new
public source requires reviewing it and updating both this list and `.gitignore`.
Local hooks and CI check the Git contents, not only the working directory.
These safeguards reduce accidental disclosure; they do not replace manual review.

Licensed under the [MIT License](LICENSE). Original attribution is retained.
