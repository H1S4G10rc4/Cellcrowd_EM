"""Shared locations for command-line scripts; no model settings are changed."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "scripts/experiments"
FIGURE_SOURCE_ROOT = PROJECT_ROOT / "scripts/figures"
TEST_ROOT = PROJECT_ROOT / "tests"
REPORT_ROOT = WORKSPACE_ROOT / "comparison_report"
REPORT_SCRIPT_ROOT = REPORT_ROOT / "scripts"
METHOD_OUTPUT_ROOT = PROJECT_ROOT / "outputs/method_schematics"
SLIDE_OUTPUT_ROOT = PROJECT_ROOT / "outputs/slide_figures"
FONT_ROOT = Path(os.environ.get("ANNOTATION_FONT_DIR", Path.home() / ".cache/annotation-figure-fonts")).expanduser()

# Keep the existing import names while separating sources from saved outputs.
for directory in reversed((SOURCE_ROOT, SOURCE_ROOT / "historical", EXPERIMENT_ROOT,
                           PROJECT_ROOT / "scripts/history", FIGURE_SOURCE_ROOT,
                           TEST_ROOT, REPORT_SCRIPT_ROOT)):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

SOURCE_LOCATIONS = {
    "data.py": "point_annotation_trial/src/data.py",
    "configuration_model.py": "point_annotation_trial/src/configuration_model.py",
    "local_regions.py": "point_annotation_trial/src/local_regions.py",
    "model.py": "point_annotation_trial/src/historical/model.py",
    "run_configuration_pilot.py": "point_annotation_trial/scripts/experiments/run_configuration_pilot.py",
    "validate_configuration_pilot.py": "point_annotation_trial/scripts/experiments/validate_configuration_pilot.py",
    "audit_configuration_results.py": "point_annotation_trial/scripts/experiments/audit_configuration_results.py",
    "run.py": "point_annotation_trial/scripts/history/run.py",
    "test_configuration_model.py": "point_annotation_trial/tests/test_configuration_model.py",
    "test_model.py": "point_annotation_trial/tests/test_model.py",
    "make_configuration_figures.py": "point_annotation_trial/scripts/figures/make_configuration_figures.py",
    "make_compact_configuration_figures.py": "point_annotation_trial/scripts/figures/make_compact_configuration_figures.py",
    "make_estimation_split.py": "point_annotation_trial/scripts/figures/make_estimation_split.py",
    "make_estimation_overview.py": "point_annotation_trial/scripts/figures/make_estimation_overview.py",
    "make_annotation_flow.py": "point_annotation_trial/scripts/figures/make_annotation_flow.py",
    "make_model_concept.py": "point_annotation_trial/scripts/figures/make_model_concept.py",
    "make_nature_png.py": "point_annotation_trial/scripts/figures/make_nature_png.py",
    "make_inference_schematics.py": "point_annotation_trial/scripts/figures/make_inference_schematics.py",
    "make_generative_model.py": "point_annotation_trial/scripts/figures/make_generative_model.py",
    "make_method_schematics.py": "point_annotation_trial/scripts/figures/make_method_schematics.py",
    "make_slide_figures.py": "point_annotation_trial/scripts/figures/make_slide_figures.py",
    "make_synthetic_comparison_figures.py": "point_annotation_trial/scripts/figures/make_synthetic_comparison_figures.py",
    "build_manuscript_sources.py": "comparison_report/scripts/build_manuscript_sources.py",
    "build_report_assets.py": "comparison_report/scripts/build_report_assets.py",
    "make_connected_cell_case.py": "comparison_report/scripts/make_connected_cell_case.py",
    "make_report_figures.py": "comparison_report/scripts/make_report_figures.py",
    "run_comparison.py": "comparison_report/scripts/run_comparison.py",
    "run_connected_split_case.py": "comparison_report/scripts/run_connected_split_case.py",
    "run_pairing_demo.py": "comparison_report/scripts/run_pairing_demo.py",
    "README.md": "point_annotation_trial/README.md"
}


def source_path(name):
    """Resolve a source basename used by snapshot and audit routines."""
    return WORKSPACE_ROOT / SOURCE_LOCATIONS[name]


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _relocation():
    manifest_path = PROJECT_ROOT / "docs/source_layout.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Historical relocation audits require private provenance files, "
            "which are intentionally absent from the public distribution. "
            "New runs record checksums of the bundled sources directly."
        )
    manifest = json.loads(manifest_path.read_text())
    archive = WORKSPACE_ROOT / manifest["original_source_archive"]
    if _sha(archive) != manifest["original_source_archive_sha256"]:
        raise ValueError("The original-source archive no longer matches its checksum")
    return manifest, archive


def historical_sha256(path):
    """Verify old provenance against original bytes, not rewritten path strings.

    For a moved source, also verify that its active copy still matches the
    recorded post-move version. Unmoved inputs/results are hashed directly.
    This keeps historical manifests immutable and makes the relocation explicit.
    """
    path = Path(path).resolve()
    try:
        relative = str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return _sha(path)
    manifest, archive = _relocation()
    entry = next((item for item in manifest["files"] if item["from"] == relative), None)
    if entry is None:
        return _sha(path)
    current = WORKSPACE_ROOT / entry["to"]
    if _sha(current) != entry["sha256_after"]:
        raise ValueError(f"Moved source changed after reorganization: {current}")
    with zipfile.ZipFile(archive) as saved:
        original = hashlib.sha256(saved.read(relative)).hexdigest()
    if original != entry["sha256_before"]:
        raise ValueError(f"Original source checksum mismatch: {relative}")
    return original
