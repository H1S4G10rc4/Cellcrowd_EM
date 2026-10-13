"""Check source locations without fitting models or redrawing stored results."""
from pathlib import Path
import os
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_paths as paths


class SourceLayoutTests(unittest.TestCase):
    def test_source_index_resolves(self):
        for name in paths.SOURCE_LOCATIONS:
            with self.subTest(source=name):
                self.assertTrue(paths.source_path(name).is_file())

    def test_current_model_is_separate_from_historical_model(self):
        import configuration_model
        import model
        self.assertEqual(Path(configuration_model.__file__).parent, paths.SOURCE_ROOT)
        self.assertEqual(Path(model.__file__).parent, paths.SOURCE_ROOT / "historical")

    def test_private_inputs_are_not_required_for_imports(self):
        import data
        import run_configuration_pilot as pilot
        import audit_configuration_results as audit
        self.assertIsInstance(data.DEFAULT_DATA_ROOT, Path)
        self.assertEqual(pilot.HERE, paths.PROJECT_ROOT)
        self.assertEqual(audit.ROOT, paths.PROJECT_ROOT / "outputs/configuration_consensus_v2")
        self.assertEqual(paths.METHOD_OUTPUT_ROOT.parent, paths.PROJECT_ROOT / "outputs")
        self.assertEqual(paths.SLIDE_OUTPUT_ROOT.parent, paths.PROJECT_ROOT / "outputs")

    def test_command_help_outside_project(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
        for name in ("run_configuration_pilot.py", "validate_configuration_pilot.py",
                     "make_configuration_figures.py", "make_compact_configuration_figures.py"):
            with self.subTest(command=name):
                result = subprocess.run([sys.executable, "-B", str(paths.source_path(name)), "--help"],
                                        cwd=paths.WORKSPACE_ROOT, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
