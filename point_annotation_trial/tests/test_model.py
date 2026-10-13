"""Six existing checks and two save/log checks, without statistical retuning."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy.integrate import simpson

from data import (Answer, QC_IDS, View, baseline_weights, expected_count, expected_mass,
                  load_image_views, sample_point_sets, select_qc_workers)
from model import (fit, full_objective, load_fit_npz, log_count_probability,
                   log_f_for_answer, log_f_matrix, prepare, predictive_from_log_f,
                   predictive_log_density, save_fit_npz, save_qc_inputs, spatial_density)


def _view(image_id: str, view_id: str, answers: tuple[Answer, ...], size: int = 100) -> View:
    return View(image_id, view_id, 0, 0, size, size, size, size, answers)


def _fit_two(prepared):
    attempts = [fit(prepared, 2, name) for name in ("ospa_pair", "random_20261008")]
    usable = [candidate for candidate in attempts if candidate.converged]
    if not usable:
        usable = [candidate for candidate in attempts if candidate.status != "numerical_failure"]
    selected = max(usable, key=lambda item: item.objective) if usable else None
    return selected, attempts


def _attempts(attempts) -> list[dict]:
    return [{"initialization": a.initialization, "status": a.status, "iterations": len(a.history),
             "objective": a.objective, "runtime_seconds": a.runtime_seconds,
             "representative_workers": a.representative_workers} for a in attempts]


def check_input_candidates() -> dict:
    points = ([], [], [(10, 20)], [(10, 20)], [(11, 20)])
    view = _view("synthetic.png", "0", tuple(Answer(w, np.array(p, dtype=float).reshape(-1, 2))
                                           for w, p in zip("ABCDE", points)))
    ordered = _view("synthetic.png", "1", (
        Answer("A", np.array([[10., 20.], [30., 40.]])),
        Answer("B", np.array([[30., 40.], [10., 20.]])),
        Answer("C", np.array([[10., 20.], [10., 20.]])),
    ))
    assert len(view.answers) == 5 and len(view.candidates) == 3
    assert [s.support_count for s in view.candidates] == [2, 2, 1]
    assert len(ordered.candidates) == 2 and ordered.candidates[0].support_count == 2
    assert len(ordered.candidates[1].points) == 2
    return {"passed": True, "raw_answers": 5, "candidate_count": 3,
            "support_counts": [2, 2, 1], "point_order_ignored": True, "repeated_clicks_retained": True}


def check_observation_density() -> dict:
    count_sums = {str(m): float(np.exp(log_count_probability(m, np.arange(m + 81))).sum()) for m in (0, 1, 5)}
    assert all(abs(value - 1) < 1e-12 for value in count_sums.values())
    axis = np.linspace(0, 1, 401)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    integrals = {}
    for label, point in (("corner", (0., 0.)), ("center", (.5, .5))):
        f = spatial_density(xy, np.array([point])).reshape(len(axis), len(axis))
        kernel = (f - 0.01) / 0.99
        integral = float(simpson(simpson(kernel, x=axis, axis=1), x=axis))
        integrals[label] = integral
        assert abs(integral - 1) < 1e-5
    view = _view("synthetic.png", "0", (
        Answer("A", np.empty((0, 2))), Answer("B", np.array([[30., 50.]])),
    ))
    before = tuple(c.points.copy() for c in view.candidates)
    arbitrary = Answer("unseen", np.array([[10., 20.], [70., 80.], [30., 40.]]))
    log_values = log_f_for_answer(view, arbitrary)
    assert np.isfinite(log_values).all()
    assert len(view.candidates) == len(before) and all(np.array_equal(a.points, p) for a, p in zip(view.candidates, before))
    return {"passed": True, "count_probability_sums": count_sums,
            "truncated_kernel_integrals": integrals,
            "unobserved_three_point_log_f": log_values.tolist(), "candidates_unchanged": True}


def _write_fixture(root: Path, non_qc_point: float) -> None:
    for image_id in (*QC_IDS, "non_qc.png"):
        folder = root / image_id
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("L", (100, 100)).save(folder / image_id)
        for worker, location in (("a", 20.), ("b", 80.), ("c", None)):
            location = non_qc_point if image_id == "non_qc.png" and worker == "a" else location
            point_list = [] if location is None else [{"x": location, "y": 50.}]
            rows = [{"crop_coordinates": {"x": 0, "y": 0, "width": 100, "height": 100},
                     "cell_coordinates": {"points": point_list}, "cell_count": len(point_list)}]
            (folder / f"{image_id}_{worker}.json").write_text(json.dumps(rows), encoding="utf-8")


def check_estimation_qc_scope() -> dict:
    with tempfile.TemporaryDirectory(prefix="point-annotation-trial-check-") as temporary:
        root = Path(temporary)
        _write_fixture(root, 5.)
        workers, _ = select_qc_workers(root)
        views = tuple(view for name in QC_IDS for view in load_image_views(root, name, workers, require_complete=True))
        first = fit(prepare(views, workers), 1, "one")
        _write_fixture(root, 95.)
        repeated = tuple(view for name in QC_IDS for view in load_image_views(root, name, workers, require_complete=True))
        second = fit(prepare(repeated, workers), 1, "one")
        np.testing.assert_array_equal(first.parameters.rho, second.parameters.rho)
        np.testing.assert_array_equal(first.parameters.q, second.parameters.q)
        np.testing.assert_array_equal(first.parameters.pi, second.parameters.pi)
        accepted = [row for row in first.history if row["accepted"]]
        objective_path = [row["objective"] for row in accepted]
        assert first.status == "converged" and second.status == "converged"
        assert all(b >= a - 1e-8 * max(1, abs(a)) for a, b in zip(objective_path, objective_path[1:]))
        return {"passed": True, "objective_iterations": len(objective_path),
                "minimum_objective_step": float(min(np.diff(objective_path))) if len(objective_path) > 1 else 0.,
                "non_qc_change_altered_worker_parameters": False,
                "non_qc_observations_in_fit": 0}


def _stable_views(with_unstable: bool) -> tuple[tuple[View, ...], tuple[str, ...], tuple[str, ...]]:
    group_a = tuple(f"A{i}" for i in range(1, 8))
    group_b = tuple(f"B{i}" for i in range(1, 4))
    shuffled = np.array([0] * 25 + [1] * 25)
    np.random.default_rng(20261008).shuffle(shuffled)
    a, b = np.array([[20., 50.]]), np.array([[80., 50.]])
    views = []
    for index in range(50):
        answers = [Answer(w, a.copy()) for w in group_a] + [Answer(w, b.copy()) for w in group_b]
        if with_unstable:
            answers.append(Answer("unstable", a.copy() if shuffled[index] == 0 else b.copy()))
        views.append(_view(QC_IDS[index // 10], str(index % 10), tuple(answers)))
    return tuple(views), group_a, group_b


def check_stable_70_30() -> dict:
    views, group_a, group_b = _stable_views(False)
    workers = group_a + group_b
    fitted, attempts = _fit_two(prepare(views, workers))
    if fitted is None:
        return {"passed": False, "reason": "both_initializations_failed", "attempts": _attempts(attempts)}
    params = fitted.parameters
    weights = np.stack([params.q.mean(axis=0) @ params.pi[r, :, :2] for r in range(len(views))])
    median_a = float(np.median(params.rho[:7]))
    median_b = float(np.median(params.rho[7:10]))
    scores = {"group_A_on_A": [], "group_A_on_B": [], "group_B_on_A": [], "group_B_on_B": []}
    for r, view in enumerate(views):
        for j, worker in enumerate(workers):
            for state, points in (("A", np.array([[20., 50.]])), ("B", np.array([[80., 50.]]))):
                label = "group_A" if worker in group_a else "group_B"
                scores[f"{label}_on_{state}"].append(predictive_log_density(view, Answer(worker, points), fitted, j, r))
    average = {name: float(np.mean(values)) for name, values in scores.items()}
    minority = float(weights[:, 1].mean())
    passed = (fitted.converged and 0.20 <= minority <= 0.40 and abs(median_a - median_b) <= 0.10
              and average["group_A_on_A"] > average["group_A_on_B"]
              and average["group_B_on_B"] > average["group_B_on_A"])
    return {"passed": bool(passed), "selected_initialization": fitted.initialization,
            "selected_status": fitted.status, "attempts": _attempts(attempts),
            "candidate_weight_A_mean": float(weights[:, 0].mean()),
            "candidate_weight_B_mean": minority, "rho_group_A": params.rho[:7].tolist(),
            "rho_group_B": params.rho[7:10].tolist(), "rho_group_A_median": median_a,
            "rho_group_B_median": median_b, "rho_median_difference": abs(median_a - median_b),
            "predictive_log_density_means": average,
            "fixed_acceptance_bounds": {"minority_weight": [0.20, 0.40], "rho_median_difference_max": 0.10}}


def check_unstable_worker() -> dict:
    views, group_a, group_b = _stable_views(True)
    workers = group_a + group_b + ("unstable",)
    fitted, attempts = _fit_two(prepare(views, workers))
    if fitted is None:
        return {"passed": False, "reason": "both_initializations_failed", "attempts": _attempts(attempts)}
    rho = fitted.parameters.rho
    stable_median = float(np.median(rho[:10]))
    unstable = float(rho[10])
    difference = stable_median - unstable
    return {"passed": bool(fitted.converged and difference >= 0.20),
            "selected_initialization": fitted.initialization, "selected_status": fitted.status,
            "attempts": _attempts(attempts), "stable_rho": rho[:10].tolist(),
            "stable_rho_median": stable_median, "unstable_rho": unstable,
            "difference": difference, "fixed_minimum_difference": 0.20,
            "unstable_answer_counts": {"A": 25, "B": 25}, "shuffle_seed": 20261008}


def check_special_cases() -> dict:
    workers = tuple(f"w{i}" for i in range(10))
    empty = np.empty((0, 2))
    all_empty = tuple(_view(QC_IDS[r // 10], str(r % 10), tuple(Answer(w, empty) for w in workers))
                      for r in range(50))
    prepared = prepare(all_empty, workers)
    fit_empty = fit(prepared, 2, "ospa_pair")
    blank = expected_mass(all_empty[0], np.ones(1))
    assert fit_empty.status == "unidentifiable" and len(all_empty[0].candidates) == 1
    assert np.all(fit_empty.parameters.rho == 0.5) and blank.sum() == 0
    left, right = np.array([[30., 50.]]), np.array([[70., 50.]])
    view = _view("complementary.png", "0", tuple(Answer(w, left if j < 5 else right)
                                                   for j, w in enumerate(workers)))
    weights = baseline_weights(view)
    mass = expected_mass(view, weights)
    samples = sample_point_sets(view, weights, 10000, 20261008)
    union = Answer("unobserved", np.vstack((left, right)))
    union_log_f = log_f_for_answer(view, union)
    assert len(view.candidates) == 2 and np.allclose(weights, [0.5, 0.5])
    assert expected_count(view, weights) == 1 and mass.sum() == 1
    assert mass[50, 30] == 0.5 and mass[50, 70] == 0.5
    assert not any(len(points) == 2 for points in samples)
    assert np.isfinite(union_log_f).all()
    return {"passed": True, "all_empty": {"candidate_count": 1, "mass_sum": float(blank.sum()),
                                            "rho_information": "unidentifiable; prior MAP 0.5"},
            "complementary": {"candidate_count": 2, "candidate_weights": weights.tolist(),
                              "expected_count": expected_count(view, weights),
                              "mass_at_left": float(mass[50, 30]), "mass_at_right": float(mass[50, 70]),
                              "mass_sum": float(mass.sum()), "samples": len(samples),
                              "unobserved_union_samples": 0,
                              "unobserved_union_log_f": union_log_f.tolist(),
                              "kernel_can_score_union_but_candidates_cannot_generate_it": True}}


def run_six_checks(on_check=None) -> dict:
    checks = {}
    for name, function in (
        ("input_candidates", check_input_candidates),
        ("observation_density", check_observation_density),
        ("estimation_qc_scope", check_estimation_qc_scope),
        ("stable_70_30", check_stable_70_30),
        ("unstable_worker", check_unstable_worker),
        ("special_cases", check_special_cases),
    ):
        checks[name] = function()
        if on_check is not None:
            on_check(name, checks[name])
    return checks


def _save_fixture() -> tuple[tuple[View, ...], tuple[str, ...], tuple[View, ...]]:
    workers = ("A", "B", "C")
    left = np.array([[20., 50.]])
    right = np.array([[80., 50.]])
    views = (
        _view("QC1.png", "0_0_100_100", (Answer("B", np.empty((0, 2))), Answer("C", left))),
        _view("QC2.png", "0_0_100_100", (Answer("A", right), Answer("B", right),
                                           Answer("C", np.empty((0, 2))))),
    )
    heldout = (_view("QC1.png", "0_0_100_100", (Answer("A", left),)),)
    return views, workers, heldout


def check_save_restore() -> dict:
    views, workers, heldout = _save_fixture()
    prepared = prepare(views, workers)
    fitted = fit(prepared, 1, "one")
    with tempfile.TemporaryDirectory(prefix="point-annotation-save-check-") as temporary:
        directory = Path(temporary)
        inputs, state = directory / "qc_inputs.json.gz", directory / "G1_one_fit.npz"
        save_qc_inputs(inputs, views, workers, heldout)
        metadata = save_fit_npz(state, fitted, prepared, inputs)
        restored, rebuilt, held, loaded_metadata = load_fit_npz(state, inputs)
        for name in ("omega", "q", "rho", "pi"):
            np.testing.assert_array_equal(getattr(fitted.parameters, name), getattr(restored.parameters, name))
        assert rebuilt.worker_ids == workers
        assert tuple((v.image_id, v.view_id) for v in rebuilt.views) == tuple((v.image_id, v.view_id) for v in views)
        assert all(np.array_equal(a.points, b.points) and a.source_worker_ids == b.source_worker_ids
                   for va, vb in zip(views, rebuilt.views) for a, b in zip(va.candidates, vb.candidates))
        objective = full_objective(rebuilt, restored.parameters)
        np.testing.assert_allclose(objective, fitted.objective, rtol=1e-10, atol=1e-8)
        assert len(held) == 1 and held[0][2].worker_id == "A"
        original_score = predictive_from_log_f(log_f_for_answer(views[0], heldout[0].answers[0]), fitted, 0, 0)
        reloaded_score = predictive_from_log_f(log_f_for_answer(rebuilt.views[0], held[0][2]), restored, 0, 0)
        np.testing.assert_allclose(reloaded_score, original_score, rtol=1e-10, atol=1e-8)
        assert loaded_metadata["inputs_sha256"] == metadata["inputs_sha256"]
        assert not any(directory.glob("*.png"))
        reordered_workers = directory / "workers_reordered.json.gz"
        save_qc_inputs(reordered_workers, views, tuple(reversed(workers)), heldout)
        try:
            load_fit_npz(state, reordered_workers)
        except ValueError as exc:
            assert "worker order" in str(exc)
        else:
            raise AssertionError("Reordered workers were accepted")
        changed = (_view("QC1.png", "0_0_100_100", tuple(reversed(views[0].answers))), views[1])
        reordered_candidates = directory / "candidates_reordered.json.gz"
        save_qc_inputs(reordered_candidates, changed, workers, heldout)
        try:
            load_fit_npz(state, reordered_candidates)
        except ValueError as exc:
            assert "candidate order" in str(exc)
        else:
            raise AssertionError("Reordered candidates were accepted")
    return {"passed": True, "fit_status": fitted.status, "objective_absolute_difference": abs(objective - fitted.objective),
            "heldout_score_absolute_difference": abs(reloaded_score - original_score),
            "image_file_needed": False, "worker_order_mismatch_rejected": True,
            "candidate_order_mismatch_rejected": True}


def check_log_and_terminal_state() -> dict:
    views, workers, _ = _save_fixture()
    prepared = prepare(views, workers)
    fitted = fit(prepared, 1, "one", max_iterations=1)
    assert fitted.status == "max_iterations" and len(fitted.history) == 1
    row = fitted.history[-1]
    names = ("max_delta_omega", "max_delta_q", "max_delta_rho", "max_delta_pi")
    assert row["accepted"] and row["stable_iterations"] < 5
    assert row["maximum_parameter_change"] == max(row[name] for name in names)
    with tempfile.TemporaryDirectory(prefix="point-annotation-terminal-check-") as temporary:
        directory = Path(temporary)
        inputs, state = directory / "qc_inputs.json.gz", directory / "G1_one_fit.npz"
        save_qc_inputs(inputs, views, workers)
        metadata = save_fit_npz(state, fitted, prepared, inputs)
        restored, rebuilt, _, _ = load_fit_npz(state, inputs)
        assert restored.status == "max_iterations" and metadata["last_accepted_iteration"] == 1
        np.testing.assert_allclose(full_objective(rebuilt, restored.parameters), fitted.objective,
                                   rtol=1e-10, atol=1e-8)
    return {"passed": True, "status": fitted.status, "last_accepted_iteration": 1,
            "stable_iterations": row["stable_iterations"],
            "deltas": {name: row[name] for name in names},
            "maximum_parameter_change": row["maximum_parameter_change"],
            "valid_state_saved_at_iteration_limit": True}


def run_eight_checks(on_check=None) -> dict:
    checks = run_six_checks(on_check)
    for name, function in (("save_restore", check_save_restore),
                           ("log_and_terminal_state", check_log_and_terminal_state)):
        checks[name] = function()
        if on_check is not None:
            on_check(name, checks[name])
    return checks


if __name__ == "__main__":
    print(json.dumps(run_eight_checks(), ensure_ascii=False, indent=2, allow_nan=False))
