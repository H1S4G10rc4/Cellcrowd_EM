"""Numerical and data-contract tests for the configuration truth model."""
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1] / "src"))
from project_paths import (PROJECT_ROOT, WORKSPACE_ROOT, SOURCE_ROOT, EXPERIMENT_ROOT,
                           FIGURE_SOURCE_ROOT, TEST_ROOT, REPORT_ROOT, REPORT_SCRIPT_ROOT,
                           METHOD_OUTPUT_ROOT, SLIDE_OUTPUT_ROOT, source_path, historical_sha256)

import unittest
import numpy as np
from scipy.integrate import simpson

from configuration_model import (Observation, Region, Settings, candidate_prior,
    expected_error_counts, fit, infer, log_count_tail, objective, point_key,
    position_density, posterior, prepare, log_position_probability,
    _single_center_log_density)
from local_regions import coarse_boxes, regions_from_views, make_candidates
from data import Answer, View


EMPTY = np.empty((0, 2))
ONE = np.array([[50., 50.]])
TWO = np.array([[35., 50.], [65., 50.]])
BOUNDS = (0., 0., 100., 100.)


def region(points, candidates=(EMPTY, ONE, TWO), name="r"):
    return Region("image", name, BOUNDS,
        tuple(Observation(str(j), p, BOUNDS) for j, p in enumerate(points)),
        candidates, tuple("observed" for _ in candidates))


class ModelChecks(unittest.TestCase):
    def test_count_normalization_and_unbounded_tail(self):
        table = np.array([.1, .2, .7])
        for m in range(12):
            p = np.array([table[min(n, 2)] * np.exp(log_count_tail(m, n)) for n in range(150)])
            self.assertAlmostEqual(float(p.sum()), 1., places=12)
            self.assertGreater(p[30], 0)

    def test_spatial_normalization(self):
        axis = np.linspace(0, 100, 501)
        xx, yy = np.meshgrid(axis, axis)
        grid = np.c_[xx.ravel(), yy.ravel()]
        for candidate in (EMPTY, ONE, TWO, np.array([[0., 0.]])):
            for count in (1, 2):
                density = position_density(grid, candidate, BOUNDS, reported_count=count).reshape(xx.shape)
                integral = simpson(simpson(density, x=axis, axis=1), x=axis)
                self.assertAlmostEqual(float(integral), 1., places=6)

    def test_partial_visibility_changes_underlying_count(self):
        r = Region("i", "r", BOUNDS,
            (Observation("a", TWO[:1], (0., 0., 50., 100.), "left"),),
            (TWO,), ("observed_full_configuration",))
        p = prepare((r,))
        self.assertEqual(p.underlying_bins[0][0, 0], 1)

    def test_empty_and_missing_are_not_equivalent(self):
        r = Region("i", "r", BOUNDS, (Observation("a", EMPTY, BOUNDS),), (EMPTY, ONE), ("empty", "observed"))
        p = prepare((r,), ("a", "b"))
        tables = np.tile(np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])[None], (2, 1, 1))
        w, _ = posterior(p, tables)
        counts = expected_error_counts(p, w)
        self.assertEqual(float(counts[1].sum()), 0.)
        self.assertAlmostEqual(float(counts[0].sum()), 1.)
        self.assertGreater(w[0][0], w[0][1])

    def test_binary_e_step_matches_direct_bayes(self):
        r = region((EMPTY, ONE), (EMPTY, ONE))
        p = prepare((r,))
        tables = np.array([[[.8, .1, .1], [.2, .7, .1], [.1, .1, .8]],
                           [[.6, .3, .1], [.1, .8, .1], [.1, .1, .8]]])
        # Exactly the independent binary observer calculation, including the
        # fixed position density, with equal configuration prior.
        raw = np.exp(p.log_position_tail[0].sum(axis=0)) * np.array([.8 * .3, .2 * .8])
        expected = raw / raw.sum()
        np.testing.assert_allclose(posterior(p, tables)[0][0], expected)

    def test_duplicate_worker_rejected_and_repeated_click_preserved(self):
        a = Observation("a", np.vstack([ONE, ONE]), BOUNDS)
        with self.assertRaises(ValueError):
            Region("i", "r", BOUNDS, (a, a), (ONE,), ("observed",))
        self.assertEqual(len(a.points), 2)
        self.assertNotEqual(point_key(a.points), point_key(ONE))

    def test_candidate_prior_does_not_reward_number_of_jittered_candidates(self):
        r = region((ONE,), (EMPTY, ONE, ONE + 1, TWO))
        np.testing.assert_allclose(candidate_prior(r), [1/3, 1/6, 1/6, 1/3])

    def test_same_count_different_positions_remain_distinguishable(self):
        left, right = TWO[:1], TWO[1:]
        r = region((left, left, left), (left, right))
        p = prepare((r,))
        w, _ = posterior(p, np.full((3, 3, 3), 1/3))
        self.assertGreater(w[0][0], .999)

    def test_merge_and_omission_are_supported(self):
        # A single click can be at the centroid or either retained point.
        for answer in (ONE, TWO[:1], TWO[1:]):
            self.assertGreater(float(position_density(answer, TWO, BOUNDS)[0]), .001)

    def test_two_point_density_is_order_invariant(self):
        p = np.array([[44., 49.], [56., 51.]])
        for s in (ONE, TWO, TWO[::-1]):
            self.assertAlmostEqual(log_position_probability(p, s, BOUNDS),
                                   log_position_probability(p[::-1], s, BOUNDS), places=12)
        self.assertAlmostEqual(log_position_probability(p, TWO, BOUNDS),
                               log_position_probability(p, TWO[::-1], BOUNDS), places=12)

    def test_reflected_conditional_is_normalized_at_edges(self):
        # The joint split density is a symmetrized product of normalized
        # conditionals. Check reflected centers inside and outside the domain.
        axis = np.linspace(0, 20, 151)
        for center in ((10., 10.), (-10., 10.), (30., 30.)):
            grid = np.array([[_single_center_log_density(np.array([x, y]), np.array(center),
                                                        (0., 0., 20., 20.), Settings())
                              for x in axis] for y in axis])
            integral = simpson(simpson(np.exp(grid), x=axis, axis=1), x=axis)
            self.assertAlmostEqual(float(integral), 1., places=4)

    def test_candidate_proposals_preserve_observed_configurations(self):
        obs = tuple(Observation(str(j), p, BOUNDS) for j, p in enumerate((TWO[:1], TWO[1:])))
        candidates, sources = make_candidates(obs, BOUNDS)
        self.assertIn(point_key(TWO), [point_key(s) for s in candidates])
        k = [point_key(s) for s in candidates].index(point_key(TWO))
        self.assertEqual(sources[k], "two_center_proposal")
        observed_only, _ = make_candidates(obs, BOUNDS, include_proposals=False)
        self.assertNotIn(point_key(TWO), [point_key(s) for s in observed_only])

    def test_region_chain_guard_no_overlap_or_point_loss(self):
        p = np.array([[x, y] for x in range(5, 200, 7) for y in range(5, 200, 7)], dtype=float)
        boxes, audit = coarse_boxes(p, (210, 210))
        self.assertGreater(len(boxes), 1)
        occurrences = sum(((p[:, 0] >= b[0]) & (p[:, 0] <= b[2]) &
                           (p[:, 1] >= b[1]) & (p[:, 1] <= b[3])).astype(int) for b in boxes)
        np.testing.assert_array_equal(occurrences, np.ones(len(p)))
        for b in boxes:
            self.assertLessEqual(max(b[2] - b[0], b[3] - b[1]), 64.000001)
        self.assertTrue(audit["chain_guard_splits"])

    def test_overlap_views_are_not_two_independent_votes(self):
        first = View("i", "first", 0, 0, 80, 80, 150, 80,
                     (Answer("a", np.array([[75., 40.]])), Answer("b", EMPTY)))
        second = View("i", "second", 70, 0, 80, 80, 150, 80,
                      (Answer("a", np.array([[5., 40.]])), Answer("b", EMPTY)))
        regions, audit = regions_from_views((first, second))
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0].observations), 2)
        self.assertEqual(sum(len(a.points) for a in regions[0].observations), 1)
        self.assertEqual(audit[0]["overlap_alternatives_not_counted"], 2)
        self.assertEqual(audit[0]["partial_region_answers"], 0)
        self.assertEqual(regions[0].observations[0].bounds,(0,0,80,80))

    def test_clean_two_point_response_is_not_pre_split_into_single_cells(self):
        view = View("i", "r", 0, 0, 100, 100, 100, 100,
                    tuple(Answer(str(j), TWO + j*.1) for j in range(10)))
        regions,_ = regions_from_views((view,))
        self.assertEqual(len(regions),1)
        self.assertTrue(all(len(a.points)==2 for a in regions[0].observations))

    def test_empty_gap_is_preferred_over_forced_balanced_cut(self):
        views=tuple(View("i",str(i),100*i,0,100,100,1000,100,
                         tuple(Answer(str(j),TWO+j*.1) for j in range(10))) for i in range(10))
        regions,_=regions_from_views(views)
        self.assertEqual(len(regions),10)
        self.assertTrue(all(len(a.points)==2 for r in regions for a in r.observations))

    def test_em_final_posterior_and_fixed_application(self):
        rng = np.random.default_rng(9)
        examples = (EMPTY, ONE, TWO)
        regions = []
        for i in range(45):
            underlying = i % 3
            answers = [examples[underlying if rng.random() < .85 else rng.integers(3)] for _ in range(6)]
            regions.append(region(answers, name=str(i)))
        p = prepare(regions)
        results = [fit(p, start) for start in ("diagonal", "uniform", "random")]
        for result in results:
            self.assertTrue(result.converged)
            self.assertGreaterEqual(float(np.diff([h["objective"] for h in result.history]).min()), -1e-8)
            value, refreshed = objective(p, result.error_tables)
            self.assertAlmostEqual(value, result.objective, places=9)
            for a, b in zip(refreshed, result.weights):
                np.testing.assert_array_equal(a, b)
        best = max(results, key=lambda x: x.objective)
        before = best.error_tables.copy()
        inferred, audit = infer((region((TWO,) * 6),), best)
        np.testing.assert_array_equal(before, best.error_tables)
        self.assertEqual(audit["unknown_workers"], [])
        self.assertAlmostEqual(float(inferred[0].sum()), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
