"""Whole-configuration latent truth with a small worker count-error table.

Only C[j, underlying_count_bin, reported_count_bin] is fitted. Count bins are
0, 1, and 2+ (six free numbers per worker). Candidate locations, pixel bandwidth,
tail distribution, and regularization are fixed. No worker classes or scales
are estimated. All probabilities are conditional on the supplied candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.special import logsumexp, ndtr, log_ndtr


@dataclass(frozen=True)
class Settings:
    sigma_pixels: float = 4.0
    uniform_fraction: float = 0.01
    tail_ratio: float = 0.5
    prior_strength: float = 3.0
    prior_diagonal: float = 0.8
    max_iterations: int = 2000
    tolerance: float = 1e-7

    def __post_init__(self):
        if not (self.sigma_pixels > 0 and 0 < self.uniform_fraction < 1
                and 0 < self.tail_ratio < 1 and self.prior_strength > 0
                and 0 < self.prior_diagonal < 1 and self.max_iterations > 0
                and self.tolerance > 0):
            raise ValueError("Invalid fixed model settings")

    @property
    def pseudocounts(self):
        p = np.full((3, 3), (1 - self.prior_diagonal) / 2)
        np.fill_diagonal(p, self.prior_diagonal)
        return self.prior_strength * p


def as_points(value):
    a = np.asarray(value, dtype=float).reshape(-1, 2).copy()
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite point coordinates")
    a.setflags(write=False)
    return a


def point_key(value):
    """An exact multiset: ignore order, never round or remove repeated clicks."""
    return tuple(sorted(map(tuple, as_points(value))))


def inside(points, bounds):
    x0, y0, x1, y1 = bounds
    return ((points[:, 0] >= x0) & (points[:, 0] <= x1)
            & (points[:, 1] >= y0) & (points[:, 1] <= y1))


@dataclass(frozen=True)
class Observation:
    worker_id: str
    points: np.ndarray
    bounds: tuple[float, float, float, float]
    view_id: str = ""

    def __post_init__(self):
        p = as_points(self.points)
        x0, y0, x1, y1 = self.bounds
        if not np.isfinite(self.bounds).all() or x0 >= x1 or y0 >= y1:
            raise ValueError("Observation needs a positive visible area")
        if not inside(p, self.bounds).all():
            raise ValueError("Response outside the visible region")
        object.__setattr__(self, "points", p)


@dataclass(frozen=True)
class Region:
    image_id: str
    region_id: str
    bounds: tuple[float, float, float, float]
    observations: tuple[Observation, ...]
    candidates: tuple[np.ndarray, ...]
    candidate_sources: tuple[str, ...]

    def __post_init__(self):
        workers = [a.worker_id for a in self.observations]
        if len(set(workers)) != len(workers):
            raise ValueError("Use at most one original response per worker and region")
        if not self.candidates or len(self.candidates) != len(self.candidate_sources):
            raise ValueError("Candidate provenance is required")
        candidates = tuple(as_points(s) for s in self.candidates)
        if len(set(map(point_key, candidates))) != len(candidates):
            raise ValueError("Duplicate candidate multisets")
        if not all(inside(s, self.bounds).all() for s in candidates):
            raise ValueError("Candidate outside region bounds")
        object.__setattr__(self, "candidates", candidates)


def candidate_prior(region: Region):
    """Equal mass per represented exact count, then per configuration of that count.

    More jittered responses at one count must not silently multiply its prior.
    This is a fixed, candidate-conditional prior, not an estimated prevalence.
    """
    counts = np.array([len(s) for s in region.candidates])
    distinct, frequency = np.unique(counts, return_counts=True)
    return np.array([1 / (len(distinct) * frequency[np.flatnonzero(distinct == n)[0]])
                     for n in counts])


def log_count_tail(m: int, n: int, ratio: float = 0.5):
    """Normalized P(n | reported bin, m); actual counts >=2 are never discarded."""
    if m < 0 or n < 0:
        raise ValueError("Negative point count")
    if n < 2:
        return 0.0
    center = max(m, 2) - 2
    z = (1 + ratio - ratio ** (center + 1)) / (1 - ratio)
    return abs(n - 2 - center) * np.log(ratio) - np.log(z)


def position_density(points, candidate, bounds, settings=Settings(), reported_count=None):
    """Fixed normalized spatial observation density, in inverse square pixels.

    When several latent points yield one click, its center may be an individual
    point (omission) or the configuration centroid (merging), with equal fixed
    mixture weight. Otherwise use the original equal Gaussian mixture. This
    count-conditional extension is essential: a central merged click is not a
    distant random click. It adds no fitted parameters. For >2 underlying points,
    intermediate partial merges are not modeled.
    """
    p, s = as_points(points), as_points(candidate)
    x0, y0, x1, y1 = bounds
    area = (x1 - x0) * (y1 - y0)
    if area <= 0 or not inside(p, bounds).all() or not inside(s, bounds).all():
        raise ValueError("Invalid spatial domain")
    if not len(s):
        return np.full(len(p), 1 / area)
    weights = np.full(len(s), 1 / len(s))
    n = len(p) if reported_count is None else reported_count
    if n == 1 and len(s) > 1:
        s = np.vstack((s, s.mean(axis=0)))
        weights = np.r_[weights * 0.5, 0.5]
    sigma = settings.sigma_pixels
    norm = np.prod(ndtr((np.array([x1, y1]) - s) / sigma)
                   - ndtr((np.array([x0, y0]) - s) / sigma), axis=1)
    d = (p[:, None, :] - s[None, :, :]) / sigma
    gaussian = np.exp(-0.5 * np.square(d).sum(axis=2)) / (2 * np.pi * sigma**2 * norm)
    return settings.uniform_fraction / area + (1 - settings.uniform_fraction) * (gaussian @ weights)


def _single_center_log_density(point, center, bounds, settings):
    """Normalized truncated Gaussian plus uniform; center may be outside bounds."""
    lo, hi = np.array(bounds[:2]), np.array(bounds[2:])
    sigma = settings.sigma_pixels
    a, b = (lo - center) / sigma, (hi - center) / sigma
    # Use survival probabilities in the positive tail to avoid 1 - 1.
    lower = np.where(a > 0, -b, a)
    upper = np.where(a > 0, -a, b)
    log_hi, log_lo = log_ndtr(upper), log_ndtr(lower)
    log_z = log_hi + np.log(-np.expm1(log_lo - log_hi))
    log_gauss = (-.5 * np.square((point - center) / sigma).sum()
                 - np.log(2 * np.pi * sigma**2) - log_z.sum())
    return float(np.logaddexp(np.log(settings.uniform_fraction) - np.log(np.prod(hi - lo)),
                             np.log1p(-settings.uniform_fraction) + log_gauss))


def log_position_probability(points, candidate, bounds, settings=Settings()):
    """Exchangeable ordered-coordinate density conditional on the count.

    A split single point produces two clicks on opposite sides of its center:
    draw the first click, then a click near its reflection, and symmetrize their
    order. Two distinct underlying points instead produce one click each.
    Every conditional draw is normalized on the same visible rectangle. Both
    mechanisms use the existing fixed pixel bandwidth; no scale or split-rate
    parameter is added. Count probabilities are learned separately in C.
    Larger configurations retain the original iid spatial approximation.
    """
    p, s = as_points(points), as_points(candidate)
    if len(p) == 2 and len(s) in (1, 2):
        terms = []
        for first, second in ((0, 1), (1, 0)):
            if len(s) == 1:
                center1, center2 = s[0], 2 * s[0] - p[first]
            else:
                center1, center2 = s[0], s[1]
            terms.append(_single_center_log_density(p[first], center1, bounds, settings)
                         + _single_center_log_density(p[second], center2, bounds, settings))
        return float(logsumexp(terms) - np.log(2.))
    return float(np.log(position_density(p, s, bounds, settings)).sum())


@dataclass(frozen=True)
class Prepared:
    regions: tuple[Region, ...]
    worker_ids: tuple[str, ...]
    log_position_tail: tuple[np.ndarray, ...]
    underlying_bins: tuple[np.ndarray, ...]
    reported_bins: tuple[np.ndarray, ...]
    worker_indices: tuple[np.ndarray, ...]
    log_priors: tuple[np.ndarray, ...]
    settings: Settings


def prepare(regions, worker_ids=None, settings=Settings()):
    regions = tuple(regions)
    workers = tuple(sorted({a.worker_id for r in regions for a in r.observations})) if worker_ids is None else tuple(worker_ids)
    if not regions or not workers or len(set(workers)) != len(workers):
        raise ValueError("Nonempty regions and unique workers required")
    lookup = {w: j for j, w in enumerate(workers)}
    logs, bins, reported, indices, priors = [], [], [], [], []
    for region in regions:
        f = np.zeros((len(region.observations), len(region.candidates)))
        c = np.zeros(f.shape, dtype=int)
        for j, a in enumerate(region.observations):
            for k, s in enumerate(region.candidates):
                visible = s[inside(s, a.bounds)]
                m, n = len(visible), len(a.points)
                c[j, k] = min(m, 2)
                f[j, k] = (log_count_tail(m, n, settings.tail_ratio)
                           + log_position_probability(a.points, visible, a.bounds, settings))
        logs.append(f)
        bins.append(c)
        reported.append(np.array([min(len(a.points), 2) for a in region.observations], dtype=int))
        indices.append(np.array([lookup[a.worker_id] for a in region.observations], dtype=int))
        priors.append(np.log(candidate_prior(region)))
    return Prepared(regions, workers, tuple(logs), tuple(bins), tuple(reported), tuple(indices), tuple(priors), settings)


def posterior(prepared: Prepared, error_tables):
    c = np.asarray(error_tables)
    if c.shape != (len(prepared.worker_ids), 3, 3) or np.any(c <= 0) or not np.allclose(c.sum(axis=2), 1):
        raise ValueError("Strictly positive normalized worker error tables required")
    weights, likelihood = [], 0.0
    for f, truth, obs, idx, prior in zip(prepared.log_position_tail, prepared.underlying_bins,
                                        prepared.reported_bins, prepared.worker_indices, prepared.log_priors):
        a = prior + (f + np.log(c[idx[:, None], truth, obs[:, None]])).sum(axis=0)
        z = float(logsumexp(a))
        weights.append(np.exp(a - z))
        likelihood += z
    return tuple(weights), likelihood


def objective(prepared, error_tables):
    weights, likelihood = posterior(prepared, error_tables)
    penalty = float((prepared.settings.pseudocounts[None] * np.log(error_tables)).sum())
    return likelihood + penalty, weights


def expected_error_counts(prepared, weights):
    counts = np.zeros((len(prepared.worker_ids), 3, 3))
    for w, truth, obs, idx in zip(weights, prepared.underlying_bins,
                                  prepared.reported_bins, prepared.worker_indices):
        np.add.at(counts, (idx[:, None], truth, obs[:, None]), np.broadcast_to(w, truth.shape))
    return counts


@dataclass
class Fit:
    worker_ids: tuple[str, ...]
    error_tables: np.ndarray
    weights: tuple[np.ndarray, ...]
    expected_counts: np.ndarray
    history: list[dict]
    converged: bool
    initialization: str
    settings: Settings
    runtime_seconds: float

    @property
    def objective(self):
        return self.history[-1]["objective"]


def fit(prepared: Prepared, initialization="diagonal", seed=20261010):
    started = time.perf_counter()
    cfg = prepared.settings
    base = cfg.pseudocounts
    if initialization == "diagonal":
        tables = np.tile((base / base.sum(axis=1, keepdims=True))[None], (len(prepared.worker_ids), 1, 1))
    elif initialization == "uniform":
        tables = np.full((len(prepared.worker_ids), 3, 3), 1 / 3)
    elif initialization == "random":
        rng = np.random.default_rng(seed)
        tables = np.stack([np.stack([rng.dirichlet(row + 1) for row in base]) for _ in prepared.worker_ids])
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    value, weights = objective(prepared, tables)
    history = [{"iteration": 0, "objective": value, "max_parameter_change": 0., "max_posterior_change": 0.}]
    converged = False
    for iteration in range(1, cfg.max_iterations + 1):
        counts = expected_error_counts(prepared, weights) + base[None]
        updated = counts / counts.sum(axis=2, keepdims=True)
        new_value, new_weights = objective(prepared, updated)
        delta = new_value - value
        if delta < -1e-9 * max(1, abs(value)):
            raise FloatingPointError("MAP objective decreased")
        parameter_change = float(np.max(abs(updated - tables)))
        posterior_change = max(float(np.max(abs(a - b))) for a, b in zip(new_weights, weights))
        history.append({"iteration": iteration, "objective": new_value,
                        "max_parameter_change": parameter_change, "max_posterior_change": posterior_change})
        tables, weights, value = updated, new_weights, new_value
        if max(parameter_change, posterior_change) < cfg.tolerance and abs(delta) < cfg.tolerance * max(1, abs(value)):
            converged = True
            break
    # The returned posterior uses the final table, not the preceding E-step.
    return Fit(prepared.worker_ids, tables, weights, expected_error_counts(prepared, weights),
               history, converged, initialization, cfg, time.perf_counter() - started)


def infer(regions, fitted: Fit):
    if not fitted.converged:
        raise ValueError("Do not apply an unconverged fit")
    prepared = prepare(regions, settings=fitted.settings)
    known = {w: fitted.error_tables[j] for j, w in enumerate(fitted.worker_ids)}
    fallback = fitted.error_tables.mean(axis=0)
    tables = np.stack([known.get(w, fallback) for w in prepared.worker_ids])
    weights, likelihood = posterior(prepared, tables)
    return weights, {"log_likelihood": likelihood,
                     "unknown_workers": [w for w in prepared.worker_ids if w not in known]}


def summarize(region, weights):
    distribution = {}
    for s, w in zip(region.candidates, weights):
        distribution[len(s)] = distribution.get(len(s), 0.) + float(w)
    return {"image_id": region.image_id, "region_id": region.region_id,
            "bounds": list(region.bounds), "candidate_points": [s.tolist() for s in region.candidates],
            "candidate_sources": list(region.candidate_sources), "weights": np.asarray(weights).tolist(),
            "count_distribution": dict(sorted(distribution.items())),
            "expected_count": sum(n * p for n, p in distribution.items()),
            "map_candidate_index": int(np.argmax(weights)), "workers": len(region.observations)}
