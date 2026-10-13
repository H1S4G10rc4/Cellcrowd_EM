"""Fixed observation density and one/two-class conditional MAP inference."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, ndtr

from data import Answer, View

Q_COUNT = 0.5
BANDWIDTH = 0.05
UNIFORM_FRACTION = 0.01
MAX_ITERATIONS = 200
MAX_SECONDS_PER_START = 180.0


def log_count_probability(m: int, n: np.ndarray | int) -> np.ndarray:
    if m < 0 or np.any(np.asarray(n) < 0):
        raise ValueError("Counts must be nonnegative")
    z = (1 + Q_COUNT - Q_COUNT ** (m + 1)) / (1 - Q_COUNT)
    return np.abs(np.asarray(n, dtype=np.float64) - m) * np.log(Q_COUNT) - np.log(z)


def truncation_normalizer(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    return ndtr((1.0 - centers) / BANDWIDTH) - ndtr(-centers / BANDWIDTH)


def spatial_density(points: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Density for arbitrary normalized positions, including held-out points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all() or not np.isfinite(candidate).all() or np.any(points < 0) or np.any(points > 1) or np.any(candidate < 0) or np.any(candidate > 1):
        raise ValueError("Points must be finite coordinates in [0,1]^2")
    if len(candidate) == 0:
        return np.ones(len(points), dtype=np.float64)
    diff = (points[:, None, :] - candidate[None, :, :]) / BANDWIDTH
    normalizers = np.prod(truncation_normalizer(candidate), axis=1)
    kernel = np.exp(-0.5 * np.square(diff).sum(axis=2)) / (2 * np.pi * BANDWIDTH**2 * normalizers[None, :])
    return UNIFORM_FRACTION + (1 - UNIFORM_FRACTION) * kernel.sum(axis=1) / len(candidate)


def log_f_for_points(normalized_points: np.ndarray, normalized_candidates: tuple[np.ndarray, ...]) -> np.ndarray:
    points = np.asarray(normalized_points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        raise ValueError("Observation coordinates must be in [0,1]^2")
    result = np.empty(len(normalized_candidates), dtype=np.float64)
    for k, candidate in enumerate(normalized_candidates):
        density = spatial_density(points, candidate)
        result[k] = float(log_count_probability(len(candidate), len(points)) + np.log(density).sum())
    return result


def log_f_for_answer(view: View, answer: Answer) -> np.ndarray:
    points = answer.points / np.array([view.width, view.height], dtype=np.float64)
    return log_f_for_points(points, view.normalized_candidates)


def log_f_matrix(view: View) -> np.ndarray:
    """Calculate each observed-answer/candidate density once per view."""
    lengths = np.array([len(a.points) for a in view.answers], dtype=int)
    ends = np.cumsum(lengths)
    starts = ends - lengths
    scale = np.array([view.width, view.height], dtype=np.float64)
    all_points = np.concatenate([a.points for a in view.answers], axis=0) / scale
    matrix = np.empty((len(view.answers), len(view.candidates)), dtype=np.float64)
    for k, candidate in enumerate(view.normalized_candidates):
        log_density = np.log(spatial_density(all_points, candidate))
        prefix = np.r_[0.0, np.cumsum(log_density)]
        matrix[:, k] = log_count_probability(len(candidate), lengths) + prefix[ends] - prefix[starts]
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"Non-finite observation density: {view.image_id}/{view.view_id}")
    return matrix


@dataclass(frozen=True)
class Prepared:
    views: tuple[View, ...]
    worker_ids: tuple[str, ...]
    log_f: np.ndarray        # [view, worker, candidate], -inf for absent/padded
    log_b: np.ndarray        # [view, worker], zero for absent
    present: np.ndarray      # actual answers on K>=2 learning views
    observed_index: np.ndarray
    k_per_view: np.ndarray
    k1_constant: float

    @property
    def number_of_learning_views(self) -> int:
        return int(np.sum(self.k_per_view >= 2))

    @property
    def learning_count_per_worker(self) -> np.ndarray:
        return self.present.sum(axis=0)


def prepare(views: tuple[View, ...], worker_ids: tuple[str, ...]) -> Prepared:
    if not views or not worker_ids or len(set(worker_ids)) != len(worker_ids):
        raise ValueError("Nonempty views and distinct workers are required")
    n_views, n_workers = len(views), len(worker_ids)
    k_values = np.array([len(v.candidates) for v in views], dtype=int)
    max_k = int(k_values.max())
    log_f = np.full((n_views, n_workers, max_k), -np.inf, dtype=np.float64)
    log_b = np.zeros((n_views, n_workers), dtype=np.float64)
    present = np.zeros((n_views, n_workers), dtype=bool)
    observed = np.full((n_views, n_workers), -1, dtype=int)
    worker_lookup = {worker: j for j, worker in enumerate(worker_ids)}
    k1_constant = 0.0
    for r, view in enumerate(views):
        matrix = log_f_matrix(view)
        for row, answer in enumerate(view.answers):
            j = worker_lookup[answer.worker_id]
            observed[r, j] = view.observed_index[answer.worker_id]
            if k_values[r] == 1:
                k1_constant += float(matrix[row, 0])
            else:
                present[r, j] = True
                log_f[r, j, :k_values[r]] = matrix[row]
                log_b[r, j] = float(logsumexp(matrix[row]) - np.log(k_values[r]))
    return Prepared(views, worker_ids, log_f, log_b, present, observed, k_values, k1_constant)


def ospa_distance(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, dtype=np.float64).reshape(-1, 2), np.asarray(second, dtype=np.float64).reshape(-1, 2)
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    if n == 0:
        return 0.0
    if m == 0:
        return 1.0
    cost = cdist(a, b, metric="sqeuclidean") / 2.0
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt((cost[rows, columns].sum() + n - m) / n))


def ospa_pair_initialization(prepared: Prepared) -> tuple[np.ndarray, tuple[str, str]]:
    """Mean co-observed OSPA; choose the minimum two-medoids score."""
    workers = prepared.worker_ids
    j_count = len(workers)
    totals = np.zeros((j_count, j_count), dtype=np.float64)
    counts = np.zeros_like(totals)
    for r, view in enumerate(prepared.views):
        if prepared.k_per_view[r] == 1:
            continue
        present_ids = np.flatnonzero(prepared.present[r])
        states = view.normalized_candidates
        distances = np.zeros((len(states), len(states)), dtype=np.float64)
        for k in range(len(states)):
            for l in range(k):
                distances[k, l] = distances[l, k] = ospa_distance(states[k], states[l])
        observed = prepared.observed_index[r, present_ids]
        ix = np.ix_(present_ids, present_ids)
        totals[ix] += distances[np.ix_(observed, observed)]
        counts[ix] += 1
    mean = np.full_like(totals, np.inf)
    np.divide(totals, counts, out=mean, where=counts > 0)
    eligible = np.flatnonzero(prepared.learning_count_per_worker > 0)
    if len(eligible) < 2:
        raise ValueError("Two informative workers are needed for the OSPA initialization")
    best_pair, best_score = None, np.inf
    for x, i in enumerate(eligible):
        for j in eligible[x+1:]:
            score = float(np.minimum(mean[eligible, i], mean[eligible, j]).sum())
            if score < best_score - 1e-12:
                best_pair, best_score = (int(i), int(j)), score
    if best_pair is None:
        raise ValueError("No finite OSPA representative pair")
    i, j = best_pair
    initial = np.full((j_count, 2), 0.5, dtype=np.float64)
    initial[eligible, 0] = np.where(mean[eligible, i] <= mean[eligible, j], 0.98, 0.02)
    initial[eligible, 1] = 1 - initial[eligible, 0]
    return initial, (workers[i], workers[j])


@dataclass
class Parameters:
    omega: np.ndarray     # G
    rho: np.ndarray       # worker
    pi: np.ndarray        # view, G, candidate (zero padding)
    q: np.ndarray         # worker, G


@dataclass
class Fit:
    G: int
    initialization: str
    parameters: Parameters
    objective: float
    history: list[dict]
    status: str
    runtime_seconds: float
    representative_workers: tuple[str, str] | None = None

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _initial_parameters(prepared: Prepared, G: int, initialization: str) -> tuple[Parameters, tuple[str, str] | None]:
    j_count = len(prepared.worker_ids)
    representatives = None
    if G == 1:
        q = np.ones((j_count, 1), dtype=np.float64)
    elif initialization == "ospa_pair":
        q, representatives = ospa_pair_initialization(prepared)
    elif initialization == "random_20261008":
        first = np.random.default_rng(20261008).uniform(0.2, 0.8, j_count)
        q = np.column_stack((first, 1 - first))
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    omega = (1.0 + q.sum(axis=0)) / (G + j_count)
    rho = np.full(j_count, 0.8, dtype=np.float64)
    rho[prepared.learning_count_per_worker == 0] = 0.5
    pi = np.zeros((len(prepared.views), G, prepared.log_f.shape[-1]), dtype=np.float64)
    for r, k in enumerate(prepared.k_per_view):
        if k == 1:
            pi[r, :, 0] = 1
            continue
        present_ids = np.flatnonzero(prepared.present[r])
        observed = prepared.observed_index[r, present_ids]
        for g in range(G):
            weighted = np.bincount(observed, weights=q[present_ids, g], minlength=int(k))
            pi[r, g, :k] = (weighted + 1.0 / k) / (weighted.sum() + 1.0)
    return Parameters(omega, rho, pi, q), representatives


def _log_m(prepared: Prepared, parameters: Parameters) -> np.ndarray:
    with np.errstate(divide="ignore"):
        log_pi = np.log(parameters.pi)
    log_class_density = logsumexp(prepared.log_f[:, :, None, :] + log_pi[:, None, :, :], axis=3)
    log_m = np.logaddexp(
        np.log(parameters.rho)[None, :, None] + log_class_density,
        np.log1p(-parameters.rho)[None, :, None] + prepared.log_b[:, :, None],
    )
    return np.where(prepared.present[:, :, None], log_m, 0.0)


def _posterior_q(log_m: np.ndarray, omega: np.ndarray) -> np.ndarray:
    logits = np.log(omega)[None, :] + log_m.sum(axis=0)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def full_objective(prepared: Prepared, parameters: Parameters, log_m: np.ndarray | None = None) -> float:
    log_m = _log_m(prepared, parameters) if log_m is None else log_m
    data = np.sum(logsumexp(np.log(parameters.omega)[None, :] + log_m.sum(axis=0), axis=1))
    prior_omega = np.log(parameters.omega).sum()
    prior_rho = np.log(parameters.rho).sum() + np.log1p(-parameters.rho).sum()
    prior_pi = sum(np.log(parameters.pi[r, :, :k]).sum() / k
                   for r, k in enumerate(prepared.k_per_view) if k >= 2)
    return float(data + prior_omega + prior_rho + prior_pi + prepared.k1_constant)


def _next_parameters(prepared: Prepared, current: Parameters, log_m: np.ndarray) -> Parameters:
    q = _posterior_q(log_m, current.omega)
    with np.errstate(divide="ignore"):
        log_pi = np.log(current.pi)
        log_q = np.log(q)
    log_a = (
        log_q[None, :, :, None]
        + np.log(current.rho)[None, :, None, None]
        + log_pi[:, None, :, :]
        + prepared.log_f[:, :, None, :]
        - log_m[:, :, :, None]
    )
    a = np.exp(log_a)
    omega = (1 + q.sum(axis=0)) / (q.shape[1] + len(prepared.worker_ids))
    rho = (1 + a.sum(axis=(0, 2, 3))) / (2 + prepared.learning_count_per_worker)
    rho[prepared.learning_count_per_worker == 0] = 0.5
    pi = np.zeros_like(current.pi)
    counts = a.sum(axis=1)
    for r, k in enumerate(prepared.k_per_view):
        if k == 1:
            pi[r, :, 0] = 1
        else:
            gamma = 1.0 / k + counts[r, :, :k]
            pi[r, :, :k] = gamma / (1 + counts[r, :, :k].sum(axis=1, keepdims=True))
    provisional = Parameters(omega, rho, pi, q)
    return Parameters(omega, rho, pi, _posterior_q(_log_m(prepared, provisional), omega))


def fit(
    prepared: Prepared, G: int, initialization: str,
    on_iteration: Callable[[dict, list[dict]], None] | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> Fit:
    if G not in (1, 2):
        raise ValueError("Only G=1 or G=2 is specified")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    started = time.perf_counter()
    if prepared.number_of_learning_views == 0:
        q = np.ones((len(prepared.worker_ids), 1)) if G == 1 else np.full((len(prepared.worker_ids), 2), 0.5)
        omega = np.ones(1) if G == 1 else np.full(2, 0.5)
        p = Parameters(omega, np.full(len(prepared.worker_ids), 0.5),
                       np.ones((len(prepared.views), G, 1)), q)
        return Fit(G, initialization, p, full_objective(prepared, p), [], "unidentifiable", time.perf_counter() - started)
    current, representatives = _initial_parameters(prepared, G, initialization)
    previous_objective = full_objective(prepared, current)
    history: list[dict] = []
    stable = 0
    status = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        if time.perf_counter() - started >= MAX_SECONDS_PER_START:
            status = "time_limit"
            break
        try:
            old_log_m = _log_m(prepared, current)
            updated = _next_parameters(prepared, current, old_log_m)
            objective = full_objective(prepared, updated)
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            status = "numerical_failure"
            history.append({"iteration": iteration, "objective": None, "accepted": False,
                            "reason": f"invalid_update: {exc}",
                            "runtime_seconds": time.perf_counter() - started})
            if on_iteration:
                on_iteration(history[-1], history)
            break
        if not np.isfinite(objective) or objective < previous_objective - 1e-8 * max(1.0, abs(previous_objective)):
            # Recompute the same step with float64 and logsumexp once. A
            # persistent decrease invalidates the entire initialization.
            try:
                old_log_m = _log_m(prepared, current)
                repeated = _next_parameters(prepared, current, old_log_m)
                objective = full_objective(prepared, repeated)
            except (ValueError, FloatingPointError, ArithmeticError) as exc:
                status = "numerical_failure"
                history.append({"iteration": iteration, "objective": None, "accepted": False,
                                "reason": f"recalculation_failed: {exc}",
                                "runtime_seconds": time.perf_counter() - started})
                if on_iteration:
                    on_iteration(history[-1], history)
                break
            if not np.isfinite(objective) or objective < previous_objective - 1e-8 * max(1.0, abs(previous_objective)):
                status = "numerical_failure"
                history.append({"iteration": iteration, "objective": float(objective) if np.isfinite(objective) else None,
                                "accepted": False, "reason": "objective_decrease_after_recalculation",
                                "runtime_seconds": time.perf_counter() - started})
                if on_iteration:
                    on_iteration(history[-1], history)
                break
            updated = repeated
        deltas = {
            "max_delta_omega": float(np.max(np.abs(current.omega - updated.omega))),
            "max_delta_q": float(np.max(np.abs(current.q - updated.q))),
            "max_delta_rho": float(np.max(np.abs(current.rho - updated.rho))),
            "max_delta_pi": max(float(np.max(np.abs(current.pi[r, :, :k] - updated.pi[r, :, :k])))
                                for r, k in enumerate(prepared.k_per_view)),
        }
        delta = max(deltas.values())
        relative = abs(objective - previous_objective) / max(1.0, abs(previous_objective))
        stable = stable + 1 if relative < 1e-7 and delta < 1e-5 else 0
        current, previous_objective = updated, objective
        row = {"iteration": iteration, "objective": objective, "relative_objective_change": relative,
               "maximum_parameter_change": delta, "stable_iterations": stable, "accepted": True,
               "runtime_seconds": time.perf_counter() - started, **deltas}
        history.append(row)
        if on_iteration:
            on_iteration(row, history)
        if stable >= 5:
            status = "converged"
            break
        if time.perf_counter() - started >= MAX_SECONDS_PER_START:
            status = "time_limit"
            break
    return Fit(G, initialization, current, previous_objective, history, status,
               time.perf_counter() - started, representatives)


def predictive_from_log_f(log_f: np.ndarray, fitted: Fit, worker_index: int, view_index: int) -> float:
    """Held-out score using fixed q/rho/pi and the common F convention."""
    log_b = float(logsumexp(log_f) - np.log(len(log_f)))
    p = fitted.parameters
    log_class = np.array([logsumexp(np.log(p.pi[view_index, g, :len(log_f)]) + log_f)
                          for g in range(fitted.G)])
    log_signal = float(logsumexp(np.log(p.q[worker_index]) + log_class))
    return float(np.logaddexp(np.log(p.rho[worker_index]) + log_signal,
                             np.log1p(-p.rho[worker_index]) + log_b))


def predictive_log_density(view: View, answer: Answer, fitted: Fit, worker_index: int, view_index: int) -> float:
    return predictive_from_log_f(log_f_for_answer(view, answer), fitted, worker_index, view_index)


def infer_view(view: View, worker_parameters: dict[str, tuple[np.ndarray, float]], G: int) -> dict:
    """Update only class-specific view weights with QC q and rho fixed."""
    k = len(view.candidates)
    if k == 1:
        q = np.stack([worker_parameters[a.worker_id][0] for a in view.answers])
        return {"pi": np.ones((G, 1)), "weights": np.ones(1), "v": q.mean(axis=0),
                "iterations": 0, "converged": True, "objective": None,
                "status": "single_candidate"}
    workers = [a.worker_id for a in view.answers]
    q = np.stack([worker_parameters[w][0] for w in workers])
    rho = np.array([worker_parameters[w][1] for w in workers])
    with np.errstate(divide="ignore"):
        log_q = np.log(q)
    log_f = log_f_matrix(view)
    log_b = logsumexp(log_f, axis=1) - np.log(k)
    pi = np.full((G, k), 1 / k, dtype=np.float64)
    v = q.mean(axis=0)
    previous = None
    converged = False
    objective = None
    for iteration in range(1, 101):
        log_class = logsumexp(np.log(pi)[None, :, :] + log_f[:, None, :], axis=2)
        log_t = np.logaddexp(np.log(rho) + logsumexp(log_q + log_class, axis=1),
                            np.log1p(-rho) + log_b)
        log_u = (np.log(rho)[:, None, None] + log_q[:, :, None]
                 + np.log(pi)[None, :, :] + log_f[:, None, :] - log_t[:, None, None])
        u = np.exp(log_u)
        total = u.sum(axis=0)
        updated = (1.0 / k + total) / (1.0 + total.sum(axis=1, keepdims=True))
        log_class_updated = logsumexp(np.log(updated)[None, :, :] + log_f[:, None, :], axis=2)
        log_t_updated = np.logaddexp(np.log(rho) + logsumexp(log_q + log_class_updated, axis=1),
                                    np.log1p(-rho) + log_b)
        objective = float(log_t_updated.sum() + np.log(updated).sum() / k)
        change = float(np.max(np.abs(updated - pi)))
        relative = float("inf") if previous is None else abs(objective - previous) / max(1.0, abs(previous))
        pi, previous = updated, objective
        if change < 1e-6 and relative < 1e-7:
            converged = True
            break
    weights = v @ pi
    return {"pi": pi, "weights": weights, "v": v, "iterations": iteration,
            "converged": converged, "objective": objective,
            "status": "converged" if converged else "max_iterations"}


def _view_key(view: View) -> str:
    return f"{view.image_id}/{view.view_id}"


def _candidate_order_digest(view: View) -> str:
    candidates = [{"points": candidate.points.tolist(), "source_worker_ids": list(candidate.source_worker_ids)}
                  for candidate in view.candidates]
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_qc_inputs(path: Path, views: tuple[View, ...], worker_ids: tuple[str, ...],
                   heldout_views: tuple[View, ...] = ()) -> str:
    """Store only the ordered QC records needed for image-free recomputation."""
    payload = {
        "format": "point_annotation_trial_qc_inputs_v1",
        "worker_ids": list(worker_ids),
        "views": [{"image_id": view.image_id, "view_id": view.view_id,
                   "origin_x": view.origin_x, "origin_y": view.origin_y,
                   "width": view.width, "height": view.height,
                   "candidates": [{"points": candidate.points.tolist(),
                                   "support_count": candidate.support_count,
                                   "source_worker_ids": list(candidate.source_worker_ids)}
                                  for candidate in view.candidates],
                   "answers": [{"worker_id": answer.worker_id, "points": answer.points.tolist()}
                               for answer in view.answers]}
                  for view in views],
        "heldout_views": [{"image_id": view.image_id, "view_id": view.view_id,
                           "answers": [{"worker_id": answer.worker_id, "points": answer.points.tolist()}
                                       for answer in view.answers]}
                          for view in heldout_views],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_qc_inputs(path: Path) -> tuple[tuple[View, ...], tuple[str, ...], tuple[tuple[str, str, Answer], ...]]:
    payload = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    if payload.get("format") != "point_annotation_trial_qc_inputs_v1":
        raise ValueError("Unsupported QC input format")
    workers = tuple(payload["worker_ids"])
    if len(set(workers)) != len(workers):
        raise ValueError("Duplicate worker IDs in QC inputs")
    views = []
    for entry in payload["views"]:
        answers = tuple(Answer(item["worker_id"], np.asarray(item["points"], dtype=np.float64).reshape(-1, 2))
                        for item in entry["answers"])
        x, y, width, height = (int(entry[name]) for name in ("origin_x", "origin_y", "width", "height"))
        view = View(entry["image_id"], entry["view_id"], x, y, width, height,
                    x + width, y + height, answers)
        expected_id = f"{x}_{y}_{width}_{height}"
        if view.view_id != expected_id:
            raise ValueError(f"View geometry/ID mismatch at {_view_key(view)}")
        saved_candidates = entry["candidates"]
        if len(saved_candidates) != len(view.candidates):
            raise ValueError(f"Candidate count differs at {_view_key(view)}")
        for k, (saved, reconstructed) in enumerate(zip(saved_candidates, view.candidates)):
            if (not np.array_equal(np.asarray(saved["points"], dtype=np.float64).reshape(-1, 2), reconstructed.points)
                    or int(saved["support_count"]) != reconstructed.support_count
                    or tuple(saved["source_worker_ids"]) != reconstructed.source_worker_ids):
                raise ValueError(f"Candidate order/content mismatch at {_view_key(view)} candidate {k}")
        views.append(view)
    heldout = tuple((entry["image_id"], entry["view_id"],
                     Answer(answer["worker_id"], np.asarray(answer["points"], dtype=np.float64).reshape(-1, 2)))
                    for entry in payload["heldout_views"] for answer in entry["answers"])
    return tuple(views), workers, heldout


def save_fit_npz(path: Path, fitted: Fit, prepared: Prepared, inputs_path: Path) -> dict:
    """Save the last accepted state, including all candidate dimensions and provenance."""
    accepted = [row for row in fitted.history if row.get("accepted")]
    last = accepted[-1] if accepted else None
    p = fitted.parameters
    # The heuristic initial q may not be its own posterior if time expires
    # before any update; persist q for the same omega/rho/pi as the objective.
    posterior_q = _posterior_q(_log_m(prepared, p), p.omega)
    metadata = {
        "format": "point_annotation_trial_fit_v1",
        "G": fitted.G, "initialization": fitted.initialization,
        "seed": 20261008 if fitted.initialization == "random_20261008" else None,
        "worker_ids": list(prepared.worker_ids),
        "view_order": [_view_key(view) for view in prepared.views],
        "candidate_order_sha256_by_view": [_candidate_order_digest(view) for view in prepared.views],
        "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "last_accepted_iteration": int(last["iteration"]) if last else 0,
        "objective": float(fitted.objective), "status": fitted.status,
        "stable_iterations": int(last["stable_iterations"]) if last else 0,
        "runtime_seconds": float(fitted.runtime_seconds),
        "failure_reason": fitted.history[-1].get("reason") if fitted.status == "numerical_failure" and fitted.history else None,
        "model_settings": {
            "q_count": Q_COUNT, "h": BANDWIDTH, "uniform_fraction": UNIFORM_FRACTION,
            "omega_prior": "Dirichlet(2,...,2)", "rho_prior": "Beta(2,2)",
            "pi_prior": "Dirichlet(1+1/K,...,1+1/K)",
            "density": "count mass times ordered coordinate density",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, omega=np.asarray(p.omega, dtype=np.float64),
                        q=np.asarray(posterior_q, dtype=np.float64),
                        rho=np.asarray(p.rho, dtype=np.float64),
                        pi=np.asarray(p.pi, dtype=np.float64),
                        k_per_view=np.asarray(prepared.k_per_view, dtype=np.int64),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
    return metadata


def load_fit_npz(path: Path, inputs_path: Path) -> tuple[Fit, Prepared, tuple[tuple[str, str, Answer], ...], dict]:
    """Reject mismatched order/content, then reconstruct a usable Fit."""
    views, workers, heldout = load_qc_inputs(inputs_path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        omega, q, rho, pi = (np.asarray(archive[name], dtype=np.float64) for name in ("omega", "q", "rho", "pi"))
        k_values = np.asarray(archive["k_per_view"], dtype=np.int64)
    if metadata.get("format") != "point_annotation_trial_fit_v1":
        raise ValueError("Unsupported saved fit format")
    if tuple(metadata["worker_ids"]) != workers:
        raise ValueError("QC worker order differs from saved fit")
    view_order = [_view_key(view) for view in views]
    if metadata["view_order"] != view_order:
        raise ValueError("QC view order differs from saved fit")
    digests = [_candidate_order_digest(view) for view in views]
    for index, (actual, expected) in enumerate(zip(digests, metadata["candidate_order_sha256_by_view"])):
        if actual != expected:
            raise ValueError(f"QC candidate order/content differs at {view_order[index]}")
    actual_hash = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    if actual_hash != metadata["inputs_sha256"]:
        raise ValueError(f"QC input SHA-256 mismatch: expected {metadata['inputs_sha256']}, found {actual_hash}")
    prepared = prepare(views, workers)
    if not np.array_equal(k_values, prepared.k_per_view):
        raise ValueError("QC candidate count per view differs from saved fit")
    G = int(metadata["G"])
    if (omega.shape != (G,) or q.shape != (len(workers), G) or rho.shape != (len(workers),)
            or pi.shape != (len(views), G, prepared.log_f.shape[-1])):
        raise ValueError("Saved fit array dimensions differ from QC inputs")
    if not all(np.isfinite(array).all() for array in (omega, q, rho, pi)):
        raise ValueError("Non-finite saved fit parameter")
    parameters = Parameters(omega, rho, pi, q)
    reconstructed_q = _posterior_q(_log_m(prepared, parameters), omega)
    if not np.allclose(q, reconstructed_q, rtol=1e-10, atol=1e-8):
        raise ValueError("Saved q does not match saved omega/rho/pi and QC inputs")
    fitted = Fit(G, metadata["initialization"], parameters, float(metadata["objective"]), [],
                 metadata["status"], float(metadata["runtime_seconds"]))
    return fitted, prepared, heldout, metadata
