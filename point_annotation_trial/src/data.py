"""Input records, exact observed candidates, and CSV/NPZ result output."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image

QC_IDS = tuple(f"QC{i}.png" for i in range(1, 6))
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "POINT_ANNOTATION_DATA_ROOT",
    Path(__file__).resolve().parents[2] / "private_data/by_image_tag",
)).expanduser()


@dataclass(frozen=True)
class Answer:
    worker_id: str
    points: np.ndarray  # Original, local pixel coordinates; repeated clicks remain.


@dataclass(frozen=True)
class Candidate:
    points: np.ndarray
    source_worker_ids: tuple[str, ...]

    @property
    def support_count(self) -> int:
        return len(self.source_worker_ids)


@dataclass(frozen=True)
class View:
    image_id: str
    view_id: str
    origin_x: int
    origin_y: int
    width: int
    height: int
    image_width: int
    image_height: int
    answers: tuple[Answer, ...]
    candidates: tuple[Candidate, ...] = field(init=False)
    observed_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        indices: dict[tuple, int] = {}
        points: list[np.ndarray] = []
        supporters: list[list[str]] = []
        observations: dict[str, int] = {}
        for answer in self.answers:
            if answer.worker_id in observations:
                raise ValueError(f"Duplicate answer for {answer.worker_id} in {self.image_id}/{self.view_id}")
            raw = np.asarray(answer.points, dtype=np.float64).reshape(-1, 2)
            key = tuple(sorted(map(tuple, raw)))
            if key not in indices:
                indices[key] = len(points)
                original = raw.copy()
                original.setflags(write=False)
                points.append(original)
                supporters.append([])
            k = indices[key]
            supporters[k].append(answer.worker_id)
            observations[answer.worker_id] = k
        if not points:
            raise ValueError("A view needs at least one recorded answer")
        object.__setattr__(self, "candidates", tuple(Candidate(p, tuple(s)) for p, s in zip(points, supporters)))
        object.__setattr__(self, "observed_index", observations)

    @property
    def normalized_candidates(self) -> tuple[np.ndarray, ...]:
        scale = np.array([self.width, self.height], dtype=np.float64)
        return tuple(c.points / scale for c in self.candidates)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def next_output_dir(requested: Path) -> Path:
    """Claim a fresh directory; never overwrite a previous run."""
    requested = requested.resolve()
    for number in range(10000):
        choice = requested if number == 0 else requested.with_name(f"{requested.name}_{number:03d}")
        try:
            choice.mkdir(parents=True, exist_ok=False)
            return choice
        except FileExistsError:
            continue
    raise RuntimeError(f"No free numbered output directory beside {requested}")


def resolve_data_root(requested: Path) -> Path | None:
    """Check the supplied location and only its immediate children."""
    requested = requested.expanduser().resolve()
    if all((requested / name).is_dir() for name in QC_IDS):
        return requested
    if not requested.is_dir():
        return None
    matches = [child for child in requested.iterdir() if child.is_dir() and all((child / name).is_dir() for name in QC_IDS)]
    return matches[0] if len(matches) == 1 else None


def worker_files(folder: Path) -> dict[str, Path]:
    prefix = folder.name + "_"
    found: dict[str, Path] = {}
    for path in sorted(folder.glob("*.json")):
        if not path.name.startswith(prefix):
            raise ValueError(f"Invalid worker filename: {path}")
        worker = unicodedata.normalize("NFC", path.name[len(prefix):-5])
        if not worker or worker in found:
            raise ValueError(f"Empty or duplicate worker ID: {path}")
        found[worker] = path
    if not found:
        raise ValueError(f"No worker JSON records: {folder}")
    return found


def select_qc_workers(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    recorded = [set(worker_files(root / image_id)) for image_id in QC_IDS]
    common = set.intersection(*recorded)
    if not common:
        raise ValueError("QC1–5 have no common recorded workers")
    return tuple(sorted(common)), tuple(sorted(set.union(*recorded) - common))


def _integer(value: object, label: str) -> int:
    number = float(value)
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{label} must be an integer")
    return int(number)


def load_image_views(
    root: Path, image_id: str, worker_ids: tuple[str, ...] | None = None,
    issues: list[dict] | None = None, mismatches: list[dict] | None = None,
    require_complete: bool = False,
) -> tuple[View, ...]:
    """Read only the requested workers. Invalid records are reported with provenance."""
    if Path(image_id).name != image_id:
        raise ValueError("image_id must be a basename")
    folder = root / image_id
    files = worker_files(folder)
    selected = tuple(sorted(files)) if worker_ids is None else tuple(sorted(worker_ids))
    problems = issues if issues is not None else []
    discrepancies = mismatches if mismatches is not None else []
    for worker in selected:
        if worker not in files:
            problems.append({"image_id": image_id, "worker_id": worker, "view_id": None,
                             "file": str(folder / f"{image_id}_{worker}.json"), "reason": "missing_worker_file"})
    try:
        with Image.open(folder / image_id) as image:
            image_width, image_height = image.size
    except Exception as exc:
        problems.append({"image_id": image_id, "worker_id": None, "view_id": None,
                         "file": str(folder / image_id), "reason": f"invalid_image: {exc}"})
        return ()
    records: dict[tuple[int, int, int, int], dict[str, Answer]] = {}
    for worker in selected:
        if worker not in files:
            continue
        path = files[worker]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or not payload:
                raise ValueError("expected a nonempty list of view records")
        except Exception as exc:
            problems.append({"image_id": image_id, "worker_id": worker, "view_id": None,
                             "file": str(path), "reason": f"invalid_json: {exc}"})
            continue
        for row_number, row in enumerate(payload):
            view_id = None
            try:
                crop = row["crop_coordinates"]
                x, y, width, height = (_integer(crop[key], key) for key in ("x", "y", "width", "height"))
                view_id = f"{x}_{y}_{width}_{height}"
                if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
                    raise ValueError("crop lies outside the image")
                raw_points = row["cell_coordinates"]["points"]
                if not isinstance(raw_points, list):
                    raise ValueError("an explicit points list is required")
                points = np.array([[p["x"], p["y"]] for p in raw_points], dtype=np.float64).reshape(-1, 2)
                if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > [width, height]):
                    raise ValueError("non-finite or out-of-view point coordinate")
                declared_count = _integer(row["cell_count"], "cell_count")
                if declared_count != len(points):
                    discrepancies.append({"image_id": image_id, "worker_id": worker, "view_id": view_id,
                                          "file": str(path), "record_index": row_number,
                                          "declared_cell_count": declared_count, "point_list_length": len(points)})
                points.setflags(write=False)
                geometry = (x, y, width, height)
                answers = records.setdefault(geometry, {})
                if worker in answers:
                    raise ValueError("duplicate worker/view record")
                answers[worker] = Answer(worker, points)
            except Exception as exc:
                problems.append({"image_id": image_id, "worker_id": worker, "view_id": view_id,
                                 "file": str(path), "record_index": row_number, "reason": str(exc)})
    views = []
    for (x, y, width, height), answers in sorted(records.items(), key=lambda item: (item[0][1], item[0][0])):
        view_id = f"{x}_{y}_{width}_{height}"
        if require_complete:
            for worker in selected:
                if worker not in answers:
                    problems.append({"image_id": image_id, "worker_id": worker, "view_id": view_id,
                                     "file": str(files.get(worker, folder / f"{image_id}_{worker}.json")),
                                     "reason": "missing_view_record"})
        views.append(View(image_id, view_id, x, y, width, height, image_width, image_height,
                          tuple(answers[w] for w in sorted(answers))))
    if not views:
        problems.append({"image_id": image_id, "worker_id": None, "view_id": None,
                         "file": str(folder), "reason": "no_readable_views"})
    return tuple(views)


def baseline_weights(view: View) -> np.ndarray:
    k = len(view.candidates)
    counts = np.array([c.support_count for c in view.candidates], dtype=np.float64)
    return (counts + 1.0 / k) / (len(view.answers) + 1.0)


def sample_point_sets(view: View, weights: np.ndarray, n_samples: int, seed: int) -> list[np.ndarray]:
    choice = np.random.default_rng(seed).choice(len(view.candidates), size=n_samples, p=weights)
    return [view.candidates[int(k)].points.copy() for k in choice]


def expected_count(view: View, weights: np.ndarray) -> float:
    return float(sum(float(w) * len(c.points) for w, c in zip(weights, view.candidates)))


def point_count_distribution(view: View, weights: np.ndarray) -> dict[int, float]:
    result: dict[int, float] = {}
    for candidate, weight in zip(view.candidates, weights):
        n = len(candidate.points)
        result[n] = result.get(n, 0.0) + float(weight)
    return dict(sorted(result.items()))


def expected_mass(view: View, weights: np.ndarray) -> np.ndarray:
    grid = np.zeros((view.height, view.width), dtype=np.float64)
    for candidate, probability in zip(view.candidates, weights):
        if not len(candidate.points):
            continue
        x = np.clip(candidate.points[:, 0], 0, view.width - 1)
        y = np.clip(candidate.points[:, 1], 0, view.height - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, view.width - 1), np.minimum(y0 + 1, view.height - 1)
        fx, fy = x - x0, y - y0
        for xx, yy, mass in ((x0, y0, (1-fx)*(1-fy)), (x1, y0, fx*(1-fy)),
                             (x0, y1, (1-fx)*fy), (x1, y1, fx*fy)):
            np.add.at(grid, (yy, xx), float(probability) * mass)
    if not np.isclose(grid.sum(), expected_count(view, weights), atol=1e-10, rtol=1e-12):
        raise RuntimeError(f"Mass is not conserved: {view.image_id}/{view.view_id}")
    return grid


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_image_outputs(
    views: tuple[View, ...], weights: tuple[np.ndarray, ...], output_dir: Path,
    methods: tuple[str, ...], statuses: tuple[str, ...], details: tuple[dict, ...] | None = None,
) -> dict:
    """Save whole observed candidates, their count distribution and mass maps."""
    if not views or len({v.image_id for v in views}) != 1 or not (len(views) == len(weights) == len(methods) == len(statuses)):
        raise ValueError("Expected aligned results for exactly one parent image")
    details = details or tuple({} for _ in views)
    if len(details) != len(views):
        raise ValueError("Missing view details")
    image_dir = output_dir / views[0].image_id
    image_dir.mkdir(parents=True, exist_ok=True)
    states, counts, summaries, maps = [], [], [], []
    global_mass = np.zeros((views[0].image_height, views[0].image_width), dtype=np.float64)
    coverage = np.zeros_like(global_mass, dtype=np.uint16)
    for view, values, method, status, detail in zip(views, weights, methods, statuses, details):
        probability = np.asarray(values, dtype=np.float64)
        if probability.shape != (len(view.candidates),) or not np.isfinite(probability).all() or np.any(probability < 0) or abs(float(probability.sum()) - 1) > 1e-10:
            raise ValueError(f"Invalid candidate weights: {view.image_id}/{view.view_id}")
        for k, (candidate, weight) in enumerate(zip(view.candidates, probability)):
            points = candidate.points
            states.append({"image_id": view.image_id, "view_id": view.view_id, "state_id": f"s{k:03d}",
                           "state_weight": float(weight), "estimate_method": method, "status": status,
                           "support_count": candidate.support_count,
                           "source_worker_ids": json.dumps(candidate.source_worker_ids, ensure_ascii=False),
                           "point_count": len(points),
                           "points_local": json.dumps(points.tolist()),
                           "points_normalized": json.dumps((points / [view.width, view.height]).tolist()),
                           "points_global": json.dumps((points + [view.origin_x, view.origin_y]).tolist()),
                           "view_origin_x": view.origin_x, "view_origin_y": view.origin_y,
                           "view_width": view.width, "view_height": view.height})
        count_distribution = point_count_distribution(view, probability)
        counts.extend({"image_id": view.image_id, "view_id": view.view_id, "point_count": n,
                       "probability": p, "estimate_method": method} for n, p in count_distribution.items())
        grid = expected_mass(view, probability)
        maps.append(grid)
        area = np.s_[view.origin_y:view.origin_y+view.height, view.origin_x:view.origin_x+view.width]
        global_mass[area] += grid
        coverage[area] += 1
        summaries.append({"image_id": view.image_id, "view_id": view.view_id,
                          "number_of_raw_answers": len(view.answers), "number_of_candidates": len(view.candidates),
                          "expected_point_count": expected_count(view, probability),
                          "estimate_method": method, "status": status,
                          "detail": json.dumps(detail, ensure_ascii=False)})
    np.divide(global_mass, coverage, out=global_mass, where=coverage > 0)
    write_csv(image_dir / "states.csv", states)
    write_csv(image_dir / "point_counts.csv", counts)
    write_csv(image_dir / "views.csv", summaries)
    np.savez_compressed(image_dir / "mass_maps.npz", view_ids=np.array([v.view_id for v in views]),
                        global_mean_mass=global_mass, coverage=coverage,
                        **{f"view_{i}": grid for i, grid in enumerate(maps)})
    return {"image_id": views[0].image_id, "number_of_views": len(views),
            "number_of_raw_answers": sum(len(v.answers) for v in views),
            "number_of_candidates": len(states),
            "methods": {method: methods.count(method) for method in sorted(set(methods))},
            "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
            "max_mass_error": max(abs(float(grid.sum()) - expected_count(v, w)) for v, w, grid in zip(views, weights, maps))}
