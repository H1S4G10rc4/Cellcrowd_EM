"""Data-derived local regions, retaining original display coverage.

Clustering defines calculation regions, never cells. Overlapping rectangles are
merged so a click cannot enter two regions. Within each region and worker, one
original displayed response is selected by geometry alone. Other overlapping
views are logged, not counted as independent extra votes and not stitched.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from configuration_model import Observation, Region, Settings, as_points, inside, point_key


def _intersect(a, b):
    c = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return c if c[0] < c[2] and c[1] < c[3] else None


def _area(a):
    return (a[2] - a[0]) * (a[3] - a[1])


def _radius(points, sigma):
    """Robust nearest-neighbor scale, with a four-bandwidth neighborhood floor.

    Four bandwidths keeps center/side alternatives in the same local region;
    this radius is not the location-error bandwidth or a cell-size estimate.
    """
    if len(points) < 2:
        return 4 * sigma
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    med = float(np.median(nearest))
    scale = max(med + 3 * float(np.median(abs(nearest - med))), float(np.quantile(nearest, .82)))
    return max(4 * sigma, min(scale, 2 * float(np.quantile(nearest, .9))))


def coarse_boxes(points, image_size, settings=Settings(), radius=None, coverage=(), linked_sets=()):
    """Coarse components in adaptive, bounded neighborhoods.

    A chain guard splits only neighborhoods wider than 16 fixed position
    bandwidths, at the widest data-derived coordinate gap. This is an explicit
    computational boundary, not a cell boundary; it is saved for inspection.
    The disjoint neighborhoods prevent rectangle merging from percolating
    across the entire image. No original 80-pixel grid is reused.
    """
    points = np.unique(as_points(points), axis=0)
    radius = _radius(points, settings.sigma_pixels) if radius is None else float(radius)
    max_extent = 16 * settings.sigma_pixels
    leaves, splits = [], []
    def divide(p, domain):
        if not len(p):
            return
        span = np.ptp(p, axis=0) + 2 * settings.sigma_pixels
        axis = int(np.argmax(span))
        proposed = (max(domain[0], p[:, 0].min() - settings.sigma_pixels),
                    max(domain[1], p[:, 1].min() - settings.sigma_pixels),
                    min(domain[2], p[:, 0].max() + settings.sigma_pixels),
                    min(domain[3], p[:, 1].max() + settings.sigma_pixels))
        covered = not coverage or any(b[0] <= proposed[0] and b[1] <= proposed[1]
                                      and b[2] >= proposed[2] and b[3] >= proposed[3] for b in coverage)
        if (span[axis] <= max_extent and covered) or len(p) == 1:
            leaves.append((p, domain))
            return
        values = np.unique(p[:, axis])
        centers = (values[:-1] + values[1:]) / 2
        gaps = np.diff(values)
        eligible = np.flatnonzero(np.isclose(gaps, gaps.max(), rtol=1e-12, atol=1e-12))
        # A forced balanced cut can pass through a two-point object even when
        # a larger empty gap separates entire fields nearby. Balance only ties.
        index = eligible[np.argmin(abs(centers[eligible] - (values[0]+values[-1])/2))]
        cut = float(centers[index])
        first, second = list(domain), list(domain)
        first[axis + 2], second[axis] = cut, cut
        splits.append({"axis": axis, "coordinate": cut, "domain": list(domain)})
        left = p[:, axis] < cut
        divide(p[left], tuple(first))
        divide(p[~left], tuple(second))
    divide(points, (0., 0., float(image_size[0]), float(image_size[1])))
    boxes, components, merges = [], 0, 0
    for p, domain in leaves:
        local, audit = _component_boxes(p, image_size, settings, radius, domain, linked_sets)
        boxes.extend(local)
        components += audit["components"]
        merges += audit["rectangle_merges"]
    return sorted(boxes, key=lambda b: (b[1], b[0])), {
        "radius_pixels": radius, "components": components, "rectangle_merges": merges,
        "regions": len(boxes), "max_neighborhood_extent_pixels": max_extent,
        "chain_guard_splits": splits}


def _component_boxes(points, image_size, settings, radius, domain, linked_sets):
    """Connected components followed by rectangle-overlap merging. No pruning."""
    points = np.unique(as_points(points), axis=0)
    if not len(points):
        return [], {"radius_pixels": None, "components": 0, "rectangle_merges": 0}
    radius = _radius(points, settings.sigma_pixels) if radius is None else float(radius)
    if radius <= 0:
        raise ValueError("A positive coarse radius is required")
    parent = np.arange(len(points))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in cKDTree(points).query_pairs(radius):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    # A bounded neighborhood is a calculation unit, not a pre-decided cell.
    # Preserve within-response co-occurrence even if the two points have no
    # intervening clicks. Otherwise a clean two-point answer would be split
    # into two single-point truths before its count errors can be estimated.
    lookup = {tuple(p): i for i,p in enumerate(points)}
    for s in linked_sets:
        ids = [lookup[tuple(p)] for p in s if tuple(p) in lookup]
        for i in ids[1:]:
            a,b = root(ids[0]),root(i)
            parent[max(a,b)] = min(a,b)
    groups = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    width, height = image_size
    pad = settings.sigma_pixels
    boxes = []
    for ids in groups.values():
        p = points[ids]
        lo, hi = p.min(axis=0) - pad, p.max(axis=0) + pad
        boxes.append((max(domain[0], lo[0]), max(domain[1], lo[1]),
                      min(domain[2], hi[0]), min(domain[3], hi[1])))
    original = len(boxes)
    # At most a few hundred components per image. Repeated rectangle union
    # avoids arbitrary grid boundaries and guarantees disjoint response domains.
    changed = True
    while changed:
        changed = False
        merged = []
        for b in boxes:
            i = 0
            while i < len(merged):
                a = merged[i]
                if _intersect(a, b) is not None:
                    b = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    merged.pop(i)
                    changed = True
                    i = 0
                else:
                    i += 1
            merged.append(b)
        boxes = merged
    return sorted(boxes, key=lambda b: (b[1], b[0])), {
        "radius_pixels": radius, "components": original,
        "rectangle_merges": original - len(boxes), "regions": len(boxes)}


def make_candidates(observations, bounds, include_proposals=True):
    """Keep observed multisets; add explicitly labeled one/two-center proposals.

    The two-center proposal is a deterministic Lloyd fit initialized at the
    farthest pair. It proposes a whole configuration, not independent marginal
    selections. No generated truth, worker class, or expert answer is supplied.
    """
    candidates, sources, keys = [], [], {}
    def add(points, source):
        key = point_key(points)
        if key not in keys:
            keys[key] = len(candidates)
            candidates.append(as_points(points))
            sources.append(source)
    add([], "empty_hypothesis")
    for a in observations:
        complete = (a.bounds[0] <= bounds[0] and a.bounds[1] <= bounds[1]
                    and a.bounds[2] >= bounds[2] and a.bounds[3] >= bounds[3])
        add(a.points, "observed_configuration" if complete else "partial_response_proposal")
    points = np.concatenate([a.points for a in observations], axis=0) if observations else np.empty((0, 2))
    if include_proposals and len(points):
        add(points.mean(axis=0, keepdims=True), "one_center_proposal")
        unique = np.unique(points, axis=0)
        if len(unique) >= 2:
            dist = np.square(unique[:, None] - unique[None]).sum(axis=2)
            a, b = np.unravel_index(np.argmax(dist), dist.shape)
            centers = unique[[a, b]].copy()
            for _ in range(100):
                assignments = np.square(points[:, None] - centers[None]).sum(axis=2).argmin(axis=1)
                if not all(np.any(assignments == k) for k in range(2)):
                    break
                updated = np.stack([points[assignments == k].mean(axis=0) for k in range(2)])
                if np.allclose(updated, centers, atol=1e-10, rtol=0):
                    centers = updated
                    break
                centers = updated
            add(centers, "two_center_proposal")
    return tuple(candidates), tuple(sources)


def regions_from_views(views, settings=Settings(), include_proposals=True, radius=None):
    """Pool coordinates for regions; keep original responses for inference.

    Fully visible regions are preferred. Otherwise choose the largest visible
    intersection, retaining a partial-view flag. Missing displays never become
    empty answers. All original input remains external and unchanged.
    """
    by_image = {}
    for v in views:
        by_image.setdefault(v.image_id, []).append(v)
    regions, audits = [], []
    for image_id, image_views in sorted(by_image.items()):
        linked_sets = tuple(a.points + [v.origin_x, v.origin_y] for v in image_views for a in v.answers)
        all_points = np.concatenate(linked_sets, axis=0)
        coverage = tuple((v.origin_x, v.origin_y, v.origin_x + v.width, v.origin_y + v.height) for v in image_views)
        boxes, audit = coarse_boxes(all_points, (image_views[0].image_width, image_views[0].image_height),
                                   settings, radius, coverage, linked_sets)
        audit.update({"image_id": image_id, "original_answers": sum(len(v.answers) for v in image_views),
                      "original_clicks": len(all_points), "selected_region_answers": 0,
                      "selected_clicks": 0, "partial_region_answers": 0,
                      "overlap_alternatives_not_counted": 0, "regions_wider_than_display": 0})
        choices = []
        for r, bounds in enumerate(boxes):
            available = {}
            for v in image_views:
                vb = (v.origin_x, v.origin_y, v.origin_x + v.width, v.origin_y + v.height)
                visible = _intersect(bounds, vb)
                if visible is None:
                    continue
                complete = np.allclose(visible, bounds, atol=1e-10, rtol=0)
                margin = min(bounds[0] - vb[0], bounds[1] - vb[1], vb[2] - bounds[2], vb[3] - bounds[3])
                rank = (-_area(visible), -margin, v.view_id)
                for a in v.answers:
                    absolute = a.points + [v.origin_x, v.origin_y]
                    selected = absolute[inside(absolute, bounds)]
                    # Positions are normalized over the original display, not
                    # over a tight data-dependent box around the observed
                    # clicks. Truncating at that artificial box rewards outer
                    # candidate points simply for being near its boundary.
                    available.setdefault(a.worker_id, []).append((rank, Observation(a.worker_id, selected, vb, v.view_id), complete))
            observations = []
            for worker, options in sorted(available.items()):
                _, selected, complete = min(options, key=lambda x: x[0])
                observations.append(selected)
                audit["partial_region_answers"] += int(not complete)
                audit["overlap_alternatives_not_counted"] += len(options) - 1
                choices.append({"region_id": str(r), "worker_id": worker, "view_id": selected.view_id,
                                "display_bounds": list(selected.bounds), "full_coverage": bool(complete),
                                "alternatives_not_counted": len(options) - 1, "point_count": len(selected.points)})
            candidates, sources = make_candidates(tuple(observations), bounds, include_proposals)
            regions.append(Region(image_id, str(r), bounds, tuple(observations), candidates, sources))
            audit["selected_region_answers"] += len(observations)
            audit["selected_clicks"] += sum(len(a.points) for a in observations)
            audit["regions_wider_than_display"] += int(bounds[2] - bounds[0] > max(v.width for v in image_views)
                                                       or bounds[3] - bounds[1] > max(v.height for v in image_views))
        audit["selections"] = choices
        audits.append(audit)
    return tuple(regions), audits


def save_regions(path, regions):
    payload = []
    for r in regions:
        payload.append({"image_id": r.image_id, "region_id": r.region_id, "bounds": list(r.bounds),
            "observations": [{**asdict(a), "points": a.points.tolist()} for a in r.observations],
            "candidates": [s.tolist() for s in r.candidates], "candidate_sources": list(r.candidate_sources)})
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_regions(path):
    payload = json.loads(Path(path).read_text())
    return tuple(Region(r["image_id"], r["region_id"], tuple(r["bounds"]),
                        tuple(Observation(a["worker_id"], a["points"], tuple(a["bounds"]), a["view_id"])
                              for a in r["observations"]),
                        tuple(r["candidates"]), tuple(r["candidate_sources"])) for r in payload)
