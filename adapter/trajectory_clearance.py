"""Conservative pre-write clearance for generated humanoid root motion."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


class HumanoidTrajectoryBlocked(ValueError):
    """The generated root path cannot safely contain the humanoid body."""

    def __init__(self, blockers: list[dict[str, Any]]):
        self.blockers = blockers
        first = blockers[0]
        super().__init__(
            "generated humanoid trajectory intersects "
            f"{first['object_id']} near segment {first['segment_index']}"
        )


def validate_humanoid_trajectory(
    root_positions: Sequence[Sequence[float]],
    skin_vertices: Any,
    geometry: Iterable[dict[str, Any]],
    *,
    margin_m: float = 0.08,
) -> dict[str, Any]:
    """Check every root segment against expanded static obstacle bounds.

    This is deliberately conservative: the maximum bind-mesh X/Z radius is
    swept along the authored root polyline. It can reject a visually clear
    pose, but it must not bless a path that crosses furniture or walls.
    """
    roots = np.asarray(root_positions, dtype=float)
    vertices = np.asarray(skin_vertices, dtype=float)
    if (roots.ndim != 2 or roots.shape[1] != 3 or not len(roots)
            or not np.isfinite(roots).all()):
        raise ValueError("root_positions must be finite Nx3 values")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError("skin_vertices must be finite Nx3 values")
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and non-negative")

    radius = float(np.linalg.norm(vertices[:, (0, 2)], axis=1).max()) + margin_m
    local_y_min = float(vertices[:, 1].min()) - margin_m
    local_y_max = float(vertices[:, 1].max()) + margin_m
    blockers: list[dict[str, Any]] = []
    obstacle_count = 0
    for item in geometry:
        bounds = _bounds(item)
        if bounds is None or _is_support_floor(item, bounds, roots):
            continue
        obstacle_count += 1
        low, high = bounds
        for index, (start, end) in enumerate(zip(roots, roots[1:])):
            path_low_y = min(start[1], end[1]) + local_y_min
            path_high_y = max(start[1], end[1]) + local_y_max
            if path_high_y < low[1] or path_low_y > high[1]:
                continue
            if _segment_intersects_expanded_box(start, end, low, high, radius):
                blockers.append({
                    "object_id": str(item.get("object_id") or item.get("name") or "unknown"),
                    "name": str(item.get("name") or item.get("object_id") or "unknown"),
                    "segment_index": index,
                    "segment_start": start.tolist(),
                    "segment_end": end.tolist(),
                    "bounds_min": low.tolist(),
                    "bounds_max": high.tolist(),
                    "body_radius_m": radius,
                })
                break
    if blockers:
        raise HumanoidTrajectoryBlocked(blockers)
    return {
        "status": "clear_at_root_sweep",
        "root_samples": int(len(roots)),
        "segments": max(0, int(len(roots) - 1)),
        "obstacles_checked": obstacle_count,
        "body_radius_m": radius,
        "margin_m": float(margin_m),
        "continuous_collision_certified": False,
        "scope": "conservative bind-mesh radial root sweep against static scene bounds",
    }


def _bounds(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(item.get("points") or (), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    if not np.isfinite(points).all():
        raise ValueError("scene geometry contains nonfinite points")
    return points.min(axis=0), points.max(axis=0)


def _is_support_floor(
    item: dict[str, Any],
    bounds: tuple[np.ndarray, np.ndarray],
    roots: np.ndarray,
) -> bool:
    label = f"{item.get('object_id', '')} {item.get('name', '')}".lower()
    low, high = bounds
    support_y = float(np.min(roots[:, 1]))
    return (
        ("floor" in label or "rug" in label or "carpet" in label)
        and high[1] <= support_y + 0.02
        and low[1] <= support_y + 0.001
    )


def _segment_intersects_expanded_box(
    start: np.ndarray,
    end: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    radius: float,
) -> bool:
    """Liang-Barsky intersection in X/Z against a radius-expanded AABB."""
    t_min, t_max = 0.0, 1.0
    delta = end - start
    for axis in (0, 2):
        minimum = float(low[axis] - radius)
        maximum = float(high[axis] + radius)
        if abs(float(delta[axis])) <= 1e-12:
            if float(start[axis]) < minimum or float(start[axis]) > maximum:
                return False
            continue
        a = (minimum - float(start[axis])) / float(delta[axis])
        b = (maximum - float(start[axis])) / float(delta[axis])
        enter, leave = sorted((a, b))
        t_min, t_max = max(t_min, enter), min(t_max, leave)
        if t_min > t_max:
            return False
    return True
