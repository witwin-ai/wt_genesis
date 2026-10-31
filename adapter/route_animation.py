"""A Kimodo person that walks a planned route and ends in a pose.

The agent could only give ``generate_humanoid_animation`` a text prompt and a
start position, so "walk to the sofa and sit" never worked (the audit saw the
bed's coordinates written into the prompt). This composes the pieces the
meeting samples were made with, in their order, behind one call:

1. the route and its end pose as Kimodo constraints
   (:mod:`.kimodo_path`, the ``Hips`` pin of :mod:`.kimodo_poses`), the
   settled pose grounded on its support;
2. the feet cleaned once more on the grounded clip (:mod:`.foot_lock`; the
   clip as it was kept beside it as ``.prelock.npz``);
3. a lively rest from the clip's last pose to the scene's clip end
   (:mod:`.rest_idle`: upper body only, pelvis and legs held);
4. the binding, which bakes the final clip with the pelvis-pivot root,
   densified Euler keys and the static pose at the first key (what
   motion matching's ``tools/rebind_pelvis_pivot.py`` re-baked kept samples
   with); the pivot is checked on the baked root track.

Route and pose actions need the Kimodo backend (root and end-effector
constraints); MDM only acts in place, and :func:`require_route_backend` says
so instead of approximating the route with text.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .kimodo_path import PathAnimationResult, create_humanoid_path_animation
from .trajectory_clearance import HumanoidTrajectoryBlocked

#: How a route can end.
END_POSES = ("stand", "sit", "lie", "fall")
#: Seconds a pose is held after the walk (Kimodo's time to turn and sit,
#: lie down, fall), as the meeting bench used them.
END_HOLD_S = {"stand": 1.5, "sit": 3.0, "lie": 5.0, "fall": 3.0}
DEFAULT_PROMPTS = {
    "stand": "a person walks",
    "sit": "a person walks over, turns around and sits down",
    "lie": "a person walks to the bed, sits down on it and lies down on the back",
    "fall": "a person walks, trips and falls down to the floor",
}
#: Profile cap on the generated motion: Kimodo breaks up past ~12 s (the
#: root jumped 0.3-0.5 m per frame).
MAX_MOTION_S = 10.0
#: A baked root key may differ from translation + J0 - R J0 by this.
PIVOT_TOLERANCE_M = 2e-3
#: A pose is settled once the root stays this close to its final height.
SETTLED_M = 0.03
CONTACT_EVENTS = {"sit": "seat_contact", "lie": "back_contact", "fall": "impact"}
#: Takes are refused (and the next seed tried) past these, as in the
#: meeting bench: the fastest root step, the route error while walking, a
#: root rising (or, walking, dropping) from where it stood, a settled pose
#: off its support, a seated person's soles off the floor.
MAX_ROOT_SPEED_MPS = 3.0
MAX_PATH_ERROR_M = 0.15
MAX_ROOT_RISE_M = 0.15
END_POSE_SUPPORT_TOLERANCE_M = 0.02
SEATED_FEET_TOLERANCE_M = 0.025
#: Kimodo takes per route (each with the next seed).
ATTEMPTS = 2

#: Where a seated pelvis is pinned over the seat top (SOMA -> SMPL: at 0.13 m
#: the buttocks came 5-10 cm low), and how far it sits in from the front edge.
SIT_PELVIS_ABOVE_SEAT_M = 0.24
SEAT_SIT_DEPTH_M = 0.16
#: A seat whose top is higher than this leaves the feet off the floor.
SEAT_MAX_HEIGHT_M = 0.47
#: The walk ends in front of the seat (the bench's approach is 0.45 m).
SEAT_MAX_APPROACH_M = 1.0
#: Seat and mattress contact the body may take as support, not collision.
SEAT_SUPPORT_PENETRATION_M = 0.10
BED_SUPPORT_PENETRATION_M = 0.15
#: A lying pelvis: 0.20 m over the mattress, 0.95 m from the head end.
LIE_PELVIS_ABOVE_TOP_M = 0.20
LIE_PELVIS_FROM_HEAD_M = 0.95
#: Kimodo's pelvis pin covers the last seconds of a sit or lie.
POSE_PIN_S = 1.5
#: A fall needs this much floor straight ahead of the last step, as wide as
#: the body.
FALL_REACH_M = 1.8
FALL_BODY_RADIUS_M = 0.3
#: Geometry this low over the floor is stepped on, not fallen against.
FLOOR_ITEM_M = 0.05
#: A seat's top is sampled on this grid round the sit point.
TOP_SAMPLE_HALF_M = 0.15
TOP_SAMPLE_POINTS = 5


class RouteBackendUnavailable(RuntimeError):
    """The active person backend cannot follow a route or hold an end pose."""


class RouteTargetUnsupported(ValueError):
    """The end pose cannot be placed on the requested target from this route."""


class RouteGenerationRejected(RuntimeError):
    """Every Kimodo take was refused (:func:`generation_refusal`)."""

    def __init__(self, attempts: list[dict[str, Any]]):
        self.attempts = attempts
        reasons = "; ".join(f"seed {row['seed']}: {row.get('reason')}" for row in attempts)
        super().__init__(f"no kept take after {len(attempts)} attempt(s): {reasons}")


class FallZoneBlocked(ValueError):
    """Furniture or a wall stands where the falling body would land."""

    def __init__(self, blockers: list[dict[str, Any]]):
        self.blockers = blockers
        names = ", ".join(str(item["object_id"]) for item in blockers)
        super().__init__(f"no clear floor to fall on: {names} within {FALL_REACH_M} m "
                         "ahead of the last step")


def require_route_backend(backend_name: str) -> None:
    """Raise :class:`RouteBackendUnavailable` unless ``backend_name`` is Kimodo."""
    if backend_name != "kimodo":
        raise RouteBackendUnavailable(
            f"route and end-pose actions need the Kimodo backend; the active backend is "
            f"{backend_name}, which only generates in-place motion from text "
            "(set WITWIN_GENESIS_BACKEND=kimodo)")


@dataclass(frozen=True)
class RouteAnimationResult:
    path: PathAnimationResult
    end: str
    key_events: dict[str, float]
    rest: dict[str, Any]
    foot_lock: dict[str, Any] | None
    binding: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    end_pose: dict[str, Any] | None = None
    target: dict[str, Any] | None = None

    @property
    def body_object_id(self) -> str:
        return self.path.animation.body_object_id

    @property
    def interval_s(self) -> tuple[float, float]:
        start = float(self.path.animation.start_time)
        return start, start + float(self.path.animation.duration)

    def to_dict(self) -> dict[str, Any]:
        payload = self.path.to_dict()
        start, end = self.interval_s
        payload.update(
            end=self.end, body_object_id=self.body_object_id,
            interval_s=[round(start, 4), round(end, 4)],
            route={"waypoints": payload["plan"]["waypoints"], "length_m": payload["plan"]["length_m"],
                   "arrival_s": payload["plan"]["arrival_s"]},
            key_events={name: round(time, 4) for name, time in self.key_events.items()},
            rest=self.rest, foot_lock=self.foot_lock, binding=self.binding,
            attempts=list(self.attempts), end_pose=self.end_pose, target=self.target,
        )
        return payload


def create_route_animation(
    *,
    scene: Any,
    project_dir: str | Path,
    waypoints: Sequence[Sequence[float]],
    end: str = "stand",
    target_object_id: str | None = None,
    sit_at: Sequence[float] | None = None,
    head_direction: Sequence[float] | None = None,
    support_height_m: float | None = None,
    geometry_reader: Callable[[Any], Sequence[dict[str, Any]]] | None = None,
    prompt: str | None = None,
    style_hint: str | None = None,
    speed_mps: float = 1.0,
    start_hold_s: float = 0.5,
    end_hold_s: float | None = None,
    end_heading: float | None = None,
    rest_until_s: float | str | None = "clip_end",
    rest_style: str = "lively",
    foot_lock: bool = True,
    max_motion_s: float = MAX_MOTION_S,
    attempts: int = ATTEMPTS,
    seed: int = 10,
    device: int = 0,
    output_name: str | None = None,
    body_name: str | None = None,
    start_time: float | None = None,
    floor_y: float | None = None,
    body_validator: Callable[[Any, str, float, float], dict[str, Any]] | None = None,
    generator: Callable[..., dict[str, Any]] | None = None,
    skin: Any = None,
    **path_kwargs: Any,
) -> RouteAnimationResult:
    """Generate a Kimodo person walking ``waypoints`` that ends in ``end``.

    ``rest_until_s`` is the timeline time the person rests alive to after
    its clip ends: ``"clip_end"`` (the scene's clip end when other actors
    are already on the timeline), a time, or None (the clip ends with the
    motion). ``foot_lock`` cleans the grounded clip's feet once more on the
    licensed SMPL ``skin``.

    ``end`` sit and lie rest on ``target_object_id``; its footprint and top
    come from ``geometry_reader(scene)`` (scene_geometry items), and the
    route must end in front of it. A sit's place (``sit_at``) and facing
    default to the seat edge nearest the last waypoint; a lie needs the
    bed's ``head_direction`` (x, z). A fall needs clear floor ahead of the
    last step. ``support_height_m`` (above the floor) overrides the
    measured top. Refusals happen before Kimodo runs.

    Up to ``attempts`` takes run, with seed ``seed``, ``seed + 1``, ...; a
    take :func:`generation_refusal` refuses (or whose body is blocked) is
    removed from the scene before the next. Every take is listed in the
    result's ``attempts``, or in :class:`RouteGenerationRejected`'s.
    """
    if end not in END_POSES:
        raise ValueError(f"end must be one of {END_POSES}, not {end!r}")
    if floor_y is None:
        first = [float(value) for value in waypoints[0]]
        floor_y = first[1] if len(first) == 3 else 0.0
    pose_kwargs, target = plan_end_pose(
        end, scene, waypoints, target_object_id=target_object_id, sit_at=sit_at,
        end_heading=end_heading, head_direction=head_direction,
        support_height_m=support_height_m, floor_y=float(floor_y),
        geometry_reader=geometry_reader)
    begin = float(start_time or 0.0)
    until = _rest_until(scene, rest_until_s)
    text = _prompt(end, prompt, style_hint)
    hold = float(END_HOLD_S[end] if end_hold_s is None else end_hold_s)
    if int(attempts) < 1:
        raise ValueError("attempts must be at least 1")
    rows: list[dict[str, Any]] = []
    failure: Exception | None = None
    for attempt in range(int(attempts)):
        take_seed = int(seed) + attempt
        before = set(getattr(scene, "objects", {}) or {})
        timeline = getattr(scene, "timeline_manager", None)
        duration = float(timeline.duration) if timeline is not None else None
        try:
            made = _take(
                scene, take_seed, end=end, begin=begin, until=until, rest_style=rest_style,
                foot_lock=foot_lock, skin=skin, project_dir=project_dir, prompt=text,
                waypoints=waypoints, floor_y=floor_y, speed_mps=speed_mps,
                start_hold_s=start_hold_s, end_hold_s=hold, max_duration_s=max_motion_s,
                device=device,
                output_name=None if output_name is None else f"{output_name}_{attempt}",
                body_name=body_name, start_time=start_time, body_validator=body_validator,
                generator=generator, **pose_kwargs, **path_kwargs)
        except HumanoidTrajectoryBlocked as error:
            _remove_new_objects(scene, before, duration)
            rows.append({"seed": take_seed, "status": "blocked", "reason": str(error)})
            failure = error
            continue
        reason = generation_refusal(made.path.path_following, made.path.occupancy,
                                    made.path.end_pose_grounding, end)
        if reason is None and made.binding.get("pivot") == "not_pivoted":
            reason = ("the baked root does not turn about the SMPL pelvis "
                      f"({made.binding['max_error_m']} m off)")
        if reason is not None:
            _remove_new_objects(scene, before, duration)
            rows.append({"seed": take_seed, "status": "rejected", "reason": reason})
            failure = None
            continue
        rows.append({"seed": take_seed, "status": "kept"})
        return replace(made, attempts=rows, target=target)
    if isinstance(failure, HumanoidTrajectoryBlocked):
        failure.attempts = rows
        raise failure
    raise RouteGenerationRejected(rows)


def _take(scene, seed, *, end, begin, until, rest_style, foot_lock, skin, **kwargs):
    """One Kimodo take: generate, clean the feet, rest, bind (no refusal)."""
    reports: dict[str, Any] = {}

    def prepare(motion_asset, plan):
        reports["foot_lock"] = _clean_feet(motion_asset, skin) if foot_lock else None
        reports["settled"] = _settled_after(motion_asset, plan) if end != "stand" else None
        motion_asset, frames = _rest_to(motion_asset, begin, until, seed=seed, style=rest_style)
        reports["rest_frames"] = frames
        return motion_asset

    made = create_humanoid_path_animation(scene=scene, seed=seed, skin=skin,
                                          prepare_motion=prepare, **kwargs)
    animation = made.animation
    held = 0.0
    if until is not None:
        held = _hold_tracks_to(scene, animation.body_object_id, until)
    events = {"arrival": float(animation.start_time) + made.plan.arrival_s}
    if reports.get("settled") is not None:
        events[CONTACT_EVENTS[end]] = float(animation.start_time) + reports["settled"]
    return RouteAnimationResult(
        path=made, end=end, key_events=events,
        rest={"until_s": until, "frames": int(reports.get("rest_frames") or 0),
              "style": rest_style, "held_s": round(held, 4)},
        foot_lock=reports.get("foot_lock"),
        binding=pivot_binding_check(scene, animation.body_object_id, animation.motion_asset,
                                    float(animation.start_time)),
        end_pose=made.end_pose_grounding,
    )


def generation_refusal(following: dict[str, Any], occupancy: dict[str, Any] | None,
                       grounding: dict[str, Any] | None, end: str) -> str | None:
    """Why a generated take cannot be kept, or None: the root jumps (Kimodo
    past ~12 s: 11 m/s), strays from the route, floats up or sinks while
    walking, or the settled pose is off its support (the meeting bench's
    refusals)."""
    speed = float(following.get("max_root_speed_mps") or 0.0)
    if speed > MAX_ROOT_SPEED_MPS:
        return f"motion glitch: root moves at {speed:.1f} m/s"
    error = float(following.get("timed_max_error_m") or 0.0)
    if error > MAX_PATH_ERROR_M:
        return f"strays {error:.2f} m from the planned route"
    heights = [float(sample["y"]) for sample in (occupancy or {}).get("samples") or ()]
    if heights:
        rise = max(heights) - heights[0]
        if rise > MAX_ROOT_RISE_M:
            return f"motion glitch: root rises {rise:.2f} m above where it stood"
        if end == "stand" and heights[0] - min(heights) > MAX_ROOT_RISE_M:
            return f"motion glitch: root drops {heights[0] - min(heights):.2f} m while walking"
    if grounding:
        gap = grounding.get("rest_support_gap_m")
        if gap is not None and abs(float(gap)) > END_POSE_SUPPORT_TOLERANCE_M:
            where = "over" if float(gap) > 0 else "into"
            return (f"{grounding.get('kind')}: rests {abs(float(gap)) * 100:.1f} cm {where} "
                    "its support")
        feet = grounding.get("rest_feet_lowest_m")
        if grounding.get("kind") == "sit" and feet is not None \
                and float(feet) > SEATED_FEET_TOLERANCE_M:
            return f"sit: feet {float(feet) * 100:.1f} cm off the floor"
    return None


def _remove_new_objects(scene, before, duration) -> None:
    """Take a refused take's objects and their tracks out of the scene."""
    added = set(getattr(scene, "objects", {}) or {}) - set(before)
    timeline = getattr(scene, "timeline_manager", None)
    if timeline is not None:
        for object_id in added:
            timeline.remove_object_tracks(object_id)
        if duration is not None:
            timeline.duration = duration
    for object_id in added:
        if object_id in scene.objects:
            scene.remove_object(object_id)


def plan_end_pose(end: str, scene: Any, waypoints: Sequence[Sequence[float]], *,
                  target_object_id: str | None = None, sit_at: Sequence[float] | None = None,
                  end_heading: float | None = None, head_direction: Sequence[float] | None = None,
                  support_height_m: float | None = None, floor_y: float = 0.0,
                  geometry_reader: Callable[[Any], Sequence[dict[str, Any]]] | None = None,
                  ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(create_humanoid_path_animation kwargs, target report)`` for ``end``
    (:func:`create_route_animation`), or :class:`RouteTargetUnsupported` /
    :class:`FallZoneBlocked`."""
    from .kimodo_path import _ground_points, heading_from_direction
    from .kimodo_poses import (BACK_REGION_JOINTS, SEAT_REGION_JOINTS, EndPose,
                               supine_pelvis, upright_pelvis)

    points = _ground_points(waypoints)
    last = points[-1]
    if end == "stand":
        return {"end_heading": end_heading}, None
    geometry = list(geometry_reader(scene)) if geometry_reader is not None else None
    if end == "fall":
        zone = _fall_zone(points, geometry, floor_y)
        return ({"end_heading": None,
                 "end_pose": EndPose(kind="fall", support_height=0.0, region_joints=None)},
                {"fall_zone": zone})
    if not target_object_id:
        raise RouteTargetUnsupported(f"{end} needs target_object_id (the {_support_word(end)})")
    item = _target_item(geometry, str(target_object_id))
    rect = _footprint(item)
    x0, z0, x1, z1 = rect
    if x0 < last[0] < x1 and z0 < last[1] < z1:
        raise RouteTargetUnsupported(
            f"the route ends inside {target_object_id}; end it on the floor in front of it")
    report: dict[str, Any] = {"object_id": str(target_object_id)}
    if end == "sit":
        edge = np.array([min(max(last[0], x0), x1), min(max(last[1], z0), z1)])
        gap = float(np.linalg.norm(last - edge))
        if gap > SEAT_MAX_APPROACH_M:
            raise RouteTargetUnsupported(
                f"the route ends {gap:.2f} m from the seat {target_object_id}; end it in front "
                f"of the seat edge (within {SEAT_MAX_APPROACH_M} m)")
        if sit_at is None:
            front = (last - edge) / max(gap, 1e-9)
            inward = min(SEAT_SIT_DEPTH_M, _depth(rect, edge, -front))
            seat = edge - front * inward
        else:
            seat = _ground_points([last, sit_at])[1]
        if end_heading is None:
            end_heading = heading_from_direction(last - seat)
        height, source = _support_height(item, seat, support_height_m, floor_y)
        if height > SEAT_MAX_HEIGHT_M:
            raise RouteTargetUnsupported(
                f"the seat {target_object_id} is {height:.2f} m high, too high for the feet to "
                f"reach the floor (at most {SEAT_MAX_HEIGHT_M} m)")
        pose = EndPose(kind="sit", hips=(float(seat[0]), height + SIT_PELVIS_ABOVE_SEAT_M,
                                         float(seat[1])),
                       rotvec=tuple(upright_pelvis(end_heading)), hold_s=POSE_PIN_S,
                       support_height=height, region_joints=SEAT_REGION_JOINTS,
                       support_rect=rect, plant_feet=True)
        report.update(support_height_m=round(height, 4), height_source=source,
                      support_rect=[round(value, 4) for value in rect],
                      sit_at=[round(float(value), 4) for value in seat],
                      end_heading=float(end_heading))
        return ({"end_pose": pose, "sit_at": tuple(float(value) for value in seat),
                 "end_heading": float(end_heading), "sit_support_ids": [str(target_object_id)],
                 "max_seat_penetration_m": SEAT_SUPPORT_PENETRATION_M}, report)
    # end == "lie"
    if head_direction is None:
        raise RouteTargetUnsupported(
            "lie needs head_direction (x, z): the way the bed's head end points from its centre")
    head = np.asarray([float(value) for value in head_direction][:2], dtype=np.float64)
    if not np.isfinite(head).all() or np.linalg.norm(head) < 1e-9:
        raise RouteTargetUnsupported("head_direction must be a non-zero (x, z) direction")
    head = head / np.linalg.norm(head)
    centre = np.array([(x0 + x1) / 2.0, (z0 + z1) / 2.0])
    head_end = centre + head * _depth(rect, centre, head)
    pelvis = head_end - head * LIE_PELVIS_FROM_HEAD_M
    height, source = _support_height(item, pelvis, support_height_m, floor_y)
    pose = EndPose(kind="lie", hips=(float(pelvis[0]), height + LIE_PELVIS_ABOVE_TOP_M,
                                     float(pelvis[1])),
                   rotvec=tuple(supine_pelvis(head)), hold_s=POSE_PIN_S,
                   support_height=height, region_joints=BACK_REGION_JOINTS, support_rect=rect)
    report.update(support_height_m=round(height, 4), height_source=source,
                  support_rect=[round(value, 4) for value in rect],
                  pelvis=[round(float(value), 4) for value in pelvis],
                  head_direction=[float(value) for value in head])
    return ({"end_pose": pose, "end_heading": None, "sit_support_ids": [str(target_object_id)],
             "max_seat_penetration_m": BED_SUPPORT_PENETRATION_M}, report)


def _support_word(end: str) -> str:
    return {"sit": "seat", "lie": "bed"}.get(end, "support")


def _target_item(geometry, target: str) -> dict[str, Any]:
    if geometry is None:
        raise RouteTargetUnsupported(
            f"no scene geometry reader to measure {target}; sit and lie need the target's shape")
    for item in geometry:
        if str(item.get("object_id")) == target:
            return item
    raise RouteTargetUnsupported(f"the target {target} has no geometry in the scene")


def _footprint(item) -> tuple[float, float, float, float]:
    points = np.asarray(item.get("points") or item.get("vertices") or (), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise RouteTargetUnsupported(f"the target {item.get('object_id')} has no bounds")
    return (float(points[:, 0].min()), float(points[:, 2].min()),
            float(points[:, 0].max()), float(points[:, 2].max()))


def _depth(rect, point, direction) -> float:
    """How far from ``point`` along ``direction`` (x, z) the footprint ``rect`` reaches."""
    x0, z0, x1, z1 = rect
    reach = math.inf
    for value, low, high, step in ((point[0], x0, x1, direction[0]),
                                   (point[1], z0, z1, direction[1])):
        if step > 1e-9:
            reach = min(reach, (high - value) / step)
        elif step < -1e-9:
            reach = min(reach, (low - value) / step)
    return max(0.0, float(reach)) if math.isfinite(reach) else 0.0


def _support_height(item, xz, explicit, floor_y) -> tuple[float, str]:
    """The support's top over the floor at ``xz``: ``explicit``, else the
    median top of its triangles round ``xz`` (a ray straight down), else its
    bounds top."""
    if explicit is not None:
        return float(explicit), "given"
    tops = mesh_top_heights(item, xz)
    if tops:
        return float(np.median(tops)) - floor_y, "mesh_top"
    points = np.asarray(item.get("points") or (), dtype=np.float64)
    return float(points[:, 1].max()) - floor_y, "bounds_top"


def mesh_top_heights(item, xz, *, half=TOP_SAMPLE_HALF_M, points=TOP_SAMPLE_POINTS) -> list[float]:
    """The tops of ``item``'s triangles straight under a small grid round
    ``xz``, where a triangle is hit (roomgen's measured seat height can sit
    under the mesh: bench_7107's bed said 0.35 m, its mattress edge 0.385 m)."""
    vertices = np.asarray(item.get("vertices") or (), dtype=np.float64)
    faces = np.asarray(item.get("faces") or (), dtype=np.int64)
    if not len(vertices) or not len(faces):
        return []
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    usable = np.abs(normal[:, 1]) > 1e-9
    tops = []
    for dx in np.linspace(-half, half, points):
        for dz in np.linspace(-half, half, points):
            x, z = float(xz[0]) + dx, float(xz[1]) + dz

            def side(p, q):
                return (q[:, 0] - p[:, 0]) * (z - p[:, 2]) - (q[:, 2] - p[:, 2]) * (x - p[:, 0])

            d1, d2, d3 = side(a, b), side(b, c), side(c, a)
            inside = usable & (((d1 >= 0) & (d2 >= 0) & (d3 >= 0))
                               | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)))
            if inside.any():
                n, p = normal[inside], a[inside]
                heights = p[:, 1] - (n[:, 0] * (x - p[:, 0]) + n[:, 2] * (z - p[:, 2])) / n[:, 1]
                tops.append(float(heights.max()))
    return tops


def _fall_zone(points: np.ndarray, geometry, floor_y: float) -> dict[str, Any]:
    """The floor :data:`FALL_REACH_M` ahead of the last step, as wide as the
    body, must be clear of everything standing on it."""
    zone = {"reach_m": FALL_REACH_M, "body_radius_m": FALL_BODY_RADIUS_M}
    if geometry is None:
        return {"status": "unchecked", **zone}
    legs = np.diff(points, axis=0)
    moving = np.flatnonzero(np.linalg.norm(legs, axis=1) > 1e-6)
    direction = legs[moving[-1]] / np.linalg.norm(legs[moving[-1]])
    start = points[-1]
    stop = start + direction * FALL_REACH_M
    blockers = []
    for item in geometry:
        corners = np.asarray(item.get("points") or (), dtype=np.float64)
        if corners.ndim != 2 or corners.shape[1] != 3 or not len(corners):
            continue
        if float(corners[:, 1].max()) <= floor_y + FLOOR_ITEM_M:
            continue
        low, high = corners.min(axis=0), corners.max(axis=0)
        if _segment_hits_box(start, stop, (low[0], low[2]), (high[0], high[2]), FALL_BODY_RADIUS_M):
            blockers.append({"object_id": str(item.get("object_id")),
                             "name": str(item.get("name") or item.get("object_id")),
                             "bounds_min": low.tolist(), "bounds_max": high.tolist()})
    if blockers:
        raise FallZoneBlocked(blockers)
    return {"status": "clear", **zone}


def _segment_hits_box(start, stop, low, high, radius) -> bool:
    """Whether the segment comes within ``radius`` of the (x, z) box (the box
    grown by ``radius``, a conservative square corner)."""
    t_min, t_max = 0.0, 1.0
    delta = stop - start
    for axis in (0, 1):
        minimum, maximum = low[axis] - radius, high[axis] + radius
        if abs(float(delta[axis])) <= 1e-12:
            if not minimum <= start[axis] <= maximum:
                return False
            continue
        enter, leave = sorted(((minimum - start[axis]) / delta[axis],
                               (maximum - start[axis]) / delta[axis]))
        t_min, t_max = max(t_min, enter), min(t_max, leave)
        if t_min > t_max:
            return False
    return True


def _prompt(end: str, prompt: str | None, style_hint: str | None) -> str:
    text = str(prompt).strip() if prompt else DEFAULT_PROMPTS[end]
    if style_hint and str(style_hint).strip():
        # A style changes how the person moves, never where (one sentence:
        # Kimodo's constraint frames index a single segment).
        text = f"{text.rstrip('.')}, {str(style_hint).strip().rstrip('.')}"
    return text


def _rest_until(scene: Any, rest_until_s: float | str | None) -> float | None:
    if rest_until_s is None:
        return None
    if rest_until_s == "clip_end":
        timeline = getattr(scene, "timeline_manager", None)
        if timeline is None or not timeline.clip.tracks:
            return None
        return float(timeline.duration)
    value = float(rest_until_s)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("rest_until_s must be a finite, non-negative time")
    return value


def foot_lock_clip(pose: np.ndarray, root: np.ndarray, skin: Any, fps: float):
    """The foot clean-ups of :mod:`.foot_lock` on one SMPL clip, in the order
    the Kimodo backend and the kept-sample re-bake run them."""
    from .foot_lock import (ground_upright_floats, lift_dipping_feet, limit_bone_turn_rate,
                            lock_planted_feet, set_down_hovering_feet, settle_first_frame)

    pose, root, first = settle_first_frame(pose, root)
    pose, smoothed = limit_bone_turn_rate(pose, fps)
    root, grounded = ground_upright_floats(pose, root, skin, fps)
    pose, lifted = lift_dipping_feet(pose, root, skin, fps)
    pose, set_down = set_down_hovering_feet(pose, root, skin, fps)
    pose, root, report = lock_planted_feet(pose, root, skin, fps)
    report.update(lifted=lifted, set_down=set_down, first_frame=first,
                  smoothed_bones=smoothed, floats_grounded=grounded)
    return pose, root, report


def _clean_feet(motion_asset, skin) -> dict[str, Any] | None:
    if skin is None:
        from .kimodo_backend import smpl_skin

        skin = smpl_skin()
        if skin is None:
            return {"skipped": "the licensed SMPL model files are missing"}
    path = Path(motion_asset.motion_path)
    with np.load(path) as stored:
        data = {key: stored[key] for key in stored.files}
    np.savez(str(path).replace(".npz", ".prelock.npz"), **data)
    pose, root, report = foot_lock_clip(
        np.asarray(data["smpl_pose"], dtype=np.float64),
        np.asarray(data["root"], dtype=np.float64), skin, float(motion_asset.fps))
    data["smpl_pose"] = np.asarray(pose).astype(data["smpl_pose"].dtype)
    data["root"] = np.asarray(root).astype(data["root"].dtype)
    np.savez(path, **data)
    return _plain(report)


def _settled_after(motion_asset, plan) -> float | None:
    """Clip time from which the root stays within :data:`SETTLED_M` of its
    final height (the seat, mattress or floor reached), or None.

    Measured on Kimodo's clip as made (the ``.ungrounded.npz`` kept beside
    it): the grounding blends its height correction in over the whole
    transition. Not clamped to the arrival: a trip can start while the
    route still holds the root (real 06_fall take: down 0.2 s before it).
    A clip that never left its final height falls back to the grounded one."""
    del plan
    path = Path(motion_asset.motion_path)
    for source in (Path(str(path).replace(".npz", ".ungrounded.npz")), path):
        if not source.is_file():
            continue
        with np.load(source) as data:
            heights = np.asarray(data["root"], dtype=np.float64)[:, 1]
        away = np.flatnonzero(np.abs(heights - heights[-1]) > SETTLED_M)
        if not len(away):
            continue  # never left its final height: nothing settled here
        first = int(away[-1]) + 1
        return first / float(motion_asset.fps) if first < len(heights) else None
    return None


def _rest_to(motion_asset, begin: float, until: float | None, *, seed: int, style: str):
    """The clip continued at rest (whole frames) up to ``until`` on the
    timeline; returns the asset to bind and the frames added."""
    if until is None:
        return motion_asset, 0
    from .rest_idle import rest_idle_poses

    fps = float(motion_asset.fps)
    motion_end = begin + (motion_asset.frame_count - 1) / fps
    frames = int(math.floor((until - motion_end) * fps + 1e-6))
    if frames <= 0:
        return motion_asset, 0
    path = Path(motion_asset.motion_path)
    with np.load(path) as stored:
        data = {key: stored[key] for key in stored.files}
    pose = np.asarray(data["smpl_pose"], dtype=np.float64)
    extra = rest_idle_poses(pose[-1], frames / fps, fps, seed=seed, style=style)[:frames]
    data["smpl_pose"] = np.concatenate([pose, extra]).astype(data["smpl_pose"].dtype)
    for key in ("root", "shape"):
        # Per-frame arrays only: the betas are one row per clip.
        if key in data and np.ndim(data[key]) >= 2 and len(data[key]) == len(pose):
            data[key] = np.concatenate([data[key], np.repeat(data[key][-1:], len(extra), axis=0)])
    np.savez(path, **data)
    return replace(motion_asset, frame_count=motion_asset.frame_count + len(extra)), len(extra)


def _hold_tracks_to(scene: Any, body_id: str, end: float) -> float:
    """Every track of the person (body and bones) ends on a key at ``end``,
    its last pose held for the part of a frame the rest did not fill.
    Returns the longest hold added."""
    from copy import deepcopy

    obj = scene.get_object(body_id)
    skinned = obj.get_component("SkinnedMesh") if obj is not None else None
    ids = {body_id, *(skinned.bone_ids if skinned is not None else ())}
    longest = 0.0
    for track in scene.timeline_manager.clip.tracks.values():
        if str(track.object_id) not in ids or not track.keyframes:
            continue
        last = max(track.keyframes, key=lambda key: float(key.time))
        if float(last.time) < end - 1e-6:
            longest = max(longest, end - float(last.time))
            held = deepcopy(last)
            held.time = float(end)
            track.keyframes.append(held)
            track.keyframes.sort(key=lambda key: float(key.time))
    timeline = scene.timeline_manager
    timeline.duration = max(float(timeline.duration), float(end))
    return longest


def pivot_binding_check(scene: Any, body_id: str, motion_asset, start: float) -> dict[str, Any]:
    """How far the baked root position keys at the clip's frame times are
    from ``translation + placement + J0 - R J0`` (the body turning about the
    SMPL pelvis joint J0, as SMPL does)."""
    clip = scene.timeline_manager.clip
    track = clip.get_track(body_id, "Transform", "position")
    bone0 = scene.get_object(f"{body_id}-smpl-bone-00")
    with np.load(motion_asset.motion_path) as data:
        translation = np.asarray(data[motion_asset.trans_key], dtype=np.float64).reshape(-1, 3)
        rotvecs = np.asarray(data[motion_asset.pose_key], dtype=np.float64).reshape(
            len(translation), -1, 3)[:, 0]
    if track is None or bone0 is None:
        return {"pivot": "unchecked", "frames_checked": 0, "max_error_m": None}
    j0 = np.asarray(bone0.get_component("Transform").position, dtype=np.float64)
    fps = float(motion_asset.fps)
    keys = {round((float(key.time) - start) * fps, 3): np.asarray(key.value, dtype=np.float64)
            for key in track.keyframes}
    frames = [index for index in range(len(translation)) if float(index) in keys]
    if not frames:
        return {"pivot": "unchecked", "frames_checked": 0, "max_error_m": None}
    values = np.stack([keys[float(index)] for index in frames])
    pivot = j0 - Rotation.from_rotvec(rotvecs[frames]).apply(j0)
    offset = values - translation[frames] - pivot
    error = float(np.abs(offset - offset[0]).max())
    return {"pivot": "smpl_pelvis_j0" if error <= PIVOT_TOLERANCE_M else "not_pivoted",
            "frames_checked": len(frames), "max_error_m": round(error, 5),
            "j0": [round(float(value), 4) for value in j0]}


def _plain(value: Any) -> Any:
    """JSON-ready copy of a report (numpy scalars and arrays to Python)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
