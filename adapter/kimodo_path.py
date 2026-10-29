"""Kimodo humans that walk a planned route through the scene.

Text alone does not say where a generated human goes (a "few slow small
steps" prompt walked 11.9 m in 9 s). Kimodo accepts dense root (x, z) and
heading constraints, so a route planned in scene coordinates becomes a
constraint list relative to its start, the generated clip is placed back on
the route, and the result reports how closely the final root track follows
it plus the timed occupancy a later cat + human script plans against.

Conventions: scene and Kimodo are both Y-up; Kimodo heading 0 faces +Z and
``heading = atan2(dx, dz)`` for a facing direction (dx, dz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from witwin.genesis.motion_export import generate_motion_asset

from .scene_animation import SceneAnimationResult, create_humanoid_animation


KIMODO_FPS = 30.0
#: Final seconds a requested end heading is held for (time to turn in place).
END_HEADING_S = 0.3
#: Horizontal radius of a standing/walking adult around its root: shoulder
#: half-width plus arm swing. Used for timed occupancy, not for validation.
HUMAN_OCCUPANCY_RADIUS_M = 0.3
#: Fastest leg a route may ask for (a brisk run).
MAX_LEG_SPEED_MPS = 3.0


def heading_from_direction(direction: Sequence[float]) -> float:
    """Kimodo heading of a facing direction (dx, dz) in the ground plane."""
    dx, dz = float(direction[0]), float(direction[1])
    if math.hypot(dx, dz) < 1e-9:
        raise ValueError("a heading direction must not be zero")
    return math.atan2(dx, dz)


@dataclass(frozen=True)
class RootPathPlan:
    """A timed ground route sampled at Kimodo's frame rate."""

    waypoints: np.ndarray
    knot_times: np.ndarray
    frames_xz: np.ndarray
    fps: float
    arrival_s: float
    start_heading: float | None = None
    end_heading: float | None = None
    start_heading_frames: int = 0
    end_heading_frames: int = 0
    loose_end_frames: int = 0
    sit_at: tuple[float, float] | None = None
    end_pose: Any = None
    end_pose_frames: int = 0
    loose_range: tuple[int, int] = (0, 0)
    _knot_xz: np.ndarray = field(default=None, repr=False)

    @property
    def origin(self) -> tuple[float, float]:
        return float(self.waypoints[0, 0]), float(self.waypoints[0, 1])

    @property
    def frame_count(self) -> int:
        return int(self.frames_xz.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.frame_count - 1) / self.fps

    @property
    def final_xz(self) -> tuple[float, float]:
        """Where the root ends: the seat when sitting, else the last waypoint."""
        if self.sit_at is not None:
            return self.sit_at
        return float(self.waypoints[-1, 0]), float(self.waypoints[-1, 1])

    @property
    def kimodo_duration_s(self) -> float:
        """Duration that makes Kimodo's ``int(duration * fps)`` equal frame_count."""
        return (self.frame_count + 0.5) / self.fps

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1).sum())

    def position_at(self, time: float) -> tuple[float, float]:
        """Planned scene (x, z) at motion time ``time`` (clamped)."""
        x = float(np.interp(time, self.knot_times, self._knot_xz[:, 0]))
        z = float(np.interp(time, self.knot_times, self._knot_xz[:, 1]))
        return x, z

    def kimodo_constraints(self) -> list[dict[str, Any]]:
        """Kimodo ``root2d`` constraint sets, relative to the route start.

        Every frame is constrained once: the start and end heading frames in
        sets of their own, the rest as a plain path. When sitting, the end
        hold is left to the model (turn and sit) except its last frame; with
        an ``end_pose`` its last frames pin the pelvis (Kimodo ``Hips``
        end effector) instead of the root path.
        """
        count = self.frame_count
        relative = self.frames_xz - np.asarray(self.origin).reshape(1, 2)
        pose_frames = list(range(count - self.end_pose_frames, count)) if self.end_pose_frames else []
        count -= len(pose_frames)
        end_frames = set(range(count - self.end_heading_frames, count))
        start_frames = set(range(min(self.start_heading_frames, count))) - end_frames
        loose = set(range(*self.loose_range))
        plain = [index for index in range(count)
                 if index not in end_frames and index not in start_frames
                 and index not in loose]
        sets = []
        for indices, heading in ((sorted(start_frames), self.start_heading),
                                 (plain, None),
                                 (sorted(end_frames - loose), self.end_heading)):
            if not indices:
                continue
            item = {
                "type": "root2d",
                "frame_indices": [int(index) for index in indices],
                "smooth_root_2d": [[float(relative[index, 0]), float(relative[index, 1])]
                                   for index in indices],
            }
            if heading is not None:
                item["global_root_heading"] = [[math.cos(heading), math.sin(heading)]] * len(indices)
            sets.append(item)
        if pose_frames:
            sets.append(self.end_pose.constraint(pose_frames, self.origin))
        return sets

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoints": self.waypoints.astype(float).tolist(),
            "length_m": round(self.length_m, 4),
            "duration_s": round(self.duration_s, 4),
            "arrival_s": round(self.arrival_s, 4),
            "fps": self.fps,
            "start_heading": self.start_heading,
            "end_heading": self.end_heading,
            "sit_at": None if self.sit_at is None else list(self.sit_at),
            "end_pose": None if self.end_pose is None else {
                "kind": self.end_pose.kind, "hips": list(self.end_pose.hips),
                "rotvec": list(self.end_pose.rotvec), "hold_s": self.end_pose.hold_s},
        }


def plan_root_path(
    waypoints: Sequence[Sequence[float]],
    *,
    speed_mps: float = 1.0,
    waypoint_times: Sequence[float] | None = None,
    start_hold_s: float = 0.5,
    end_hold_s: float = 1.5,
    ease_s: float = 0.4,
    start_heading: float | str | None = "auto",
    end_heading: float | None = None,
    sit_at: Sequence[float] | None = None,
    end_pose: Any = None,
    max_duration_s: float = 20.0,
    fps: float = KIMODO_FPS,
) -> RootPathPlan:
    """Time a ground route: stand, walk the waypoints, stop and hold.

    ``waypoints`` are scene (x, z) or (x, y, z) points. Without
    ``waypoint_times`` the walk runs at ``speed_mps`` with ``ease_s`` of
    acceleration at each end, after ``start_hold_s`` standing. With them,
    each waypoint is reached at its time (a repeated waypoint is a pause).
    ``start_heading="auto"`` faces the first leg while standing.
    ``sit_at`` (x, z) is a seat the root ends over after the walk: the end
    hold is left free for Kimodo to turn and sit, only its last frame is
    pinned there (with ``end_heading``, facing away from the seat).
    """
    points = _ground_points(waypoints)
    legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not legs.sum() > 1e-6:
        raise ValueError("a route must move: its waypoints are all the same point")
    for name, value in (("speed_mps", speed_mps), ("start_hold_s", start_hold_s),
                        ("end_hold_s", end_hold_s), ("ease_s", ease_s)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if waypoint_times is None:
        if not float(speed_mps) > 0.0:
            raise ValueError("speed_mps must be positive")
        knot_times, knot_xz, arrival = _eased_walk(points, legs, float(speed_mps),
                                                   float(ease_s), float(start_hold_s), fps)
    else:
        times = np.asarray([float(value) for value in waypoint_times], dtype=np.float64)
        if times.shape != (points.shape[0],) or not np.isfinite(times).all():
            raise ValueError("waypoint_times must give one finite time per waypoint")
        if times[0] < 0.0 or np.any(np.diff(times) < 0.0):
            raise ValueError("waypoint_times must start at or after 0 and never decrease")
        spans = np.diff(times)
        if np.any((spans <= 1e-9) & (legs > 1e-6)):
            raise ValueError("two different waypoints cannot share a time")
        knot_times = np.concatenate([[0.0], times]) if times[0] > 0.0 else times
        knot_xz = np.concatenate([points[:1], points]) if times[0] > 0.0 else points
        arrival = float(times[-1])
    moving = np.diff(knot_times) > 1e-9
    speeds = (np.linalg.norm(np.diff(knot_xz, axis=0), axis=1)[moving]
              / np.diff(knot_times)[moving])
    if speeds.size and speeds.max() > MAX_LEG_SPEED_MPS:
        raise ValueError(f"a leg asks for {speeds.max():.2f} m/s, above {MAX_LEG_SPEED_MPS} m/s")
    duration = arrival + float(end_hold_s)
    seat = None
    if sit_at is not None:
        seat = tuple(_ground_points([knot_xz[-1], sit_at])[1])
        if not float(end_hold_s) > 0.0:
            raise ValueError("sitting needs an end hold to turn and sit in")
        knot_times = np.concatenate([knot_times, [duration]])
        knot_xz = np.concatenate([knot_xz, [seat]])
    if duration > float(max_duration_s) + 1e-9:
        raise ValueError(f"route takes {duration:.2f} s, above max_duration_s={max_duration_s}")
    count = int(round(duration * fps)) + 1
    frame_times = np.arange(count, dtype=np.float64) / fps
    frames = np.column_stack([np.interp(frame_times, knot_times, knot_xz[:, axis])
                              for axis in (0, 1)])
    if start_heading == "auto":
        first = next(index for index, length in enumerate(legs) if length > 1e-6)
        start_heading = heading_from_direction(points[first + 1] - points[first])
    start_frames = 0
    if start_heading is not None:
        start_frames = max(1, int(round(float(start_hold_s) * fps))) if waypoint_times is None else 1
    end_frames = 0
    if end_heading is not None:
        end_frames = max(1, min(int(round(END_HEADING_S * fps)), int(round(end_hold_s * fps))))
    loose = 0
    loose_range = (0, 0)
    arrival_frame = int(round(arrival * fps))
    if seat is not None:
        # The end hold is the model's (turn and sit); the last frame is pinned.
        loose = max(0, count - 2 - arrival_frame)
        loose_range = (arrival_frame + 1, count - 1)
        end_frames = 1 if end_heading is not None else 0
    elif end_pose is not None:
        # Lying down or falling: after the walk the model is free, up to the
        # pelvis pin (if any) on the last frames.
        loose = max(0, count - 1 - arrival_frame)
        loose_range = (arrival_frame + 1, count)
        end_frames = 0
    return RootPathPlan(
        waypoints=points, knot_times=knot_times, frames_xz=frames, fps=float(fps),
        arrival_s=float(arrival),
        start_heading=None if start_heading is None else float(start_heading),
        end_heading=None if end_heading is None else float(end_heading),
        start_heading_frames=start_frames, end_heading_frames=end_frames,
        loose_end_frames=loose, sit_at=seat, end_pose=end_pose,
        end_pose_frames=(min(count - 1, max(1, int(round(end_pose.hold_s * fps))))
                         if end_pose is not None and end_pose.hips is not None else 0),
        loose_range=loose_range,
        _knot_xz=knot_xz,
    )


def _ground_points(waypoints) -> np.ndarray:
    rows = []
    for point in waypoints:
        values = [float(value) for value in point]
        if len(values) == 3:
            values = [values[0], values[2]]
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("waypoints must be finite (x, z) or (x, y, z) points")
        rows.append(values)
    if len(rows) < 2:
        raise ValueError("a route needs at least two waypoints")
    return np.asarray(rows, dtype=np.float64)


def _eased_walk(points, legs, speed, ease, start_hold, fps):
    """Knots of a trapezoidal speed profile along the polyline."""
    length = float(legs.sum())
    if ease > 0.0 and length < speed * ease:
        peak = math.sqrt(length * speed / ease)
        ramp, cruise = peak * ease / speed, 0.0
    else:
        peak, ramp = speed, ease
        cruise = (length - speed * ease) / speed if ease > 0.0 else length / speed
    walk = 2.0 * ramp + cruise
    accel = peak / ramp if ramp > 0.0 else 0.0

    def travelled(tau):
        if tau <= ramp:
            return 0.5 * accel * tau * tau
        if tau <= ramp + cruise:
            return 0.5 * accel * ramp * ramp + peak * (tau - ramp)
        rest = walk - tau
        return length - 0.5 * accel * rest * rest

    steps = max(2, int(math.ceil(walk * fps * 2)) + 1)
    taus = np.linspace(0.0, walk, steps)
    distance = np.clip([travelled(tau) for tau in taus], 0.0, length)
    cumulative = np.concatenate([[0.0], np.cumsum(legs)])
    xz = np.column_stack([np.interp(distance, cumulative, points[:, axis]) for axis in (0, 1)])
    times = start_hold + taus
    if start_hold > 0.0:
        times = np.concatenate([[0.0], times])
        xz = np.concatenate([points[:1], xz])
    return times, xz, start_hold + walk


@dataclass(frozen=True)
class PathAnimationResult:
    animation: SceneAnimationResult
    plan: RootPathPlan
    path_following: dict[str, Any]
    occupancy: dict[str, Any]
    end_pose_grounding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.animation.to_dict()
        payload.update(plan=self.plan.to_dict(), path_following=self.path_following,
                       occupancy=self.occupancy, end_pose_grounding=self.end_pose_grounding)
        return payload


def create_humanoid_path_animation(
    *,
    scene: Any,
    project_dir: str | Path,
    prompt: str,
    waypoints: Sequence[Sequence[float]],
    floor_y: float | None = None,
    speed_mps: float = 1.0,
    waypoint_times: Sequence[float] | None = None,
    start_hold_s: float = 0.5,
    end_hold_s: float = 1.5,
    ease_s: float = 0.4,
    start_heading: float | str | None = "auto",
    end_heading: float | None = None,
    sit_at: Sequence[float] | None = None,
    sit_support_ids: Sequence[str] = (),
    max_seat_penetration_m: float = 0.10,
    end_pose: Any = None,
    skin: Any = None,
    max_duration_s: float = 20.0,
    seed: int = 10,
    device: int = 0,
    output_name: str | None = None,
    body_name: str | None = None,
    start_time: float | None = None,
    playback_mode: str = "baked_transforms",
    body_validator: Callable[[Any, str, float, float], dict[str, Any]] | None = None,
    generator: Callable[..., dict[str, Any]] | None = None,
    occupancy_fps: float = 10.0,
    **animation_kwargs: Any,
) -> PathAnimationResult:
    """Generate a Kimodo human walking ``waypoints`` and bind it in ``scene``.

    The route is planned with :func:`plan_root_path`, passed to Kimodo as
    root constraints relative to its start, and the generated clip is placed
    so its root starts on the first waypoint at ``floor_y`` (the first
    waypoint's y when it has one). ``sit_support_ids`` names a seat whose
    contact with the body (up to ``max_seat_penetration_m``) is support, not
    collision; pair it with ``sit_at`` so the model can turn and sit there.
    ``end_pose`` (:class:`~.kimodo_poses.EndPose`) pins the pelvis at the end
    (sit, lie) or leaves a fall to the model, and rests the settled body on
    its support, measured on the posed SMPL ``skin`` (the licensed model by
    default).
    """
    plan = plan_root_path(
        waypoints, speed_mps=speed_mps, waypoint_times=waypoint_times,
        start_hold_s=start_hold_s, end_hold_s=end_hold_s, ease_s=ease_s,
        start_heading=start_heading, end_heading=end_heading, sit_at=sit_at,
        end_pose=end_pose, max_duration_s=max_duration_s,
    )
    if floor_y is None:
        first = [float(value) for value in waypoints[0]]
        floor_y = first[1] if len(first) == 3 else 0.0
    if generator is None:
        from .kimodo_backend import generate_smpl_motion as generator
    constraints = plan.kimodo_constraints()

    def backend(prompts, **kwargs):
        return generator(prompts, constraints=constraints, **kwargs)

    text = _single_sentence(prompt)
    motion_asset = generate_motion_asset(
        project_dir, prompt=text, motion_length=plan.kimodo_duration_s, seed=seed,
        device=device, output_name=output_name, backend=backend,
    )
    grounding = None
    if end_pose is not None and end_pose.support_height is not None:
        grounding = _ground_end_pose_asset(motion_asset, plan, end_pose, skin)
    with np.load(motion_asset.motion_path) as data:
        first_root = np.asarray(data["root"], dtype=np.float64).reshape(-1, 3)[0]
    # Placement keys position + (trans - trans[0]); starting the body at
    # origin + trans[0] puts the Kimodo frame's origin on the route start.
    position = [plan.origin[0] + float(first_root[0]), float(floor_y),
                plan.origin[1] + float(first_root[2])]
    seat = _SeatSupport(body_validator, sit_support_ids, max_seat_penetration_m)
    animation = create_humanoid_animation(
        scene=scene, project_dir=project_dir, prompt=text, seed=seed, device=device,
        body_name=body_name, position=position, start_time=start_time,
        playback_mode=playback_mode, loop=False, autoplay=False,
        body_validator=seat.validate if body_validator is not None else None,
        motion_asset=motion_asset, **animation_kwargs,
    )
    if sit_support_ids and body_validator is not None:
        animation.trajectory_clearance["seat_support_contacts"] = seat.exempted
        animation.trajectory_clearance["seat_support_ids"] = list(sit_support_ids)
    following = path_following_error(scene, animation.body_object_id, plan,
                                     start_time=animation.start_time)
    occupancy = humanoid_occupancy(scene, animation.body_object_id, fps=occupancy_fps)
    return PathAnimationResult(animation=animation, plan=plan, path_following=following,
                               occupancy=occupancy, end_pose_grounding=grounding)


def _clip_rect(rect, origin):
    """A scene footprint (x0, z0, x1, z1) in the clip's frame: placement puts
    the Kimodo origin on the route start, so clip = scene - start."""
    if rect is None:
        return None
    x0, z0, x1, z1 = (float(value) for value in rect)
    return (x0 - origin[0], z0 - origin[1], x1 - origin[0], z1 - origin[1])


def _ground_end_pose_asset(motion_asset, plan: RootPathPlan, end_pose, skin) -> dict[str, Any] | None:
    """Rest the generated clip's settled pose on its support, in the asset file."""
    from .kimodo_poses import ground_end_pose

    if skin is None:
        from .kimodo_backend import smpl_skin

        skin = smpl_skin()
        if skin is None:
            return None
    with np.load(motion_asset.motion_path) as data:
        stored = {key: data[key] for key in data.files}
    duration = (motion_asset.frame_count - 1) / motion_asset.fps
    rest_start = max(plan.arrival_s, duration - float(end_pose.rest_s))
    motion = {"smpl_pose": stored["smpl_pose"][None], "smpl_trans": stored["root"][None]}
    grounded, report = ground_end_pose(
        motion, skin, support_height=float(end_pose.support_height),
        region_joints=end_pose.region_joints, transition=(plan.arrival_s, rest_start),
        rest=(rest_start, duration), fps=motion_asset.fps,
        support_rect=_clip_rect(end_pose.support_rect, plan.origin))
    stored["root"] = np.asarray(grounded["smpl_trans"][0], dtype=stored["root"].dtype)
    np.savez(motion_asset.motion_path, **stored)
    return {"kind": end_pose.kind, **report}


def _single_sentence(prompt: str) -> str:
    parts = [part.strip() for part in str(prompt).split(".") if part.strip()]
    if not parts:
        raise ValueError("prompt must be a non-empty string")
    return "; ".join(parts) + "."


_FLOOR_LABELS = ("floor", "rug", "carpet")


class _SeatSupport:
    """Wrap a body validator so contact with the named seat counts as support.

    Kimodo bodies start sunk into the floor (its hip position is used as the
    SMPL root) until the floor-only lift raises them. While the floor is still
    penetrated, seat depth says nothing about the seated pose, so seat
    contacts are left out and the lift can run; once the body stands on the
    floor, only contact up to ``max_penetration`` counts as support.
    """

    def __init__(self, validator, seat_ids, max_penetration):
        self.validator = validator
        self.seat_ids = {str(value) for value in seat_ids}
        self.max_penetration = float(max_penetration)
        self.exempted = 0

    def _is_seat(self, contact, *, sunk: bool) -> bool:
        ids = {str(contact.get("obstacle_id") or ""), str(contact.get("obstacle") or "")}
        penetration = float(contact.get("penetration_m") or 0.0)
        return bool(ids & self.seat_ids) and (sunk or penetration <= self.max_penetration)

    @staticmethod
    def _on_floor(contact) -> bool:
        label = f"{contact.get('obstacle_id', '')} {contact.get('obstacle', '')}".lower()
        return any(name in label for name in _FLOOR_LABELS)

    def validate(self, scene, object_id, start, end):
        report = self.validator(scene, object_id, start, end)
        if not self.seat_ids or not isinstance(report, dict):
            return report
        report = dict(report)
        exempted = 0
        sunk = any(self._on_floor(contact)
                   for key in ("overlap_summary", "contacts") for contact in report.get(key) or ())
        for key in ("overlap_summary", "contacts"):
            contacts = report.get(key)
            if not contacts:
                continue
            kept = [contact for contact in contacts if not self._is_seat(contact, sunk=sunk)]
            exempted = max(exempted, len(contacts) - len(kept))
            report[key] = kept
        self.exempted = exempted
        return report


def _root_keys(scene: Any, body_object_id: str) -> tuple[np.ndarray, np.ndarray]:
    timeline = getattr(scene, "timeline_manager", None)
    track = (timeline.clip.get_track(body_object_id, "Transform", "position")
             if timeline is not None else None)
    if track is None or not track.keyframes:
        raise ValueError(f"humanoid {body_object_id} has no root position track")
    times = np.asarray([float(key.time) for key in track.keyframes], dtype=np.float64)
    values = np.asarray([list(key.value) for key in track.keyframes], dtype=np.float64)
    return times, values


def path_following_error(scene: Any, body_object_id: str, plan: RootPathPlan, *,
                         start_time: float = 0.0) -> dict[str, Any]:
    """How far the final root track strays from the plan.

    ``timed`` compares each key with where the plan says the human should be
    at that time (what a joint script relies on); ``route`` is the distance
    to the route polyline whenever it is reached. Both cover the walk and,
    unless sitting (left to the model), the end hold; ``end_error_m`` is the
    last key against the final point (the seat when sitting).
    ``max_root_speed_mps`` is the fastest root step of the whole clip.
    """
    times, values = _root_keys(scene, body_object_id)
    all_xz = values[:, [0, 2]]
    end = float(np.linalg.norm(all_xz[-1] - np.asarray(plan.final_xz)))
    # A glitching clip jumps the root (13 s Kimodo clips: 0.5 m per frame).
    spans = np.diff(times)
    steps = np.linalg.norm(np.diff(all_xz, axis=0), axis=1)
    moving = spans > 1e-9
    fastest = float((steps[moving] / spans[moving]).max()) if moving.any() else 0.0
    if plan.sit_at is not None or plan.end_pose is not None:
        # Sitting, lying down and falling leave the route on purpose.
        walking = times - start_time <= plan.arrival_s + 1e-6
        times, values = times[walking], values[walking]
    xz = values[:, [0, 2]]
    planned = np.asarray([plan.position_at(time - start_time) for time in times])
    timed = np.linalg.norm(xz - planned, axis=1)
    route = np.asarray([_distance_to_polyline(point, plan.waypoints) for point in xz])
    return {
        "samples": int(len(times)),
        "timed_mean_error_m": round(float(timed.mean()), 4),
        "timed_max_error_m": round(float(timed.max()), 4),
        "route_mean_error_m": round(float(route.mean()), 4),
        "route_max_error_m": round(float(route.max()), 4),
        "end_error_m": round(end, 4),
        "walked_m": round(float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum()), 4),
        "planned_m": round(plan.length_m, 4),
        "max_root_speed_mps": round(fastest, 3),
    }


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    best = math.inf
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        span = float(segment @ segment)
        weight = 0.0 if span <= 1e-12 else min(1.0, max(0.0, float((point - start) @ segment) / span))
        best = min(best, float(np.linalg.norm(point - (start + weight * segment))))
    return best


def humanoid_occupancy(scene: Any, body_object_id: str, *, fps: float = 10.0,
                       radius_m: float = HUMAN_OCCUPANCY_RADIUS_M) -> dict[str, Any]:
    """Where the human's root is over time, for planning another actor around it.

    Samples the baked root position track at ``fps``; a disc of ``radius_m``
    round each sample covers a standing or walking adult (not a fall or a
    lunge: check body collision for those).
    """
    if not fps > 0.0:
        raise ValueError("fps must be positive")
    times, values = _root_keys(scene, body_object_id)
    count = int(math.floor((times[-1] - times[0]) * fps + 1e-9)) + 1
    sample_times = times[0] + np.arange(count, dtype=np.float64) / fps
    samples = [
        {"t": round(float(time), 4),
         "x": round(float(np.interp(time, times, values[:, 0])), 4),
         "y": round(float(np.interp(time, times, values[:, 1])), 4),
         "z": round(float(np.interp(time, times, values[:, 2])), 4)}
        for time in sample_times
    ]
    return {"body_object_id": body_object_id, "fps": float(fps), "radius_m": float(radius_m),
            "start_s": round(float(times[0]), 4), "end_s": round(float(times[-1]), 4),
            "samples": samples}
