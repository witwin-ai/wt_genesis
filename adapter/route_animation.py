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


class RouteBackendUnavailable(RuntimeError):
    """The active person backend cannot follow a route or hold an end pose."""


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
            attempts=list(self.attempts), end_pose=self.end_pose,
        )
        return payload


def create_route_animation(
    *,
    scene: Any,
    project_dir: str | Path,
    waypoints: Sequence[Sequence[float]],
    end: str = "stand",
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
    """
    if end not in END_POSES:
        raise ValueError(f"end must be one of {END_POSES}, not {end!r}")
    begin = float(start_time or 0.0)
    until = _rest_until(scene, rest_until_s)
    text = _prompt(end, prompt, style_hint)
    hold = float(END_HOLD_S[end] if end_hold_s is None else end_hold_s)
    reports: dict[str, Any] = {}

    def prepare(motion_asset, plan):
        reports["foot_lock"] = _clean_feet(motion_asset, skin) if foot_lock else None
        reports["settled"] = _settled_after(motion_asset, plan) if end != "stand" else None
        motion_asset, frames = _rest_to(motion_asset, begin, until, seed=int(seed), style=rest_style)
        reports["rest_frames"] = frames
        return motion_asset

    made = create_humanoid_path_animation(
        scene=scene, project_dir=project_dir, prompt=text, waypoints=waypoints,
        floor_y=floor_y, speed_mps=speed_mps, start_hold_s=start_hold_s, end_hold_s=hold,
        end_heading=end_heading, max_duration_s=max_motion_s, seed=int(seed), device=device,
        output_name=output_name, body_name=body_name, start_time=start_time,
        body_validator=body_validator, generator=generator, skin=skin,
        prepare_motion=prepare, **path_kwargs)
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
        attempts=[{"seed": int(seed), "status": "kept"}],
        end_pose=made.end_pose_grounding,
    )


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
    final height (the seat, mattress or floor reached), or None."""
    with np.load(motion_asset.motion_path) as data:
        heights = np.asarray(data["root"], dtype=np.float64)[:, 1]
    fps = float(motion_asset.fps)
    away = np.flatnonzero(np.abs(heights - heights[-1]) > SETTLED_M)
    first = int(away[-1]) + 1 if len(away) else 0
    first = max(first, int(math.ceil(plan.arrival_s * fps - 1e-9)))
    return first / fps if first < len(heights) else None


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
