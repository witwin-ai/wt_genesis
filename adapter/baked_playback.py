"""Choose one playback authority for a newly generated humanoid bundle.

This is a producer-side choice, not permission for consumers to ignore unknown
animation tracks. Baked playback drives the rendered SkinnedMesh via Transform
tracks. The unkeyed SMPLMotion component retains its source NPZ for editing, but
SMPLBody.pose / PlatformGeometry.pose remain the initial source pose: they are
not a time-varying parametric SMPL export in this mode.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable


PLAYBACK_MODES = ("legacy_controller", "baked_transforms")


def validate_playback_mode(playback_mode: str) -> str:
    if playback_mode not in PLAYBACK_MODES:
        raise ValueError(f"playback_mode must be one of {PLAYBACK_MODES}")
    return playback_mode


def select_playback_keyframes(
    keyframes: Iterable[dict[str, Any]],
    *,
    playback_mode: str = "legacy_controller",
    body_object_id: str,
    motion_object_id: str,
    bone_ids: Iterable[str],
    frame_count: int,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and select only this producer's NEW keyframe bundle.

    Call before recording keys; never pass or rewrite a user's existing clip.
    Legacy mode preserves the previous bundle. Baked mode requires complete
    root and bone tracks on the source frame grid before dropping the known
    source controller keys. Unknown tracks, partial bakes and duplicate keys
    fail closed. No scene, timeline or component is modified here.
    """
    mode = validate_playback_mode(playback_mode)
    bundle = deepcopy(list(keyframes))
    if mode == "legacy_controller":
        return bundle, {
            "mode": mode,
            "source_object_id": motion_object_id,
            "pose_authority": "legacy_controller_and_baked_transforms",
            "source_controller_keyed": True,
            "source_component_retained": True,
        }
    bones = list(bone_ids)
    if (not body_object_id or not motion_object_id or body_object_id == motion_object_id
            or not bones or any(not isinstance(bone, str) or not bone for bone in bones)
            or len(set(bones)) != len(bones)
            or body_object_id in bones or motion_object_id in bones):
        raise ValueError("baked playback requires distinct body, source and bone identities")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("baked playback requires a positive integer frame_count")
    if isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("baked playback requires a finite positive fps")
    rate = float(fps)
    expected = {(body_object_id, "Transform", "position"),
                (body_object_id, "Transform", "rotation")}
    expected.update((bone, "Transform", "rotation") for bone in bones)
    control_path = (motion_object_id, "SMPLMotion", "current_frame")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key in bundle:
        path = (key.get("object_id"), key.get("component_name"), key.get("field_name"))
        if path not in expected and path != control_path:
            raise ValueError(f"unknown track in generated baked bundle: {path}")
        interpolation = getattr(key.get("interpolation"), "value", key.get("interpolation"))
        if interpolation != "linear":
            raise ValueError(f"baked playback requires explicit LINEAR keys: {path}")
        time = key.get("time")
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time):
            raise ValueError(f"baked playback has invalid key time: {path}")
        if path != control_path:
            value = key.get("value")
            if (not isinstance(value, (list, tuple)) or len(value) != 3
                    or any(isinstance(item, bool) or not isinstance(item, (int, float))
                           or not math.isfinite(item) for item in value)):
                raise ValueError(f"baked playback requires finite vector3 transforms: {path}")
        groups.setdefault(path, []).append(key)
    if set(groups) != expected | {control_path}:
        raise ValueError("incomplete generated baked bundle: root, all bones and source keys are required")
    for path in expected:
        track_keys = sorted(groups[path], key=lambda item: item["time"])
        if len(track_keys) != frame_count or any(
            not math.isclose(key["time"], frame / rate, abs_tol=1e-9, rel_tol=1e-9)
            for frame, key in enumerate(track_keys)
        ):
            raise ValueError(f"baked track must cover every source frame exactly once: {path}")
        # TimelineTrack.add_keyframe replaces keys within 0.0001 seconds.
        # Validate the actual bundle spacing, not only its nominal frame rate.
        if any(right["time"] - left["time"] <= .0001
               for left, right in zip(track_keys, track_keys[1:])):
            raise ValueError(f"baked source frames exceed Studio keyframe time resolution: {path}")
    control_keys = sorted(groups[control_path], key=lambda item: item["time"])
    end_time = max(1, frame_count - 1) / rate
    if len(control_keys) != 2 or any(
        isinstance(key.get("value"), bool) or key.get("value") != frame
        or not math.isclose(key["time"], time, abs_tol=1e-9, rel_tol=1e-9)
        for key, frame, time in zip(control_keys, (0, frame_count - 1), (0., end_time))
    ):
        raise ValueError("source controller keys do not match the generated source interval")
    selected = [key for key in bundle if
                (key["object_id"], key["component_name"], key["field_name"]) != control_path]
    return selected, {
        "mode": mode,
        "source_object_id": motion_object_id,
        "pose_authority": "baked_root_and_bone_transforms",
        "source_controller_keyed": False,
        "source_component_retained": True,
        "baked_transform_track_count": len(expected),
        "baked_transform_keyframe_count": len(selected),
        "source_frame_count": frame_count,
        "source_fps": rate,
        "smpl_parameter_pose": "initial_source_frame_only",
        "parametric_smpl_animation_export_supported": False,
        "limitations": (
            "Rendered mesh playback only; SMPLBody/PlatformGeometry pose parameters do not follow the timeline. "
            "Mesh-based radar sampling still requires its normal scene/track/scattering validation; "
            "this is not a calibrated human electromagnetic scattering model."
        ),
    }


def verify_recorded_baked_playback(
    timeline: Any,
    keyframes: list[dict[str, Any]],
    playback: dict[str, Any],
) -> None:
    """Do not report a complete bake if the host merged, lost or altered keys.

    Call inside the producer's authoring transaction immediately after recording
    the selected bundle. Existing unrelated tracks remain entirely untouched.
    """
    if playback["mode"] != "baked_transforms":
        return
    controller = timeline.clip.get_track(playback["source_object_id"], "SMPLMotion", "current_frame")
    if controller is not None and controller.keyframes:
        raise ValueError("Studio retained a competing source controller in baked playback")
    expected: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key in keyframes:
        path = (key["object_id"], key["component_name"], key["field_name"])
        expected.setdefault(path, []).append(key)
    for path, records in expected.items():
        track = timeline.clip.get_track(*path)
        if track is None or len(track.keyframes) != len(records):
            raise ValueError(f"Studio did not retain the complete baked track: {path}")
        actual = sorted(track.keyframes, key=lambda key: key.time)
        ordered = sorted(records, key=lambda key: key["time"])
        for stored, record in zip(actual, ordered):
            interpolation = getattr(stored.interpolation, "value", stored.interpolation)
            if (stored.time != record["time"] or stored.value != record["value"]
                    or interpolation != "linear"):
                raise ValueError(f"Studio altered the authored baked track: {path}")
