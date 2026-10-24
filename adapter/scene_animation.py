"""Build wt_humanoid scene animation objects from generated SMPL motion."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from witwin_server.core import SceneObject
from witwin_server.core.components import MaterialComponent, PlatformGeometryComponent
from witwin_server.features.timeline.data import InterpolationType
from witwin_server.features.timeline.manager import TimelineManager
from witwin.genesis.mdm_tiny_runtime import DEFAULT_MODEL_ROOT
from wt_humanoid.adapter.smpl_motion import load_smpl_motion
from wt_humanoid.adapter.smpl_skinned import attach_smpl_skinned_mesh, build_smpl_skin_asset

from witwin.genesis.motion_export import Backend, MotionExportResult, generate_motion_asset
from .baked_playback import (
    select_playback_keyframes, validate_playback_mode, verify_recorded_baked_playback,
)
from .trajectory_clearance import HumanoidTrajectoryBlocked, validate_humanoid_trajectory


@dataclass(frozen=True)
class SceneAnimationResult:
    ok: bool
    motion_asset: MotionExportResult
    body_object_id: str
    motion_object_id: str
    visual_mesh_object_id: str
    timeline_track: dict[str, Any]
    root_motion_tracks: dict[str, Any]
    visual_mesh_cache: dict[str, Any]
    trajectory_clearance: dict[str, Any]
    playback: dict[str, Any]
    start_time: float
    duration: float
    autoplay: bool
    loop: bool
    actor_reused: bool

    def to_dict(self) -> dict[str, Any]:
        payload = self.motion_asset.to_dict()
        payload.update(
            {
                "ok": self.ok,
                "body_object_id": self.body_object_id,
                "motion_object_id": self.motion_object_id,
                "visual_mesh_object_id": self.visual_mesh_object_id,
                "timeline_track": self.timeline_track,
                "root_motion_tracks": self.root_motion_tracks,
                "visual_mesh_cache": self.visual_mesh_cache,
                "trajectory_clearance": self.trajectory_clearance,
                "playback": self.playback,
                "start_time": self.start_time,
                "duration": self.duration,
                "autoplay": self.autoplay,
                "loop": self.loop,
                "actor_reused": self.actor_reused,
            }
        )
        return payload


def create_humanoid_animation(
    *,
    scene: Any,
    project_dir: str | Path,
    prompt: str,
    motion_length: float = 6.0,
    seed: int = 10,
    device: int = 0,
    guidance_param: float = 2.5,
    smplify_iters: int = 20,
    output_name: str | None = None,
    body_name: str | None = None,
    body_object_id: str | None = None,
    motion_name: str | None = None,
    position: Sequence[float] | None = None,
    start_time: float | None = None,
    autoplay: bool = False,
    loop: bool = True,
    playback_mode: str = "legacy_controller",
    backend: Backend | None = None,
    mesh_assets: Any = None,
    broadcast: Callable[[dict[str, Any]], None] | None = None,
    serialize_object: Callable[[Any], dict[str, Any]] | None = None,
    geometry_reader: Callable[[Any], Sequence[dict[str, Any]]] | None = None,
    body_validator: Callable[[Any, str, float, float], dict[str, Any]] | None = None,
    motion_asset: MotionExportResult | None = None,
    maximum_floor_correction_m: float = 0.35,
) -> SceneAnimationResult:
    """Generate or reuse an SMPL asset, bind it, and key timeline playback."""
    if scene is None:
        raise ValueError("scene is required")
    playback_mode = validate_playback_mode(playback_mode)
    require_write = getattr(scene, "_require_author_write", None)
    if callable(require_write):
        require_write("witwin.genesis.generate_humanoid_animation")
    # Reject malformed placement before invoking an expensive generation
    # backend. None remains meaningful for existing-actor continuation.
    requested_position = _normalize_position(position) if position is not None else None
    actor_reused = body_object_id is not None
    if actor_reused:
        body_obj = scene.get_object(str(body_object_id))
        if body_obj is None:
            raise ValueError(f"existing humanoid not found: {body_object_id}")
        body = body_obj.get_component("SMPLBody")
        skinned = body_obj.get_component("SkinnedMesh")
        transform = body_obj.get_component("Transform")
        if body is None or skinned is None or transform is None or not skinned.bone_ids:
            raise ValueError("body_object_id must reference a Genesis SMPL body with a bound SkinnedMesh")
        if body_name is not None and body_name != body_obj.name:
            raise ValueError("body_name cannot rename an existing humanoid continuation")
    if motion_asset is None:
        motion_asset = generate_motion_asset(
            project_dir,
            prompt=prompt,
            motion_length=motion_length,
            seed=seed,
            device=device,
            guidance_param=guidance_param,
            smplify_iters=smplify_iters,
            output_name=output_name,
            backend=backend,
        )
    if actor_reused:
        if start_time is None:
            start_time = _continuation_start_time(scene, body_obj, motion_asset.fps)
        if requested_position is None:
            position_value = _continuation_position(scene, body_obj)
        else:
            position_value = requested_position
    else:
        position_value = requested_position or _normalize_position(None)
        start_time = 0.0 if start_time is None else start_time
        body_obj, body = _create_smpl_body(
            body_name or _default_name("Genesis SMPL Body", prompt),
            position_value,
        )
    start_time = float(start_time)
    if not math.isfinite(start_time) or start_time < 0.0:
        raise ValueError("start_time must be finite and non-negative")
    root_motion = _build_root_motion_tracks(motion_asset, position_value)
    if not actor_reused:
        body_obj.get_component("Transform").rotation = root_motion["rotations"][0]

    motion_obj, motion = _create_smpl_motion(
        motion_name or _default_name("Genesis SMPL Motion", prompt),
        position_value,
    )

    motion.motion_path = motion_asset.motion_path
    motion.pose_key = motion_asset.pose_key
    motion.betas_key = motion_asset.betas_key
    motion.trans_key = motion_asset.trans_key
    motion.loop = "on" if loop else "off"
    motion.load_motion()
    motion.current_frame = 0
    if not actor_reused:
        motion.apply_current_frame(body)
        body.model_root = str(DEFAULT_MODEL_ROOT)
        body.sync_platform_geometry()
    skin_asset = build_smpl_skin_asset(
        body.model_root,
        gender=body.gender,
        betas=body.shape,
    )
    trajectory_clearance = (
        validate_humanoid_trajectory(
            root_motion["positions"],
            skin_asset.vertices,
            geometry_reader(scene),
        )
        if geometry_reader is not None
        else {
            "status": "not_checked",
            "continuous_collision_certified": False,
            "scope": "no scene geometry reader was provided",
        }
    )
    # Model loading and motion validation are completed on detached objects.
    # A missing licensed SMPL file must not leave half-created scene objects.
    created_ids: list[str] = []
    with _new_animation_transaction(scene, created_ids):
        if not actor_reused:
            body_obj.id = scene.allocate_id(body_obj.name)
            created_ids.append(body_obj.id)
            scene.add_object(body_obj)
        motion_obj.id = scene.allocate_id(motion_obj.name)
        created_ids.append(motion_obj.id)
        scene.add_object(motion_obj)
        motion.target_body = {
            "object_id": body_obj.id,
            "object_name": body_obj.name,
            "component_type": "SMPLBody",
        }
        if actor_reused:
            skinned_status = {
                "status": "reused_existing_skinned_mesh",
                "bone_ids": list(body_obj.get_component("SkinnedMesh").bone_ids),
            }
        else:
            bone_ids = [f"{body_obj.id}-smpl-bone-{index:02d}"
                        for index in range(len(skin_asset.joint_positions))]
            if any(scene.get_object(bone_id) is not None for bone_id in bone_ids):
                raise ValueError("generated SMPL bone ids collide with existing scene objects")
            created_ids.extend(bone_ids)
            skinned_status = attach_smpl_skinned_mesh(
                scene,
                body_obj,
                body,
                model_root=body.model_root,
                gender=body.gender,
                shape=body.shape,
                asset=skin_asset,
            )
            body.sync_skinned_mesh_pose()

        timeline = _ensure_timeline(scene)
        existing_tracks = bool(timeline.clip.tracks)
        if not existing_tracks:
            timeline.fps = motion_asset.fps
            timeline.is_looping = bool(loop)
        duration = _motion_duration(motion_asset.frame_count, motion_asset.fps)
        retained_end = timeline.clip.get_keyframe_range()[1] if existing_tracks else 0.0
        animation_end = start_time + duration
        timeline.duration = (
            max(timeline.duration, retained_end, animation_end)
            if existing_tracks else animation_end
        )

        keyframes = [
            {
                "object_id": motion_obj.id,
                "component_name": "SMPLMotion",
                "field_name": "current_frame",
                "value": 0,
                "time": 0.0,
                "interpolation": InterpolationType.LINEAR,
            },
            {
                "object_id": motion_obj.id,
                "component_name": "SMPLMotion",
                "field_name": "current_frame",
                "value": max(0, motion_asset.frame_count - 1),
                "time": duration,
                "interpolation": InterpolationType.LINEAR,
            },
        ]
        for frame in range(motion_asset.frame_count):
            time = frame / motion_asset.fps
            keyframes.append(
                {
                    "object_id": body_obj.id,
                    "component_name": "Transform",
                    "field_name": "position",
                    "value": root_motion["positions"][frame],
                    "time": time,
                    "interpolation": InterpolationType.LINEAR,
                }
            )
            keyframes.append(
                {
                    "object_id": body_obj.id,
                    "component_name": "Transform",
                    "field_name": "rotation",
                    "value": root_motion["rotations"][frame],
                    "time": time,
                    "interpolation": InterpolationType.LINEAR,
                }
            )
        skeleton_track_count, skeleton_keyframes = _build_skeleton_rotation_keyframes(body_obj, motion_asset)
        keyframes.extend(skeleton_keyframes)
        keyframes, playback = select_playback_keyframes(
            keyframes,
            playback_mode=playback_mode,
            body_object_id=body_obj.id,
            motion_object_id=motion_obj.id,
            bone_ids=body_obj.get_component("SkinnedMesh").bone_ids,
            frame_count=motion_asset.frame_count,
            fps=motion_asset.fps,
        )
        if start_time:
            for keyframe in keyframes:
                keyframe["time"] = float(keyframe["time"]) + start_time
        timeline.record_keyframes(keyframes)
        verify_recorded_baked_playback(
            timeline, keyframes, playback, allow_existing=actor_reused,
        )
        if body_validator is not None:
            trajectory_clearance = _validate_with_floor_support_correction(
                scene,
                body_obj.id,
                start_time,
                animation_end,
                body_validator,
                maximum_correction_m=maximum_floor_correction_m,
            )
        playback_track = ((body_obj.id, "Transform", "position")
                          if playback_mode == "baked_transforms"
                          else (motion_obj.id, "SMPLMotion", "current_frame"))
        track = timeline.clip.get_track(*playback_track)
        if not existing_tracks:
            timeline.set_time(0.0, apply_to_scene=True)
        visual_mesh_cache = _skinned_mesh_visual_status(skinned_status, skeleton_track_count=skeleton_track_count)
        if autoplay:
            timeline.play()

        _broadcast_live_scene_update(
            scene=scene,
            body_obj=body_obj,
            motion_obj=motion_obj,
            skinned_status=skinned_status,
            timeline=timeline,
            broadcast=broadcast,
            serialize_object=serialize_object,
        )
        return SceneAnimationResult(
            ok=True,
            motion_asset=motion_asset,
            body_object_id=body_obj.id,
            motion_object_id=motion_obj.id,
            visual_mesh_object_id=body_obj.id,
            timeline_track={
                "path": track.path if track else "",
                "object_id": playback_track[0],
                "component_name": playback_track[1],
                "field_name": playback_track[2],
                "keyframe_count": len(track.keyframes) if track else 0,
            },
            root_motion_tracks={
                "position_track": f"{body_obj.id}/Transform/position",
                "rotation_track": f"{body_obj.id}/Transform/rotation",
                "keyframe_count": motion_asset.frame_count,
            },
            visual_mesh_cache=visual_mesh_cache,
            trajectory_clearance=trajectory_clearance,
            playback=playback,
            start_time=start_time,
            duration=duration,
            autoplay=bool(autoplay),
            loop=bool(loop),
            actor_reused=actor_reused,
        )


def _validate_with_floor_support_correction(
    scene: Any,
    object_id: str,
    start_time: float,
    end_time: float,
    validator: Callable[[Any, str, float, float], dict[str, Any]],
    *,
    maximum_correction_m: float = 0.35,
    maximum_attempts: int = 8,
) -> dict[str, Any]:
    """Lift only floor-contacting suffixes, then rerun final-LBS validation."""

    corrections = []
    total_correction = 0.0
    automatic_relocation = None
    for _attempt in range(maximum_attempts + 1):
        report = validator(scene, object_id, start_time, end_time)
        if report.get("automatic_relocation"):
            automatic_relocation = report["automatic_relocation"]
        try:
            result = _validated_body_clearance(report)
        except HumanoidTrajectoryBlocked as exc:
            blockers = exc.blockers
            if not blockers or any(
                not any(
                    label in f"{item.get('object_id', '')} {item.get('name', '')}".lower()
                    for label in ("floor", "rug", "carpet")
                )
                for item in blockers
            ):
                raise
            penetrations = [item.get("penetration_m") for item in blockers]
            times = [item.get("time") for item in blockers]
            if (
                any(value is None or not math.isfinite(float(value)) for value in penetrations)
                or any(value is None or not math.isfinite(float(value)) for value in times)
            ):
                raise
            correction = max(float(value) for value in penetrations) + 0.001
            if total_correction + correction > maximum_correction_m + 1e-9:
                raise
            first_contact = min(float(value) for value in times)
            _lift_root_track_suffix(
                scene,
                object_id,
                first_contact=first_contact,
                correction=correction,
            )
            total_correction += correction
            corrections.append({
                "first_contact_s": first_contact,
                "lift_m": correction,
            })
            continue
        if corrections:
            result["limitations"] = [
                value for value in result.get("limitations", [])
                if "No automatic support-contact exemption or motion correction" not in value
            ]
            result["floor_support_correction"] = {
                "status": "applied_and_revalidated",
                "attempts": len(corrections),
                "total_lift_m": total_correction,
                "maximum_lift_m": maximum_correction_m,
                "corrections": corrections,
                "scope": "bounded upward root correction after floor-only contact",
            }
        if automatic_relocation:
            result["automatic_relocation"] = automatic_relocation
        return result
    raise ValueError("humanoid floor support correction did not converge")


def _lift_root_track_suffix(
    scene: Any,
    object_id: str,
    *,
    first_contact: float,
    correction: float,
    blend_seconds: float = 0.2,
) -> None:
    timeline = getattr(scene, "timeline_manager", None)
    track = (
        timeline.clip.get_track(object_id, "Transform", "position")
        if timeline is not None else None
    )
    if track is None or not track.keyframes:
        raise ValueError("humanoid floor correction requires a recorded root position track")
    blend_start = first_contact - blend_seconds
    for keyframe in track.keyframes:
        alpha = min(1.0, max(0.0, (float(keyframe.time) - blend_start) / blend_seconds))
        if alpha <= 0.0:
            continue
        value = list(keyframe.value)
        value[1] = float(value[1]) + correction * alpha
        keyframe.value = value


def _validated_body_clearance(report: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(report, dict)
        or report.get("status") not in {"clear_at_samples", "overlap_detected"}
    ):
        raise ValueError("humanoid body validator returned no successful evidence")
    contacts = list(report.get("overlap_summary") or report.get("contacts") or ())
    blocking = []
    support = []
    for contact in contacts:
        label = f"{contact.get('obstacle_id', '')} {contact.get('obstacle', '')}".lower()
        penetration = float(contact.get("penetration_m") or 0.0)
        if any(name in label for name in ("floor", "rug", "carpet")) and penetration <= 0.02:
            support.append(contact)
        else:
            blocking.append(contact)
    if blocking:
        relocation = report.get("automatic_relocation")
        blockers = [
            {
                "object_id": str(contact.get("obstacle_id") or contact.get("obstacle") or "unknown"),
                "name": str(contact.get("obstacle") or contact.get("obstacle_id") or "unknown"),
                "time": contact.get("time"),
                "body_part": contact.get("body_part"),
                "penetration_m": contact.get("penetration_m"),
                "kind": contact.get("kind"),
                "segment_index": -1,
                **({"automatic_relocation": relocation} if relocation and index == 0 else {}),
            }
            for index, contact in enumerate(blocking[:20])
        ]
        raise HumanoidTrajectoryBlocked(blockers)
    return {
        "status": "clear_at_samples" if not contacts else "clear_except_support_contacts",
        "samples": int(report.get("samples") or 0),
        "affected_samples": int(report.get("affected_samples") or 0),
        "support_contacts_ignored": sum(int(contact.get("samples") or 1) for contact in support),
        "continuous_collision_certified": False,
        "scope": "sampled final-LBS whole-skin collision against live scene geometry",
        "limitations": list(report.get("limitations") or ()),
    }


@contextmanager
def _new_animation_transaction(scene: Any, created_ids: list[str]):
    """Roll back only this new animation, never restore/replace other objects.

    Generated NPZ assets are deliberately retained for diagnosis/reuse after a
    scene-authoring failure. Existing scene objects and tracks are not deleted.
    """
    original_timeline = getattr(scene, "timeline_manager", None)
    original_state = None if original_timeline is None else {
        field: getattr(original_timeline, field)
        for field in ("fps", "duration", "is_looping", "current_time", "is_playing", "is_recording")
    }
    # Authoring components emits change notifications; an armed recorder must
    # not add accidental keys at the user's current playhead during construction.
    if original_timeline is not None:
        original_timeline.is_recording = False
    try:
        yield
    except Exception:
        timeline = getattr(scene, "timeline_manager", None)
        if timeline is not None:
            for object_id in created_ids:
                timeline.remove_object_tracks(object_id)
        for object_id in reversed(created_ids):
            if scene.get_object(object_id) is not None:
                scene.remove_object(object_id)
        if original_timeline is None:
            if timeline is not None:
                scene.off("component_updated", timeline._on_component_updated)
            scene.timeline_manager = None
        else:
            for field, value in original_state.items():
                setattr(original_timeline, field, value)
        raise
    finally:
        if original_timeline is not None:
            original_timeline.is_recording = original_state["is_recording"]


def _broadcast_live_scene_update(
    *,
    scene: Any,
    body_obj: SceneObject,
    motion_obj: SceneObject,
    skinned_status: dict[str, Any],
    timeline: TimelineManager,
    broadcast: Callable[[dict[str, Any]], None] | None,
    serialize_object: Callable[[Any], dict[str, Any]] | None,
) -> None:
    if broadcast is None:
        return

    if serialize_object is not None:
        object_ids = [body_obj.id, motion_obj.id]
        object_ids.extend(str(bone_id) for bone_id in skinned_status.get("bone_ids", []) if bone_id)

        objects = []
        seen = set()
        for object_id in object_ids:
            if object_id in seen:
                continue
            seen.add(object_id)
            obj = scene.get_object(object_id) if hasattr(scene, "get_object") else None
            if obj is None:
                continue
            objects.append(serialize_object(obj))

        if objects:
            children_order = {}
            if hasattr(scene, "build_children_order"):
                children_order = scene.build_children_order(include_synthetic=True)
            elif hasattr(scene, "to_dict"):
                children_order = scene.to_dict().get("children_order", {})
            broadcast(
                {
                    "type": "objects_batch_created",
                    "scene_id": getattr(scene, "scene_id", "main"),
                    "objects": objects,
                    "children_order": children_order,
                }
            )

    broadcast(_timeline_state_message(timeline))


def _timeline_state_message(timeline: TimelineManager) -> dict[str, Any]:
    tracks = []
    for track in timeline.clip.tracks.values():
        tracks.append(
            {
                "path": track.path,
                "object_id": track.object_id,
                "component_name": track.component_name,
                "field_name": track.field_name,
                "keyframes": [kf.to_dict() for kf in track.keyframes],
            }
        )

    return {
        "type": "timeline_state",
        "current_time": timeline.current_time,
        "current_frame": timeline.current_frame,
        "is_playing": timeline.is_playing,
        "is_recording": timeline.is_recording,
        "is_looping": timeline.is_looping,
        "duration": timeline.duration,
        "fps": timeline.fps,
        "total_frames": timeline.total_frames,
        "tracks": tracks,
    }


def _create_smpl_body(name: str, position: list[float], rotation: list[float] | None = None):
    from wt_humanoid.components.smpl import SMPLBodyComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    body = SMPLBodyComponent()
    geom = PlatformGeometryComponent()
    geom.kind = "smpl"
    obj.add_component(MaterialComponent())
    obj.add_component(body)
    obj.add_component(geom)
    transform = obj.get_component("Transform")
    transform.position = position
    if rotation is not None:
        transform.rotation = rotation
    body.sync_platform_geometry()
    return obj, body


def _create_smpl_motion(name: str, position: list[float]):
    from wt_humanoid.components.smpl import SMPLMotionComponent

    obj = SceneObject(name=name, mesh_type="Empty")
    motion = SMPLMotionComponent()
    obj.add_component(motion)
    obj.get_component("Transform").position = position
    return obj, motion


def _ensure_timeline(scene: Any) -> TimelineManager:
    if getattr(scene, "timeline_manager", None) is None:
        scene.timeline_manager = TimelineManager(scene)
    return scene.timeline_manager


def _continuation_position(scene: Any, body_obj: SceneObject) -> list[float]:
    timeline = getattr(scene, "timeline_manager", None)
    track = (
        timeline.clip.get_track(body_obj.id, "Transform", "position")
        if timeline is not None else None
    )
    if track is not None and track.keyframes:
        return _normalize_position(track.keyframes[-1].value)
    return _normalize_position(body_obj.get_component("Transform").position)


def _continuation_start_time(scene: Any, body_obj: SceneObject, fps: float) -> float:
    timeline = getattr(scene, "timeline_manager", None)
    if timeline is None:
        return 0.0
    skinned = body_obj.get_component("SkinnedMesh")
    actor_ids = {body_obj.id, *(skinned.bone_ids if skinned is not None else ())}
    key_times = [
        float(keyframe.time)
        for track in timeline.clip.tracks.values()
        if track.object_id in actor_ids
        for keyframe in track.keyframes
    ]
    return (max(key_times) + 1.0 / float(fps)) if key_times else 0.0


def _motion_duration(frame_count: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_count <= 1:
        return 1.0 / fps
    return (frame_count - 1) / fps


def _build_root_motion_tracks(motion_asset: MotionExportResult, position: list[float]) -> dict[str, list[list[float]]]:
    motion = load_smpl_motion(
        motion_asset.motion_path,
        pose_key=motion_asset.pose_key,
        betas_key=motion_asset.betas_key,
        trans_key=motion_asset.trans_key,
    )
    frame_count = motion.frame_count
    if frame_count != motion_asset.frame_count or frame_count <= 0:
        raise ValueError("generated motion frame count does not match its asset metadata")
    if not math.isfinite(motion_asset.fps) or motion_asset.fps <= 0:
        raise ValueError("generated motion fps must be finite and positive")
    if any(not np.isfinite(values).all()
           for values in (motion.poses, motion.betas, motion.translations)):
        raise ValueError("generated motion contains nonfinite pose, shape, or root values")
    translations = np.asarray(motion.translations, dtype=np.float32).reshape(frame_count, 3)
    base_position = np.asarray(position, dtype=np.float32).reshape(1, 3)
    relative_translation = translations - translations[0].reshape(1, 3)
    positions_array = base_position + relative_translation
    positions_array[:, 1] = base_position[:, 1] + translations[:, 1]
    positions = positions_array.astype(float).tolist()

    poses = np.asarray(motion.poses, dtype=np.float32).reshape(frame_count, -1)
    root_orient = np.zeros((frame_count, 3), dtype=np.float32)
    root_orient[:, : min(3, poses.shape[1])] = poses[:, : min(3, poses.shape[1])]
    rotations = _continuous_euler_values(
        Rotation.from_rotvec(root_orient).as_euler("ZXY", degrees=False).astype(float).tolist()
    )
    return {"positions": positions, "rotations": rotations}


def _build_skeleton_rotation_keyframes(body_obj: SceneObject, motion_asset: MotionExportResult) -> tuple[int, list[dict[str, Any]]]:
    skinned = body_obj.get_component("SkinnedMesh")
    if skinned is None:
        return 0, []
    motion = load_smpl_motion(
        motion_asset.motion_path,
        pose_key=motion_asset.pose_key,
        betas_key=motion_asset.betas_key,
        trans_key=motion_asset.trans_key,
    )
    poses = np.asarray(motion.poses, dtype=np.float32).reshape(motion.frame_count, -1)
    track_count = 0
    keyframes: list[dict[str, Any]] = []
    for bone_index, bone_id in enumerate(skinned.bone_ids):
        rotations = _continuous_euler_values([
            _bone_rotation_euler(poses[frame], bone_index)
            for frame in range(motion.frame_count)
        ])
        for frame in range(motion.frame_count):
            time = frame / motion_asset.fps
            keyframes.append(
                {
                    "object_id": bone_id,
                    "component_name": "Transform",
                    "field_name": "rotation",
                    "value": rotations[frame],
                    "time": time,
                    "interpolation": InterpolationType.LINEAR,
                }
            )
        track_count += 1
    return track_count, keyframes


def _bone_rotation_euler(pose: np.ndarray, bone_index: int) -> list[float]:
    if bone_index == 0:
        return [0.0, 0.0, 0.0]
    start = bone_index * 3
    rotvec = np.zeros(3, dtype=np.float32)
    available = max(0, min(3, pose.shape[0] - start))
    if available:
        rotvec[:available] = pose[start : start + available]
    return Rotation.from_rotvec(rotvec).as_euler("ZXY", degrees=False).astype(float).tolist()


def _continuous_euler_values(values: list[list[float]]) -> list[list[float]]:
    """Select the ZXY-equivalent representation nearest the preceding frame."""
    if not values:
        return []
    result = [list(values[0])]
    for raw in values[1:]:
        alternatives = (
            list(raw),
            [raw[0] + math.pi, math.pi - raw[1], raw[2] + math.pi],
            [raw[0] + math.pi, -math.pi - raw[1], raw[2] + math.pi],
        )
        previous = np.asarray(result[-1], dtype=float)
        candidates = []
        for alternative in alternatives:
            candidate = np.asarray(alternative, dtype=float)
            candidate += 2.0 * math.pi * np.round((previous - candidate) / (2.0 * math.pi))
            candidates.append(candidate)
        chosen = min(candidates, key=lambda item: float(np.linalg.norm(item - previous)))
        result.append(chosen.tolist())
    return result


def _skinned_mesh_visual_status(skinned_status: dict[str, Any], *, skeleton_track_count: int) -> dict[str, Any]:
    return {
        "ok": bool(skinned_status.get("ok")),
        "mode": "skinned_mesh",
        "message_type": None,
        "cached_object_ids": [],
        "cached_object_count": 0,
        "bone_count": int(skinned_status.get("bone_count", 0)),
        "skeleton_track_count": int(skeleton_track_count),
        "root_bone_id": skinned_status.get("root_bone_id"),
    }


def _normalize_position(position: Sequence[float] | None) -> list[float]:
    if position is None:
        return [0.0, 0.0, 0.0]
    value = [float(item) for item in position]
    if len(value) != 3:
        raise ValueError("position must contain exactly 3 numbers")
    if not all(math.isfinite(item) for item in value):
        raise ValueError("position must contain finite numbers")
    return value


def _default_name(prefix: str, prompt: str) -> str:
    text = " ".join(str(prompt).split())
    if len(text) > 36:
        text = text[:33].rstrip() + "..."
    return f"{prefix} - {text}" if text else prefix
