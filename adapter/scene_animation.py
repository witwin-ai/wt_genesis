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
from .trajectory_clearance import validate_humanoid_trajectory


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
    duration: float
    autoplay: bool
    loop: bool

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
                "duration": self.duration,
                "autoplay": self.autoplay,
                "loop": self.loop,
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
    motion_name: str | None = None,
    position: Sequence[float] | None = None,
    autoplay: bool = False,
    loop: bool = True,
    playback_mode: str = "legacy_controller",
    backend: Backend | None = None,
    mesh_assets: Any = None,
    broadcast: Callable[[dict[str, Any]], None] | None = None,
    serialize_object: Callable[[Any], dict[str, Any]] | None = None,
    geometry_reader: Callable[[Any], Sequence[dict[str, Any]]] | None = None,
) -> SceneAnimationResult:
    """Generate an SMPL asset, bind it to wt_humanoid objects, and key timeline playback."""
    if scene is None:
        raise ValueError("scene is required")
    playback_mode = validate_playback_mode(playback_mode)
    require_write = getattr(scene, "_require_author_write", None)
    if callable(require_write):
        require_write("witwin.genesis.generate_humanoid_animation")
    position_value = _normalize_position(position)

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
    root_motion = _build_root_motion_tracks(motion_asset, position_value)

    body_obj, body = _create_smpl_body(
        body_name or _default_name("Genesis SMPL Body", prompt),
        root_motion["positions"][0],
        root_motion["rotations"][0],
    )

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
        timeline.duration = max(timeline.duration, retained_end, duration) if existing_tracks else duration

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
        timeline.record_keyframes(keyframes)
        verify_recorded_baked_playback(timeline, keyframes, playback)
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
            duration=duration,
            autoplay=bool(autoplay),
            loop=bool(loop),
        )


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
    rotations = Rotation.from_rotvec(root_orient).as_euler("ZXY", degrees=False).astype(float).tolist()
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
        for frame in range(motion.frame_count):
            time = frame / motion_asset.fps
            rotation = _bone_rotation_euler(poses[frame], bone_index)
            keyframes.append(
                {
                    "object_id": bone_id,
                    "component_name": "Transform",
                    "field_name": "rotation",
                    "value": rotation,
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
