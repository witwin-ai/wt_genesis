"""Build wt_humanoid scene animation objects from generated SMPL motion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from witwin_server.core import SceneObject
from witwin_server.core.components import MaterialComponent, PlatformGeometryComponent
from witwin_server.features.timeline.data import InterpolationType
from witwin_server.features.timeline.manager import TimelineManager
from wt_genesis.components.smpl_mesh import DEFAULT_MODEL_ROOT
from wt_humanoid.adapter.smpl_motion import load_smpl_motion
from wt_humanoid.adapter.smpl_skinned import attach_smpl_skinned_mesh, build_smpl_skin_asset

from .motion_export import Backend, MotionExportResult, generate_motion_asset


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
    backend: Backend | None = None,
    mesh_assets: Any = None,
    broadcast: Callable[[dict[str, Any]], None] | None = None,
) -> SceneAnimationResult:
    """Generate an SMPL asset, bind it to wt_humanoid objects, and key timeline playback."""
    if scene is None:
        raise ValueError("scene is required")

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
    position_value = _normalize_position(position)
    root_motion = _build_root_motion_tracks(motion_asset, position_value)

    body_obj, body = _create_smpl_body(
        body_name or _default_name("Genesis SMPL Body", prompt),
        root_motion["positions"][0],
        root_motion["rotations"][0],
    )
    scene.add_object(body_obj)

    motion_obj, motion = _create_smpl_motion(
        motion_name or _default_name("Genesis SMPL Motion", prompt),
        position_value,
    )
    scene.add_object(motion_obj)

    motion.motion_path = motion_asset.motion_path
    motion.pose_key = motion_asset.pose_key
    motion.betas_key = motion_asset.betas_key
    motion.trans_key = motion_asset.trans_key
    motion.loop = "on" if loop else "off"
    motion.target_body = {
        "object_id": body_obj.id,
        "object_name": body_obj.name,
        "component_type": "SMPLBody",
    }
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
    timeline.fps = motion_asset.fps
    timeline.is_looping = bool(loop)
    duration = _motion_duration(motion_asset.frame_count, motion_asset.fps)
    timeline.duration = duration

    timeline.record_keyframe(
        motion_obj.id,
        "SMPLMotion",
        "current_frame",
        0,
        time=0.0,
        interpolation=InterpolationType.LINEAR,
    )
    timeline.record_keyframe(
        motion_obj.id,
        "SMPLMotion",
        "current_frame",
        max(0, motion_asset.frame_count - 1),
        time=duration,
        interpolation=InterpolationType.LINEAR,
    )
    for frame in range(motion_asset.frame_count):
        time = frame / motion_asset.fps
        timeline.record_keyframe(
            body_obj.id,
            "Transform",
            "position",
            root_motion["positions"][frame],
            time=time,
            interpolation=InterpolationType.LINEAR,
        )
        timeline.record_keyframe(
            body_obj.id,
            "Transform",
            "rotation",
            root_motion["rotations"][frame],
            time=time,
            interpolation=InterpolationType.LINEAR,
        )
    skeleton_track_count = _record_skeleton_rotation_tracks(timeline, body_obj, motion_asset)
    track = timeline.clip.get_track(motion_obj.id, "SMPLMotion", "current_frame")
    timeline.set_time(0.0, apply_to_scene=True)
    visual_mesh_cache = _skinned_mesh_visual_status(skinned_status, skeleton_track_count=skeleton_track_count)
    if autoplay:
        timeline.play()

    return SceneAnimationResult(
        ok=True,
        motion_asset=motion_asset,
        body_object_id=body_obj.id,
        motion_object_id=motion_obj.id,
        visual_mesh_object_id=body_obj.id,
        timeline_track={
            "path": track.path if track else "",
            "object_id": motion_obj.id,
            "component_name": "SMPLMotion",
            "field_name": "current_frame",
            "keyframe_count": len(track.keyframes) if track else 0,
        },
        root_motion_tracks={
            "position_track": f"{body_obj.id}/Transform/position",
            "rotation_track": f"{body_obj.id}/Transform/rotation",
            "keyframe_count": motion_asset.frame_count,
        },
        visual_mesh_cache=visual_mesh_cache,
        duration=duration,
        autoplay=bool(autoplay),
        loop=bool(loop),
    )


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
    translations = np.asarray(motion.translations, dtype=np.float32).reshape(frame_count, 3)
    base_position = np.asarray(position, dtype=np.float32).reshape(1, 3)
    relative_translation = translations - translations[0].reshape(1, 3)
    positions = (base_position + relative_translation).astype(float).tolist()

    poses = np.asarray(motion.poses, dtype=np.float32).reshape(frame_count, -1)
    root_orient = np.zeros((frame_count, 3), dtype=np.float32)
    root_orient[:, : min(3, poses.shape[1])] = poses[:, : min(3, poses.shape[1])]
    rotations = Rotation.from_rotvec(root_orient).as_euler("ZXY", degrees=False).astype(float).tolist()
    return {"positions": positions, "rotations": rotations}


def _record_skeleton_rotation_tracks(timeline: TimelineManager, body_obj: SceneObject, motion_asset: MotionExportResult) -> int:
    skinned = body_obj.get_component("SkinnedMesh")
    if skinned is None:
        return 0
    motion = load_smpl_motion(
        motion_asset.motion_path,
        pose_key=motion_asset.pose_key,
        betas_key=motion_asset.betas_key,
        trans_key=motion_asset.trans_key,
    )
    poses = np.asarray(motion.poses, dtype=np.float32).reshape(motion.frame_count, -1)
    track_count = 0
    for bone_index, bone_id in enumerate(skinned.bone_ids):
        for frame in range(motion.frame_count):
            time = frame / motion_asset.fps
            rotation = _bone_rotation_euler(poses[frame], bone_index)
            timeline.record_keyframe(
                bone_id,
                "Transform",
                "rotation",
                rotation,
                time=time,
                interpolation=InterpolationType.LINEAR,
            )
        track_count += 1
    return track_count


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
    return value


def _default_name(prefix: str, prompt: str) -> str:
    text = " ".join(str(prompt).split())
    if len(text) > 36:
        text = text[:33].rstrip() + "..."
    return f"{prefix} - {text}" if text else prefix
