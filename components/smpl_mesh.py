"""Visible SMPL mesh component for Genesis-generated humanoid playback."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from witwin_server.core.components import (
    component,
    float_field,
    int_field,
    string_field,
)
from witwin_server.core.components.mesh import MeshComponent

from wt_humanoid.adapter.smpl_motion import load_smpl_motion


PLUGIN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = PLUGIN_DIR / "vendor" / "mdm_tiny" / "data" / "body_models"
ROOT_MOTION_BAKED = "baked"
ROOT_MOTION_TRANSFORM = "transform"


@component(name="GenesisSMPLMesh", display_name="Genesis SMPL Mesh", category="Human/SMPL")
class GenesisSMPLMeshComponent(MeshComponent):
    """Parametric mesh view of a wt_humanoid SMPL motion frame."""

    motion_path = string_field("", description="Generated SMPL .npz motion file")
    pose_key = string_field("smpl_pose", description="Pose key in the motion file")
    betas_key = string_field("shape", description="Shape/betas key in the motion file")
    trans_key = string_field("root", description="Root translation key in the motion file")
    model_root = string_field(str(DEFAULT_MODEL_ROOT), description="Directory containing SMPL model files")
    root_motion_mode = string_field(
        ROOT_MOTION_BAKED,
        options=[ROOT_MOTION_BAKED, ROOT_MOTION_TRANSFORM],
        enum_toggle=True,
        description="Bake root motion into vertices or leave it for the object's Transform tracks",
    )
    gender = string_field("neutral", options=["male", "female", "neutral"], enum_toggle=True)
    fps = float_field(20.0, min=0.001, readonly=True)
    frame_count = int_field(0, min=0, readonly=True)
    current_frame = int_field(0, min=0, description="Frame to render")
    loop = string_field("on", options=["off", "on"], enum_toggle=True)
    status = string_field("", readonly=True)

    def __init__(self, **kwargs):
        super().__init__(mesh_type="Custom", **kwargs)

    def on_value_change(self, field_name, old_val, new_val):
        if field_name in {
            "motion_path",
            "pose_key",
            "betas_key",
            "trans_key",
            "model_root",
            "root_motion_mode",
            "gender",
            "current_frame",
        }:
            self._mark_dirty()

    def on_timeline_apply(self, *, field_name, value, time, frame, timeline):
        if field_name == "current_frame":
            self._ensure_generated()

    def _generate_mesh(self):
        if not self.motion_path:
            self.status = "No SMPL motion path selected"
            return _empty_mesh()
        try:
            vertices, faces = bake_smpl_sequence(
                str(self.motion_path),
                str(self.pose_key).strip() or None,
                str(self.betas_key).strip() or None,
                str(self.trans_key).strip() or None,
                str(self.model_root).strip() or str(DEFAULT_MODEL_ROOT),
                str(self.gender or "neutral"),
                root_motion_mode=str(self.root_motion_mode or ROOT_MOTION_BAKED),
            )
        except ImportError as exc:
            self.status = f"SMPL mesh bake unavailable: {exc}"
            return _empty_mesh()
        except Exception as exc:  # noqa: BLE001 - component status should surface bake failures
            self.status = f"SMPL mesh bake failed: {exc}"
            return _empty_mesh()

        frame_count = int(vertices.shape[0])
        self.frame_count = frame_count
        if frame_count <= 0:
            self.status = "SMPL mesh bake produced no frames"
            return _empty_mesh()

        index = int(self.current_frame)
        if self.loop == "on":
            index %= frame_count
        else:
            index = max(0, min(index, frame_count - 1))
        self.status = f"Rendered SMPL mesh frame {index}"
        return vertices[index], faces


@lru_cache(maxsize=8)
def bake_smpl_sequence(
    motion_path: str,
    pose_key: Optional[str],
    betas_key: Optional[str],
    trans_key: Optional[str],
    model_root: str,
    gender: str,
    root_motion_mode: str = ROOT_MOTION_BAKED,
) -> tuple[np.ndarray, np.ndarray]:
    """Bake all frames of an SMPL motion file into vertex arrays."""
    import torch
    import smplx

    root_mode = _normalize_root_motion_mode(root_motion_mode)
    motion = load_smpl_motion(motion_path, pose_key=pose_key, betas_key=betas_key, trans_key=trans_key)
    frame_count = motion.frame_count
    poses = np.asarray(motion.poses, dtype=np.float32).reshape(frame_count, -1)
    if poses.shape[1] < 72:
        padded = np.zeros((frame_count, 72), dtype=np.float32)
        padded[:, : poses.shape[1]] = poses
        poses = padded
    elif poses.shape[1] > 72:
        poses = poses[:, :72]

    betas = np.asarray(motion.betas, dtype=np.float32).reshape(1, -1)
    if betas.shape[1] < 10:
        padded_betas = np.zeros((1, 10), dtype=np.float32)
        padded_betas[:, : betas.shape[1]] = betas
        betas = padded_betas
    betas = np.repeat(betas[:, :10], frame_count, axis=0)
    translations = np.asarray(motion.translations, dtype=np.float32).reshape(frame_count, 3)
    global_orient = np.ascontiguousarray(poses[:, :3], dtype=np.float32)
    if root_mode == ROOT_MOTION_TRANSFORM:
        global_orient = np.zeros_like(global_orient, dtype=np.float32)
        translations = np.zeros_like(translations, dtype=np.float32)

    device = torch.device("cpu")
    model = smplx.create(
        model_root,
        model_type="smpl",
        gender=gender,
        ext="pkl",
        batch_size=frame_count,
    ).to(device)
    pose_t = torch.from_numpy(poses).to(device)
    betas_t = torch.from_numpy(betas).to(device)
    orient_t = torch.from_numpy(global_orient).to(device)
    trans_t = torch.from_numpy(translations).to(device)
    with torch.no_grad():
        output = model(
            global_orient=orient_t,
            body_pose=pose_t[:, 3:72],
            betas=betas_t,
            transl=trans_t,
        )
    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.uint32).reshape(-1, 3)
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _empty_mesh() -> tuple[np.ndarray, np.ndarray]:
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)


def _normalize_root_motion_mode(value: str) -> str:
    mode = str(value or ROOT_MOTION_BAKED).strip().lower()
    if mode == ROOT_MOTION_TRANSFORM:
        return ROOT_MOTION_TRANSFORM
    return ROOT_MOTION_BAKED
