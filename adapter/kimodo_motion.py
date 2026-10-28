"""Convert Kimodo motion files to the canonical WT Genesis SMPL arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


_SOMA77_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd",
    "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm",
    "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandThumbEnd", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandIndex4", "LeftHandIndexEnd", "LeftHandMiddle1", "LeftHandMiddle2",
    "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd", "LeftHandRing1",
    "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4",
    "LeftHandPinkyEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
    "RightHandIndexEnd", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandMiddle4", "RightHandMiddleEnd", "RightHandRing1", "RightHandRing2",
    "RightHandRing3", "RightHandRing4", "RightHandRingEnd", "RightHandPinky1",
    "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd", "RightLeg",
    "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
)

# SMPL's first 22 joints and SOMA's major body joints have equivalent chains.
_SOMA_TO_SMPL = {
    0: "Hips", 1: "LeftLeg", 2: "RightLeg", 3: "Spine1",
    4: "LeftShin", 5: "RightShin", 6: "Spine2", 7: "LeftFoot",
    8: "RightFoot", 9: "Chest", 10: "LeftToeBase", 11: "RightToeBase",
    12: "Neck1", 13: "LeftShoulder", 14: "RightShoulder", 15: "Head",
    16: "LeftArm", 17: "RightArm", 18: "LeftForeArm", 19: "RightForeArm",
    20: "LeftHand", 21: "RightHand",
}


def load_kimodo_motion(path: str | Path, *, target_fps: float = 20.0) -> dict[str, np.ndarray]:
    """Load either SOMA Kimodo NPZ or SMPL-X AMASS NPZ as Genesis output."""
    with np.load(Path(path), allow_pickle=False) as data:
        files = set(data.files)
        if {"root_orient", "pose_body", "trans"} <= files:
            pose, trans, source_fps = _from_amass(data)
        elif {"local_rot_mats", "root_positions"} <= files:
            pose, trans, source_fps = _from_soma(data)
        else:
            raise ValueError("Kimodo output is neither an AMASS nor a SOMA motion NPZ")
    pose, trans = _resample(pose, trans, source_fps, target_fps)
    frames = pose.shape[0]
    return {
        "smpl_pose": pose.reshape(1, frames, 24, 3),
        "smpl_betas": np.zeros((1, frames, 10), dtype=np.float32),
        "smpl_trans": trans.reshape(1, frames, 3),
    }


#: Share of frames whose lowest foot may still be below the floor after
#: grounding (glitch frames, a foot sliding through the floor).
GROUNDING_PERCENTILE = 5.0


@dataclass(frozen=True)
class SMPLRestSkeleton:
    """What grounding needs from the SMPL body: rest joints (body frame,
    root translation at 0), parents, the foot joints and how far the sole is
    below the lowest of them."""

    joints: np.ndarray
    parents: Sequence[int]
    foot_joints: Sequence[int]
    sole_below_feet_m: float


def ground_smpl_motion(motion: dict, skeleton: SMPLRestSkeleton) -> tuple[dict, float]:
    """Raise or lower a whole SMPL clip by one constant so its soles touch y=0.

    Kimodo's SOMA hip height is used as the SMPL root translation, so the
    SMPL ankles start ~0.13 m below the floor. One offset for the clip (not
    per frame) keeps sitting, crouching and jumping as generated; the floor
    is where the lowest foot is in all but ``GROUNDING_PERCENTILE`` % of frames.
    """
    pose = np.asarray(motion["smpl_pose"], dtype=np.float64)
    trans = np.asarray(motion["smpl_trans"], dtype=np.float64)
    frames = pose.shape[1]
    pose = pose.reshape(frames, -1, 3)
    trans = trans.reshape(frames, 3)
    joints = np.asarray(skeleton.joints, dtype=np.float64)
    count = joints.shape[0]
    rotations = Rotation.from_rotvec(pose[:, :count].reshape(-1, 3)).as_matrix()
    rotations = rotations.reshape(frames, count, 3, 3)
    world_rot = [None] * count
    world_pos = [None] * count
    for joint in range(count):
        parent = skeleton.parents[joint]
        if parent < 0:
            world_rot[joint] = rotations[:, joint]
            world_pos[joint] = trans + joints[joint]
        else:
            offset = joints[joint] - joints[parent]
            world_rot[joint] = world_rot[parent] @ rotations[:, joint]
            world_pos[joint] = world_pos[parent] + world_rot[parent] @ offset
    lowest = np.min([world_pos[joint][:, 1] for joint in skeleton.foot_joints], axis=0)
    floor = float(np.percentile(lowest, GROUNDING_PERCENTILE))
    lift = float(skeleton.sole_below_feet_m) - floor
    grounded = dict(motion)
    shifted = np.array(motion["smpl_trans"], dtype=np.float32, copy=True)
    shifted[..., 1] += np.float32(lift)
    grounded["smpl_trans"] = shifted
    return grounded, lift


def _from_amass(data) -> tuple[np.ndarray, np.ndarray, float]:
    root = np.asarray(data["root_orient"], dtype=np.float32)
    body = np.asarray(data["pose_body"], dtype=np.float32).reshape(root.shape[0], -1, 3)
    if body.shape[1] < 21:
        raise ValueError("Kimodo AMASS pose_body must contain at least 21 body joints")
    pose = np.zeros((root.shape[0], 24, 3), dtype=np.float32)
    pose[:, 0] = root
    pose[:, 1:22] = body[:, :21]
    trans = np.asarray(data["trans"], dtype=np.float32)
    # Kimodo AMASS export is Z-up. WT Studio/SMPL playback is Y-up.
    coord = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    trans = trans @ coord.T
    root_mats = Rotation.from_rotvec(pose[:, 0]).as_matrix()
    pose[:, 0] = Rotation.from_matrix(coord.T @ root_mats).as_rotvec()
    return pose, trans, _fps(data)


def _from_soma(data) -> tuple[np.ndarray, np.ndarray, float]:
    local = np.asarray(data["local_rot_mats"], dtype=np.float32)
    if local.ndim == 5 and local.shape[0] == 1:
        local = local[0]
    if local.ndim != 4 or local.shape[1:] != (77, 3, 3):
        raise ValueError(f"Kimodo SOMA local_rot_mats must have shape (T,77,3,3); got {local.shape}")
    indices = {name: index for index, name in enumerate(_SOMA77_NAMES)}
    pose = np.zeros((local.shape[0], 24, 3), dtype=np.float32)
    for smpl_index, soma_name in _SOMA_TO_SMPL.items():
        matrix = local[:, indices[soma_name]]
        if smpl_index == 15:
            matrix = local[:, indices["Neck2"]] @ matrix
        pose[:, smpl_index] = Rotation.from_matrix(matrix).as_rotvec()
    trans = np.asarray(data["root_positions"], dtype=np.float32)
    if trans.ndim == 3 and trans.shape[0] == 1:
        trans = trans[0]
    if trans.shape != (local.shape[0], 3):
        raise ValueError("Kimodo SOMA root_positions frame count does not match rotations")
    return pose, trans, _fps(data, default=30.0)


def _fps(data, default: float = 30.0) -> float:
    for key in ("mocap_frame_rate", "fps"):
        if key in data.files:
            value = float(np.asarray(data[key]).reshape(-1)[0])
            if value > 0:
                return value
    return default


def _resample(pose: np.ndarray, trans: np.ndarray, source_fps: float, target_fps: float):
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("motion frame rates must be positive")
    frames = pose.shape[0]
    if frames <= 1 or abs(source_fps - target_fps) < 1e-6:
        return pose.astype(np.float32), trans.astype(np.float32)
    duration = (frames - 1) / source_fps
    output_frames = max(2, int(round(duration * target_fps)) + 1)
    source_times = np.arange(frames, dtype=np.float64) / source_fps
    target_times = np.linspace(0.0, duration, output_frames)
    out_pose = np.empty((output_frames, 24, 3), dtype=np.float32)
    for joint in range(24):
        rotations = Rotation.from_rotvec(pose[:, joint])
        out_pose[:, joint] = Slerp(source_times, rotations)(target_times).as_rotvec()
    out_trans = np.column_stack([
        np.interp(target_times, source_times, trans[:, axis]) for axis in range(3)
    ]).astype(np.float32)
    return out_pose, out_trans
