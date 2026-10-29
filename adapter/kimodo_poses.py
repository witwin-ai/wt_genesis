"""End poses for a Kimodo person: sit on a seat, lie on a bed, fall to the floor.

Constraining only the root (x, z) let Kimodo sit at its own seat height: the
first sofa sits sank 0.12-0.17 m. Kimodo pins the pelvis itself with its
``Hips`` end effector (position and orientation) on chosen frames, so the end
pose names where the pelvis rests and how it is turned. The SOMA -> SMPL
retarget still leaves a few centimetres, so the settled pose is grounded once
more on the posed SMPL skin against its support: the buttocks on the seat,
the back on the mattress, the lowest point on the floor after a fall.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

#: Kimodo's constraint files carry the 30-joint SOMA skeleton.
SOMA30_JOINTS = 30
#: SMPL joints whose skin rests on a seat (pelvis, hips) or on a mattress
#: when lying on the back (pelvis and spine).
SEAT_REGION_JOINTS = (0, 1, 2)
BACK_REGION_JOINTS = (0, 3, 6, 9)
#: Ankles and toes: the soles a walk stands on.
FOOT_REGION_JOINTS = (7, 8, 10, 11)
#: A vertex belongs to a region when this share of its weight is on it.
REGION_WEIGHT = 0.5


def upright_pelvis(heading: float) -> np.ndarray:
    """Pelvis rotation (axis-angle) standing or sitting upright, facing
    ``heading`` (Kimodo heading: 0 faces +Z)."""
    return np.array([0.0, float(heading), 0.0])


def supine_pelvis(head_direction: Sequence[float]) -> np.ndarray:
    """Pelvis rotation lying on the back, head towards ``head_direction``
    (x, z) and the face up."""
    dx, dz = float(head_direction[0]), float(head_direction[1])
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        raise ValueError("a head direction must not be zero")
    up = np.array([dx / norm, 0.0, dz / norm])        # body +Y (head)
    forward = np.array([0.0, 1.0, 0.0])                # body +Z (face)
    side = np.cross(up, forward)                       # body +X
    return Rotation.from_matrix(np.column_stack([side, up, forward])).as_rotvec()


@dataclass(frozen=True)
class EndPose:
    """How a walk ends: ``kind`` (sit, lie, fall), where the pelvis rests
    (``hips`` (x, y, z) in scene coordinates, y above the floor; ``rotvec`` its
    Kimodo orientation) for the last ``hold_s`` seconds, and what the settled
    body rests on: the lowest vertex of ``region_joints`` (None: the whole
    body) goes on ``support_height`` over the last ``rest_s`` seconds. No
    ``hips`` leaves the pose to Kimodo (a fall); no ``support_height`` skips
    the grounding."""

    kind: str
    hips: tuple[float, float, float] | None = None
    rotvec: tuple[float, float, float] | None = None
    hold_s: float = 0.5
    support_height: float | None = None
    region_joints: tuple[int, ...] | None = None
    rest_s: float = 0.6

    def constraint(self, frames: Sequence[int], origin: Sequence[float]) -> dict:
        """The Kimodo ``end-effector`` constraint (Hips) on ``frames``,
        relative to the route start ``origin`` (x, z)."""
        x, y, z = (float(value) for value in self.hips)
        local = [[list(map(float, self.rotvec))] + [[0.0, 0.0, 0.0]] * (SOMA30_JOINTS - 1)
                 for _ in frames]
        relative = [x - float(origin[0]), y, z - float(origin[1])]
        return {"type": "end-effector", "joint_names": ["Hips"],
                "frame_indices": [int(index) for index in frames],
                "local_joints_rot": local,
                "root_positions": [list(relative) for _ in frames],
                "smooth_root_2d": [[relative[0], relative[2]] for _ in frames]}


@dataclass(frozen=True)
class SMPLSkin:
    """The rest SMPL body for linear blend skinning: vertices and joints in
    the body frame (root translation 0), parents, 4 bone indices/weights per
    vertex."""

    vertices: np.ndarray
    joints: np.ndarray
    parents: Sequence[int]
    skin_indices: np.ndarray
    skin_weights: np.ndarray

    def region(self, joints: Sequence[int] | None) -> np.ndarray:
        """Vertex indices with at least ``REGION_WEIGHT`` on ``joints`` (all for None)."""
        if joints is None:
            return np.arange(len(self.vertices))
        share = np.where(np.isin(self.skin_indices, list(joints)), self.skin_weights, 0.0).sum(axis=1)
        return np.flatnonzero(share >= REGION_WEIGHT)


def _world_joints(pose: np.ndarray, trans: np.ndarray, skin: SMPLSkin):
    frames, count = pose.shape[0], len(skin.joints)
    local = Rotation.from_rotvec(pose[:, :count].reshape(-1, 3)).as_matrix().reshape(frames, count, 3, 3)
    rotations = [None] * count
    positions = [None] * count
    for joint in range(count):
        parent = skin.parents[joint]
        if parent < 0:
            rotations[joint] = local[:, joint]
            positions[joint] = trans + skin.joints[joint]
        else:
            rotations[joint] = rotations[parent] @ local[:, joint]
            positions[joint] = positions[parent] + rotations[parent] @ (skin.joints[joint] - skin.joints[parent])
    return np.stack(rotations, axis=1), np.stack(positions, axis=1)


def posed_vertices(pose, trans, skin: SMPLSkin, vertex_ids=None) -> np.ndarray:
    """(frames, V, 3) skinned vertices for SMPL ``pose`` (frames, 24, 3) and
    root ``trans`` (frames, 3)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(len(trans), -1, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    ids = np.arange(len(skin.vertices)) if vertex_ids is None else np.asarray(vertex_ids)
    rotations, positions = _world_joints(pose, trans, skin)
    rest = skin.vertices[ids]
    out = np.zeros((len(trans), len(ids), 3))
    for slot in range(skin.skin_indices.shape[1]):
        bones = skin.skin_indices[ids, slot]
        weights = skin.skin_weights[ids, slot][None, :, None]
        offset = rest - skin.joints[bones]
        moved = np.einsum("fvij,vj->fvi", rotations[:, bones], offset) + positions[:, bones]
        out += weights * moved
    return out


def ground_end_pose(motion: dict, skin: SMPLSkin, *, support_height: float,
                    region_joints: Sequence[int] | None, transition: Sequence[float],
                    rest: Sequence[float], fps: float,
                    foot_joints: Sequence[int] = FOOT_REGION_JOINTS) -> tuple[dict, dict]:
    """Stand the walk on the floor and rest the settled pose on its support.

    Before ``transition`` the lowest sole is put on the floor (the median over
    the walk: a walk always has a foot down); in the ``rest`` window (seconds)
    the lowest vertex of the ``region`` (buttocks, back, or the whole body)
    goes onto ``support_height``. The two corrections are blended over
    ``transition``. Afterwards no frame may have a vertex below the floor.
    One clip-wide offset could not do both: after the SOMA -> SMPL retarget
    the seated feet sit lower than the walking ones, so the walk floated.
    """
    pose = np.asarray(motion["smpl_pose"], dtype=np.float64)
    trans = np.array(motion["smpl_trans"], dtype=np.float64, copy=True)
    frames = pose.shape[1]
    pose = pose.reshape(frames, -1, 3)
    root = trans.reshape(frames, 3)
    times = np.arange(frames) / float(fps)
    ids = skin.region(region_joints)
    resting = (times >= rest[0] - 1e-9) & (times <= rest[1] + 1e-9)
    if not resting.any():
        resting[-1] = True
    lowest = posed_vertices(pose[resting], root[resting], skin, ids)[:, :, 1].min(axis=1)
    gap = float(np.median(lowest)) - float(support_height)
    start, end = float(transition[0]), float(transition[1])
    walking = times < start - 1e-9
    walk_level = 0.0
    sole_ids = skin.region(foot_joints)
    if walking.any() and len(sole_ids):
        soles = posed_vertices(pose[walking], root[walking], skin, sole_ids)
        walk_level = float(np.median(soles[:, :, 1].min(axis=1)))
    blend = np.clip((times - start) / max(end - start, 1e-9), 0.0, 1.0)
    root[:, 1] += -walk_level * (1.0 - blend) - gap * blend
    floor = posed_vertices(pose, root, skin)[:, :, 1].min(axis=1)
    lifted = floor < 0.0
    root[lifted, 1] += -floor[lifted]
    # What the settled pose ends as (the floor guard can hold it up).
    settled = posed_vertices(pose[resting], root[resting], skin)
    support_gap = float(np.median(settled[:, ids, 1].min(axis=1))) - float(support_height)
    feet = float(np.median(settled[:, sole_ids, 1].min(axis=1))) if len(sole_ids) else None
    grounded = dict(motion)
    grounded["smpl_trans"] = root.reshape(np.asarray(motion["smpl_trans"]).shape).astype(np.float32)
    report = {"support_height_m": round(float(support_height), 4),
              "walk_correction_m": round(-walk_level, 4),
              "rest_gap_before_m": round(gap, 4), "rest_correction_m": round(-gap, 4),
              "rest_support_gap_m": round(support_gap, 4),
              "rest_feet_lowest_m": None if feet is None else round(feet, 4),
              "frames_lifted_off_floor": int(lifted.sum()),
              "deepest_floor_lift_m": round(float(-floor[lifted].min()) if lifted.any() else 0.0, 4)}
    return grounded, report
