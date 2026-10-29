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
    #: The support's footprint (x0, z0, x1, z1) in the scene: no frame may
    #: sink into it on the way to the end pose.
    support_rect: tuple[float, float, float, float] | None = None
    #: A sit also plants both feet (Kimodo ``LeftFoot``/``RightFoot`` end
    #: effectors) where a seated leg pose puts them: shins vertical, soles flat
    #: on the floor, ahead of the seat. With the pelvis alone Kimodo kept its
    #: feet ~0.1 m ahead of the pelvis and the shins came down inside the seat
    #: front (10.5 cm into an armchair, 10.1 cm into a bed side).
    plant_feet: bool = False

    def constraint(self, frames: Sequence[int], origin: Sequence[float]) -> dict:
        """The Kimodo ``end-effector`` constraint (Hips, and the feet with
        :attr:`plant_feet`) on ``frames``, relative to the route start
        ``origin`` (x, z)."""
        x, y, z = (float(value) for value in self.hips)
        pose = [[0.0, 0.0, 0.0]] * SOMA30_JOINTS
        pose[0] = list(map(float, self.rotvec))
        names = ["Hips"]
        if self.plant_feet:
            for joint, rotvec in seated_leg_rotations(y).items():
                pose[joint] = rotvec
            names += ["LeftFoot", "RightFoot"]
        local = [list(pose) for _ in frames]
        relative = [x - float(origin[0]), y, z - float(origin[1])]
        return {"type": "end-effector", "joint_names": names,
                "frame_indices": [int(index) for index in frames],
                "local_joints_rot": local,
                "root_positions": [list(relative) for _ in frames],
                "smooth_root_2d": [[relative[0], relative[2]] for _ in frames]}


#: SOMA30 legs (Kimodo's bone order: LeftLeg 22, LeftShin 23, RightLeg 26,
#: RightShin 27) and their rest lengths, from its rest skeleton: the hip
#: joints 0.084 m under and 0.026 m ahead of Hips, thigh 0.433 m, shin
#: 0.421 m (knee to the LeftFoot/ankle joint). Rest frames are aligned with
#: the body (+Z forward), so a hip turned -a about local X lifts the thigh
#: forward and the shin turned +a brings it back to vertical.
SOMA_LEGS = {"left": (22, 23), "right": (26, 27)}
SOMA_HIP_DROP_M = 0.084
SOMA_THIGH_M = 0.433
SOMA_SHIN_M = 0.421
#: The ankle (LeftFoot) joint over the floor with the sole flat: the toe
#: joint is 0.051 m under it at rest, the sole a little lower. At 0.075 m the
#: retargeted SMPL soles came 3.2 cm under the floor with the buttocks on a
#: 0.35 m bed (the floor guard then held the buttocks 3.2 cm up; bench_7107).
SOMA_ANKLE_OVER_FLOOR_M = 0.11


def seated_leg_rotations(hips_y: float, floor_y: float = 0.0) -> dict[int, list[float]]:
    """Local axis-angle rotations of both SOMA legs for a seated pose with
    the Hips joint at ``hips_y``: shins vertical, feet flat, the thighs tilted
    down from horizontal just enough for the soles to reach the floor (up,
    knees over the hips, on a seat lower than the shins)."""
    drop = float(hips_y) - float(floor_y) - SOMA_HIP_DROP_M - SOMA_ANKLE_OVER_FLOOR_M
    tilt = math.asin(min(max((drop - SOMA_SHIN_M) / SOMA_THIGH_M, -0.6), 0.8))
    lift = math.pi / 2 - tilt
    out = {}
    for hip, knee in SOMA_LEGS.values():
        out[hip] = [-lift, 0.0, 0.0]
        out[knee] = [lift, 0.0, 0.0]
    return out


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
                    foot_joints: Sequence[int] = FOOT_REGION_JOINTS,
                    support_rect: Sequence[float] | None = None,
                    support_tolerance: float = 0.02,
                    smooth_s: float = 0.15, lift_feet: bool = False) -> tuple[dict, dict]:
    """Stand the walk on the floor and rest the settled pose on its support.

    Before ``transition`` the lowest sole is put on the floor (the median over
    the walk: a walk always has a foot down); in the ``rest`` window (seconds)
    the lowest vertex of the ``region`` (buttocks, back, or the whole body)
    goes onto ``support_height``. The two corrections are blended over
    ``transition``. Afterwards no frame may have a vertex below the floor.
    One clip-wide offset could not do both: after the SOMA -> SMPL retarget
    the seated feet sit lower than the walking ones, so the walk floated.

    On the way there no frame may sink into the support: the region's skin
    over its footprint ``support_rect`` (x0, z0, x1, z1) stays within
    ``support_tolerance`` of its top (getting onto a bed went 16 cm into the
    mattress), and no skin goes below the floor. The lift this takes is
    smoothed over ``smooth_s`` (a running maximum: never less than needed).
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
    legs_raised = 0.0
    if lift_feet and len(sole_ids):
        # Buttocks on the seat put the soles under the floor (a seat lower
        # than the SMPL lower leg): bend the legs, knees up, so the soles
        # come up onto the floor instead of the floor guard holding the
        # whole body off the seat (5.5 cm, bed_7601 couch).
        # Raising the thighs lifts their undersides off the seat too: settle
        # the buttocks and raise the soles again, twice.
        for _round in range(3):
            soles = posed_vertices(pose[resting], root[resting], skin, sole_ids)
            under = -float(np.median(soles[:, :, 1].min(axis=1)))
            if under <= 0.002 or legs_raised >= MAX_LEG_RAISE_M:
                break
            step = min(under, MAX_LEG_RAISE_M - legs_raised)
            legs_raised += step
            pose = raise_seated_legs(pose, root, skin, step * blend)
            seated = posed_vertices(pose[resting], root[resting], skin, ids)
            over = float(np.median(seated[:, :, 1].min(axis=1))) - float(support_height)
            if over > 0.002:
                root[:, 1] -= over * blend
    everything = posed_vertices(pose, root, skin)
    floor = everything[:, :, 1].min(axis=1)
    need = np.maximum(-floor, 0.0)
    onto_support = np.zeros(frames, dtype=bool)
    if support_rect is not None:
        # Only the support region is held up: feet may stand on the floor
        # under a seat's front edge or beside a bed (they lifted a seated
        # person 30 cm when every vertex counted).
        x0, z0, x1, z1 = (float(value) for value in support_rect)
        carried = everything[:, ids]
        over = ((carried[:, :, 0] >= x0) & (carried[:, :, 0] <= x1)
                & (carried[:, :, 2] >= z0) & (carried[:, :, 2] <= z1))
        depth = np.where(over, float(support_height) - support_tolerance - carried[:, :, 1], 0.0)
        sink = np.maximum(depth.max(axis=1), 0.0)
        onto_support = sink > 0.0
        need = np.maximum(need, sink)
    width = int(round(float(smooth_s) * float(fps)))
    if width > 0:
        padded = np.pad(need, width, mode="edge")
        need = np.array([padded[index:index + 2 * width + 1].max() for index in range(frames)])
    lifted = need > 0.0
    root[:, 1] += need
    # What the settled pose ends as (the floor guard can hold it up).
    settled = posed_vertices(pose[resting], root[resting], skin)
    support_gap = float(np.median(settled[:, ids, 1].min(axis=1))) - float(support_height)
    feet = float(np.median(settled[:, sole_ids, 1].min(axis=1))) if len(sole_ids) else None
    grounded = dict(motion)
    grounded["smpl_trans"] = root.reshape(np.asarray(motion["smpl_trans"]).shape).astype(np.float32)
    grounded["smpl_pose"] = pose.reshape(np.asarray(motion["smpl_pose"]).shape).astype(
        np.asarray(motion["smpl_pose"]).dtype)
    report = {"support_height_m": round(float(support_height), 4),
              "walk_correction_m": round(-walk_level, 4),
              "rest_gap_before_m": round(gap, 4), "rest_correction_m": round(-gap, 4),
              "rest_support_gap_m": round(support_gap, 4),
              "rest_feet_lowest_m": None if feet is None else round(feet, 4),
              "frames_lifted_off_floor": int((floor < 0.0).sum()),
              "frames_lifted_onto_support": int(onto_support.sum()),
              "deepest_floor_lift_m": round(float(-floor.min()) if (floor < 0.0).any() else 0.0, 4),
              "largest_lift_m": round(float(need.max()) if lifted.any() else 0.0, 4),
              "legs_raised_m": round(legs_raised, 4)}
    return grounded, report


#: SMPL legs (hip, knee, ankle joints) and the most a seated knee is raised.
SMPL_LEGS = ((1, 4, 7), (2, 5, 8))
MAX_LEG_RAISE_M = 0.10


def raise_seated_legs(pose: np.ndarray, trans: np.ndarray, skin: SMPLSkin,
                      raise_m: np.ndarray) -> np.ndarray:
    """``pose`` (frames, J, 3) with each thigh turned up about its hip (knee
    up by ``raise_m`` per frame) and the shin and foot kept as they were in
    the world: the foot rises by the same amount, flat as before."""
    pose = np.array(pose, dtype=np.float64, copy=True)
    raise_m = np.asarray(raise_m, dtype=np.float64).reshape(-1)
    rotations, positions = _world_joints(pose, trans, skin)
    for frame in np.flatnonzero(raise_m > 1e-6):
        for hip, knee, _ankle in SMPL_LEGS:
            thigh = positions[frame, knee] - positions[frame, hip]
            length = float(np.linalg.norm(thigh))
            flat = math.hypot(float(thigh[0]), float(thigh[2]))
            if length < 1e-6 or flat < 1e-6:
                continue
            elevation = math.atan2(float(thigh[1]), flat)
            target = min(math.sin(elevation) + float(raise_m[frame]) / length, 0.99)
            turn = math.asin(target) - elevation
            # The hip turns about its own side axis (horizontal, across the thigh).
            side = np.cross(thigh / length, [0.0, 1.0, 0.0])
            side /= np.linalg.norm(side)
            spin = Rotation.from_rotvec(-side * turn).as_matrix()
            # Sign: the turn must lift the knee; flip the axis if it lowers it.
            if (spin @ thigh)[1] < thigh[1]:
                spin = spin.T
            parent = rotations[frame, skin.parents[hip]]
            hip_world = spin @ rotations[frame, hip]
            shin_world = rotations[frame, knee]
            pose[frame, hip] = Rotation.from_matrix(parent.T @ hip_world).as_rotvec()
            pose[frame, knee] = Rotation.from_matrix(hip_world.T @ shin_world).as_rotvec()
    return pose
