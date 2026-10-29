"""Hold a Kimodo person's planted feet still on the floor (foot-skate cleanup).

Kimodo makes a SOMA body; the retarget copies its joint rotations onto the
SMPL skeleton and keeps SOMA's root path. SMPL's legs are not SOMA's, so a
planted SMPL foot slides: on the meeting walks a heel moved 6-15 cm while it
stood on the floor (0.1-0.3 m/s), and the user rejects any sliding.

The cleanup works on the SMPL clip itself. A heel (the rear third of the
sole) or a toe (the front third) is planted while it is within
``contact_m`` of its own resting height and slower than ``planted_mps``
(a swinging foot that only skims the floor is not planted). While planted,
the pivot (the heel from heel strike to heel off, then the toe until toe
off) is held where it was put down: the ankle goal is moved by the pivot's
slide, and hip and knee reach it by two-bone IK in the leg's own plane,
the foot keeping its world orientation (the roll from heel to toe stays as
Kimodo made it). The root and the upper body are not touched. When the
foot lifts, the correction fades out over ``release_s`` while it swings.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .kimodo_poses import SMPLSkin, _turn, _world_joints, posed_vertices

#: SMPL hip, knee, ankle, toe joints per leg.
LEGS = {"L": (1, 4, 7, 10), "R": (2, 5, 8, 11)}
SOLE_BAND_M = 0.03
CONTACT_M = 0.01
PLANTED_MPS = 0.5
MIN_CONTACT_S = 0.1
RELEASE_S = 0.25
#: A leg is never stretched past this share of its full length.
MAX_REACH = 0.995


def sole_points(skin: SMPLSkin) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{side: (heel vertex ids, toe vertex ids)}``: the sole (the lowest
    ``SOLE_BAND_M`` of the foot's vertices at rest, SMPL faces +z), its rear
    and front thirds."""
    dominant = skin.skin_indices[np.arange(len(skin.skin_indices)), skin.skin_weights.argmax(axis=1)]
    out = {}
    for side, (_hip, _knee, ankle, toe) in LEGS.items():
        ids = np.flatnonzero(np.isin(dominant, (ankle, toe)))
        rest = skin.vertices[ids]
        sole = ids[rest[:, 1] < rest[:, 1].min() + SOLE_BAND_M]
        z = skin.vertices[sole, 2]
        low, high = float(z.min()), float(z.max())
        out[side] = (sole[z < low + 0.35 * (high - low)], sole[z > high - 0.35 * (high - low)])
    return out


def _runs(flags, min_frames):
    runs, start = [], None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_frames:
                runs.append((start, index))
            start = None
    return runs


def planted(height, xz, fps, *, contact_m=CONTACT_M, planted_mps=PLANTED_MPS,
            min_contact_s=MIN_CONTACT_S):
    """Frames a sole point is planted: near its resting height (its 5th
    percentile over the clip) and slow, in runs of at least ``min_contact_s``."""
    speed = np.linalg.norm(np.gradient(xz, axis=0), axis=1) * fps
    flags = (height < np.percentile(height, 5) + contact_m) & (speed < planted_mps)
    out = np.zeros(len(height), dtype=bool)
    for start, end in _runs(flags, max(2, int(round(min_contact_s * fps)))):
        out[start:end] = True
    return out


def _smooth_fade(x):
    x = min(max(x, 0.0), 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)


def pivot_corrections(heel_xz, toe_xz, heel_down, toe_down, fps, release_s=RELEASE_S):
    """Per frame the horizontal shift (frames, 2) that holds the planted pivot
    where it was put down; it fades out over ``release_s`` once the foot lifts."""
    frames = len(heel_xz)
    delta = np.zeros((frames, 2))
    pivot, hold, lifted, carried = None, None, None, np.zeros(2)
    for frame in range(frames):
        now = "heel" if heel_down[frame] else "toe" if toe_down[frame] else None
        orig = heel_xz[frame] if now == "heel" else toe_xz[frame] if now == "toe" else None
        if now is not None:
            if now != pivot:
                # Put down (or rolled onto the toe): hold it where it is now,
                # with the correction it already carries (no jump).
                previous = delta[frame - 1] if frame else np.zeros(2)
                hold = orig + previous
            delta[frame] = hold - orig
            lifted, carried = None, delta[frame]
        else:
            if lifted is None:
                lifted = frame
            delta[frame] = carried * _smooth_fade((frame - lifted + 1) / (release_s * fps))
        pivot = now
    return delta


def _reach(pose, rotations, positions, skin, frame, hip, knee, ankle, goal):
    """Two-bone IK of one leg at one frame toward ``goal`` (the ankle), the
    foot keeping its world orientation. Returns how far the ankle stays from
    the goal (the leg would have to stretch)."""
    h, k, a = positions[frame, hip], positions[frame, knee], positions[frame, ankle]
    thigh, shin = float(np.linalg.norm(k - h)), float(np.linalg.norm(a - k))
    reach = float(np.linalg.norm(goal - h))
    limit = MAX_REACH * (thigh + shin)
    residual = 0.0
    if reach > limit:
        target = h + (goal - h) * (limit / reach)
        residual, goal, reach = float(np.linalg.norm(goal - target)), target, limit
    unit = (goal - h) / reach
    along = (thigh * thigh - shin * shin + reach * reach) / (2.0 * reach)
    height = math.sqrt(max(thigh * thigh - along * along, 0.0))
    bend = (k - h) - np.dot(k - h, unit) * unit
    if np.linalg.norm(bend) < 1e-6:
        # A straight leg: bend the knee forward, as it bends.
        forward = rotations[frame, skin.parents[hip]] @ np.array([0.0, 0.0, 1.0])
        bend = forward - np.dot(forward, unit) * unit
    bend /= np.linalg.norm(bend)
    new_knee = h + along * unit + height * bend
    hip_world = _turn(k - h, new_knee - h) @ rotations[frame, hip]
    shin_world = _turn(a - k, goal - new_knee) @ rotations[frame, knee]
    foot_world = rotations[frame, ankle]
    parent = rotations[frame, skin.parents[hip]]
    pose[frame, hip] = Rotation.from_matrix(parent.T @ hip_world).as_rotvec()
    pose[frame, knee] = Rotation.from_matrix(hip_world.T @ shin_world).as_rotvec()
    pose[frame, ankle] = Rotation.from_matrix(shin_world.T @ foot_world).as_rotvec()
    return residual


def lock_planted_feet(pose, trans, skin: SMPLSkin, fps: float, *,
                      frames: Sequence[int] | None = None,
                      contact_m: float = CONTACT_M, planted_mps: float = PLANTED_MPS,
                      release_s: float = RELEASE_S) -> tuple[np.ndarray, dict]:
    """``(pose, report)``: ``pose`` (frames, J, 3) with planted heels and toes
    held still (module doc). ``frames`` limits the cleanup to those frames
    (a sit's legs, already placed by the grounding, are left alone)."""
    pose = np.array(pose, dtype=np.float64, copy=True).reshape(len(trans), -1, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    if len(trans) < 3:
        return pose.astype(np.float32), {}
    allowed = np.ones(len(trans), dtype=bool)
    if frames is not None:
        allowed[:] = False
        allowed[np.asarray(frames, dtype=int)] = True
    rotations, positions = _world_joints(pose, trans, skin)
    report = {}
    for side, (heel_ids, toe_ids) in sole_points(skin).items():
        hip, knee, ankle, _toe = LEGS[side]
        heel = posed_vertices(pose, trans, skin, heel_ids)
        toe = posed_vertices(pose, trans, skin, toe_ids)
        heel_xz, toe_xz = heel[:, :, [0, 2]].mean(axis=1), toe[:, :, [0, 2]].mean(axis=1)
        heel_down = planted(heel[:, :, 1].min(axis=1), heel_xz, fps, contact_m=contact_m,
                            planted_mps=planted_mps) & allowed
        toe_down = planted(toe[:, :, 1].min(axis=1), toe_xz, fps, contact_m=contact_m,
                           planted_mps=planted_mps) & allowed
        delta = pivot_corrections(heel_xz, toe_xz, heel_down, toe_down, fps, release_s)
        delta[~allowed] = 0.0
        residual = 0.0
        for frame in np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-5):
            goal = positions[frame, ankle] + np.array([delta[frame, 0], 0.0, delta[frame, 1]])
            residual = max(residual, _reach(pose, rotations, positions, skin, frame,
                                            hip, knee, ankle, goal))
        report[side] = {"planted_frames": int((heel_down | toe_down).sum()),
                        "max_correction_m": round(float(np.linalg.norm(delta, axis=1).max()), 4),
                        "max_unreached_m": round(residual, 4)}
    return pose.astype(np.float32), report


#: A foot this far under the floor is raised; the raise is eased in and out
#: over ``DIP_EASE_S`` so the leg does not jump.
DIP_TOLERANCE_M = 0.002
DIP_EASE_S = 0.15


def lift_dipping_feet(pose, trans, skin: SMPLSkin, fps: float, *, floor: float = 0.0,
                      frames: Sequence[int] | None = None) -> tuple[np.ndarray, dict]:
    """``(pose, report)``: a swinging toe that skims through the floor (Kimodo
    dipped one 1.4 cm) is raised onto it by its own leg (two-bone IK, the foot
    keeping its orientation); the body is not lifted."""
    from .kimodo_poses import raise_seated_legs

    pose = np.array(pose, dtype=np.float64, copy=True).reshape(len(trans), -1, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    dominant = skin.skin_indices[np.arange(len(skin.skin_indices)), skin.skin_weights.argmax(axis=1)]
    ease = max(1, int(round(DIP_EASE_S * fps)))
    raise_m = np.zeros((len(trans), 2))
    for leg, side in enumerate(("L", "R")):
        _hip, _knee, ankle, toe = LEGS[side]
        ids = np.flatnonzero(np.isin(dominant, (ankle, toe)))
        low = posed_vertices(pose, trans, skin, ids)[:, :, 1].min(axis=1)
        dip = np.where(floor - low > DIP_TOLERANCE_M, floor - low, 0.0)
        if frames is not None:
            keep = np.zeros(len(dip), dtype=bool)
            keep[np.asarray(frames, dtype=int)] = True
            dip[~keep] = 0.0
        # Hold each dip's depth around it and ease it in and out (no pop).
        held = np.array([dip[max(0, i - ease):i + ease + 1].max() for i in range(len(dip))])
        kernel = np.hanning(2 * ease + 3)[1:-1]
        pad = ease + 1
        smooth = np.convolve(np.pad(held, pad, mode="edge"), kernel / kernel.sum(),
                             mode="same")[pad:pad + len(held)]
        raise_m[:, leg] = np.maximum(smooth, dip)
    report = {"max_dip_m": round(float(raise_m.max()), 4),
              "raised_frames": int((raise_m > 1e-6).any(axis=1).sum())}
    if report["raised_frames"]:
        pose = raise_seated_legs(pose, trans, skin, raise_m)
    return pose.astype(np.float32), report
