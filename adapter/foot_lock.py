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
off) is held where it was put down, and a flat foot (heel and toe down)
keeps its heading too: the ankle goal is moved by the pivot's slide, and
hip and knee reach it by two-bone IK in the leg's own plane, the foot
keeping its world orientation but for that heading (the roll from heel to
toe stays as Kimodo made it). Each contact is held where the foot was on
average over it; in the swing the correction blends to the next contact's.
The upper body is not touched; the root comes down at most 3 cm where a
held foot is out of the leg's reach.

Also here, the other clean-ups of a Kimodo clip on SMPL: a swinging toe
dipping through the floor raised by its leg (``lift_dipping_feet``), a
standing foot left in the air set down (``set_down_hovering_feet``) and
the start glitch of a route walk's first frame (``settle_first_frame``).
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
#: Looser than a skate check's planted test (1 cm, 0.5 m/s), so a touchdown
#: is caught before it slows.
CONTACT_M = 0.015
PLANTED_MPS = 0.8
#: The root comes down at most this far for a held foot out of reach.
MAX_ROOT_DROP_M = 0.03
MIN_CONTACT_S = 0.1
RELEASE_S = 0.25
#: A leg is never stretched past this share of its full length.
MAX_REACH = 0.995
#: Raises, set-downs and root drops are eased in and out over this.
DIP_EASE_S = 0.15
#: A flat foot held against Kimodo turning it more than this pivots on its toe.
MAX_TURN_RAD = math.radians(10.0)


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


def _turn_xz(angle, vectors):
    """(x, z) vectors turned by ``angle`` about the vertical (+z toward +x)."""
    c, s = math.cos(angle), math.sin(angle)
    v = np.asarray(vectors, dtype=np.float64)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], axis=-1)


def pivot_corrections(ankle_xz, heel_xz, toe_xz, yaw, heel_down, toe_down, fps,
                      release_s=RELEASE_S, max_turn=MAX_TURN_RAD):
    """Per frame ``(shift (frames, 2), turn (frames,))``: the ankle's
    horizontal shift and the foot's extra turn about the vertical that hold
    the planted pivot where it was put down. Heel only down: the heel is held
    (the foot rolls about it); toe only: the toe; both (foot flat): the sole
    centre and the foot's heading, so neither end creeps. A flat foot that
    Kimodo turns more than ``max_turn`` on the spot pivots on its toe instead
    (the leg would twist).

    Each contact is held where Kimodo had the foot on average over it (not
    where it touched down): the correction is centred, half the slide at
    either end, so the leg reaches it. In the swing between two contacts the
    correction blends from the one to the next; before the first and after
    the last it eases in and out over ``release_s``."""
    frames = len(ankle_xz)
    shift, turn = np.zeros((frames, 2)), np.zeros(frames)
    state, hold, heading = None, None, 0.0
    swing = np.zeros(frames, dtype=bool)

    def pivot(kind, at):
        return {"heel": heel_xz[at], "toe": toe_xz[at],
                "flat": (heel_xz[at] + toe_xz[at]) / 2}[kind]

    for frame in range(frames):
        now = ("flat" if heel_down[frame] and toe_down[frame] else "heel" if heel_down[frame]
               else "toe" if toe_down[frame] else None)
        ankle = ankle_xz[frame]
        previous = (shift[frame - 1], turn[frame - 1]) if frame else (np.zeros(2), 0.0)
        if now is None:
            swing[frame], state = True, None
            continue
        if now == "flat" and state == "flat" \
                and abs(math.remainder(heading - yaw[frame], 2 * math.pi)) > max_turn:
            now = "toe"
        if now != state:
            # Put down, rolled or flattened: hold the pivot where the corrected
            # foot had it on the frame before (no jump).
            before = max(frame - 1, 0)
            hold = (ankle_xz[before] + previous[0]
                    + _turn_xz(previous[1], pivot(now, before) - ankle_xz[before]))
            heading = yaw[before] + previous[1]
        angle = (math.remainder(heading - yaw[frame], 2 * math.pi) if now == "flat"
                 else previous[1])
        shift[frame] = hold - ankle - _turn_xz(angle, pivot(now, frame) - ankle)
        turn[frame] = angle
        state = now
    contacts = _runs(~swing, 1)
    for start, end in contacts:
        shift[start:end] -= shift[start:end].mean(axis=0)
    ease = max(1.0, release_s * fps)
    gaps = _runs(swing, 1)
    for start, end in gaps:
        before = (shift[start - 1], turn[start - 1]) if start > 0 else None
        after = (shift[end], turn[end]) if end < frames else None
        for frame in range(start, end):
            if before is not None and after is not None:
                weight = 1.0 - _smooth_fade((frame - start + 1) / (end - start + 1))
                shift[frame] = before[0] * (1 - weight) + after[0] * weight
                turn[frame] = before[1] * (1 - weight) + after[1] * weight
            elif before is not None:
                fade = _smooth_fade((frame - start + 1) / ease)
                shift[frame], turn[frame] = before[0] * fade, before[1] * fade
            elif after is not None:
                fade = _smooth_fade((end - frame) / ease)
                shift[frame], turn[frame] = after[0] * fade, after[1] * fade
    return shift, turn


def _reach(pose, rotations, positions, skin, frame, hip, knee, ankle, goal, foot_world=None):
    """Two-bone IK of one leg at one frame toward ``goal`` (the ankle), the
    foot turned to ``foot_world`` (default: as it was). Returns how far the
    ankle stays from the goal (the leg would have to stretch)."""
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
    foot_world = rotations[frame, ankle] if foot_world is None else foot_world
    parent = rotations[frame, skin.parents[hip]]
    pose[frame, hip] = Rotation.from_matrix(parent.T @ hip_world).as_rotvec()
    pose[frame, knee] = Rotation.from_matrix(hip_world.T @ shin_world).as_rotvec()
    pose[frame, ankle] = Rotation.from_matrix(shin_world.T @ foot_world).as_rotvec()
    return residual


def _drop_to_reach(hip, goal, length):
    """How far the hip must come down for a leg of ``length`` to reach ``goal``."""
    v = goal - hip
    flat = float(v[0] ** 2 + v[2] ** 2)
    if float(np.linalg.norm(v)) <= length:
        return 0.0
    return max(0.0, -float(v[1]) - math.sqrt(max(length * length - flat, 0.0)))


def _ease_hold(values, ease):
    """``values`` held around each peak and eased in and out over ``ease``
    frames, never below the values themselves."""
    held = np.array([values[max(0, i - ease):i + ease + 1].max() for i in range(len(values))])
    kernel = np.hanning(2 * ease + 3)[1:-1]
    pad = ease + 1
    smooth = np.convolve(np.pad(held, pad, mode="edge"), kernel / kernel.sum(),
                         mode="same")[pad:pad + len(held)]
    return np.maximum(smooth, values)


def lock_planted_feet(pose, trans, skin: SMPLSkin, fps: float, *,
                      frames: Sequence[int] | None = None,
                      contact_m: float = CONTACT_M, planted_mps: float = PLANTED_MPS,
                      release_s: float = RELEASE_S) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(pose, trans, report)``: ``pose`` (frames, J, 3) with planted heels
    and toes held still (module doc). Where a held foot is out of the leg's
    reach (a long stride, the leg straight), the root comes down just enough
    (eased over ``DIP_EASE_S``) and both feet keep their places in the world.
    ``frames`` limits the cleanup to those frames (a sit's legs, already
    placed by the grounding, are left alone)."""
    pose = np.array(pose, dtype=np.float64, copy=True).reshape(len(trans), -1, 3)
    trans = np.array(trans, dtype=np.float64, copy=True).reshape(-1, 3)
    if len(trans) < 3:
        return pose.astype(np.float32), trans.astype(np.float32), {}
    allowed = np.ones(len(trans), dtype=bool)
    if frames is not None:
        allowed[:] = False
        allowed[np.asarray(frames, dtype=int)] = True
    rotations, positions = _world_joints(pose, trans, skin)
    report, goals = {}, {}
    for side, (heel_ids, toe_ids) in sole_points(skin).items():
        hip, knee, ankle, _toe = LEGS[side]
        heel = posed_vertices(pose, trans, skin, heel_ids)
        toe = posed_vertices(pose, trans, skin, toe_ids)
        heel_xz, toe_xz = heel[:, :, [0, 2]].mean(axis=1), toe[:, :, [0, 2]].mean(axis=1)
        heel_down = planted(heel[:, :, 1].min(axis=1), heel_xz, fps, contact_m=contact_m,
                            planted_mps=planted_mps) & allowed
        toe_down = planted(toe[:, :, 1].min(axis=1), toe_xz, fps, contact_m=contact_m,
                           planted_mps=planted_mps) & allowed
        forward = rotations[:, ankle] @ np.array([0.0, 0.0, 1.0])
        yaw = np.arctan2(forward[:, 0], forward[:, 2])
        shift, turn = pivot_corrections(positions[:, ankle][:, [0, 2]], heel_xz, toe_xz, yaw,
                                        heel_down, toe_down, fps, release_s)
        shift[~allowed], turn[~allowed] = 0.0, 0.0
        ankle_goal = positions[:, ankle] + np.stack(
            [shift[:, 0], np.zeros(len(shift)), shift[:, 1]], axis=1)
        foot_goal = Rotation.from_rotvec(
            np.stack([np.zeros(len(turn)), turn, np.zeros(len(turn))], axis=1)).as_matrix() \
            @ rotations[:, ankle]
        length = float(np.linalg.norm(skin.joints[knee] - skin.joints[hip])
                       + np.linalg.norm(skin.joints[ankle] - skin.joints[knee]))
        drop = np.array([_drop_to_reach(positions[f, hip], ankle_goal[f], MAX_REACH * length)
                         if heel_down[f] or toe_down[f] else 0.0 for f in range(len(trans))])
        goals[side] = (ankle_goal, foot_goal, drop)
        report[side] = {"planted_frames": int((heel_down | toe_down).sum()),
                        "max_correction_m": round(float(np.linalg.norm(shift, axis=1).max()), 4),
                        "max_turn_deg": round(float(np.degrees(np.abs(turn).max())), 1)}
    ease = max(1, int(round(DIP_EASE_S * fps)))
    drop = _ease_hold(np.minimum(np.maximum(goals["L"][2], goals["R"][2]), MAX_ROOT_DROP_M), ease)
    drop[~allowed] = 0.0
    if drop.max() > 0.0:
        trans[:, 1] -= drop
        positions = positions - np.stack([np.zeros(len(drop)), drop, np.zeros(len(drop))],
                                         axis=1)[:, None, :]
    for side, (ankle_goal, foot_goal, _drop) in goals.items():
        hip, knee, ankle, _toe = LEGS[side]
        residual = 0.0
        moved = np.linalg.norm(ankle_goal - positions[:, ankle], axis=1) > 1e-5
        turned = np.abs(np.einsum("fii->f", np.swapaxes(foot_goal, 1, 2) @ rotations[:, ankle])
                        - 3.0) > 1e-9
        for frame in np.flatnonzero(moved | turned):
            residual = max(residual, _reach(pose, rotations, positions, skin, frame,
                                            hip, knee, ankle, ankle_goal[frame], foot_goal[frame]))
        report[side]["max_unreached_m"] = round(residual, 4)
    report["max_root_drop_m"] = round(float(drop.max()), 4)
    return pose.astype(np.float32), trans.astype(np.float32), report


#: A still foot (slower than this) of an upright body (pelvis over
#: ``UPRIGHT_PELVIS_M``) standing over ``HOVER_M`` above the floor for
#: ``HOVER_MIN_S`` is set down on it.
HOVER_MPS = 0.1
#: ... and turning slower than this (rad/s). = 0.35
HOVER_M = 0.005
HOVER_MIN_S = 0.3
#: ... and turning slower than this (rad/s): a foot turning in the air is a
#: pivot step (setting it down made the heel swivel on the floor, 4 cm).
HOVER_TURN_RAD_S = 0.35
UPRIGHT_PELVIS_M = 0.75
HOVER_REST_M = 0.002


def set_down_hovering_feet(pose, trans, skin: SMPLSkin, fps: float, *, floor: float = 0.0
                           ) -> tuple[np.ndarray, dict]:
    """``(pose, report)``: a standing foot left in the air (the cross walk
    ended with its right sole 2 cm up) is lowered onto the floor by its own
    leg (two-bone IK, the foot keeping its orientation), eased in and out
    over ``DIP_EASE_S``; the body is not moved."""
    pose = np.array(pose, dtype=np.float64, copy=True).reshape(len(trans), -1, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    if len(trans) < 3:
        return pose.astype(np.float32), {}
    dominant = skin.skin_indices[np.arange(len(skin.skin_indices)), skin.skin_weights.argmax(axis=1)]
    rotations, positions = _world_joints(pose, trans, skin)
    upright = positions[:, 0, 1] > floor + UPRIGHT_PELVIS_M
    ease = max(1, int(round(DIP_EASE_S * fps)))
    report = {}
    for side, (hip, knee, ankle, toe) in LEGS.items():
        ids = np.flatnonzero(np.isin(dominant, (ankle, toe)))
        foot = posed_vertices(pose, trans, skin, ids)
        low = foot[:, :, 1].min(axis=1)
        speed = np.linalg.norm(np.gradient(foot[:, :, [0, 2]].mean(axis=1), axis=0), axis=1) * fps
        forward = rotations[:, ankle] @ np.array([0.0, 0.0, 1.0])
        turning = np.abs(np.gradient(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))) * fps
        # A foot turning in the air is a pivot step, not a foot left standing up.
        flags = (upright & (speed < HOVER_MPS) & (turning < HOVER_TURN_RAD_S)
                 & (low > floor + HOVER_M))
        drop = np.zeros(len(low))
        for start, end in _runs(flags, max(2, int(round(HOVER_MIN_S * fps)))):
            depth = low[start:end] - floor - HOVER_REST_M
            for offset in range(-ease, end - start + ease):
                frame = start + offset
                if not 0 <= frame < len(low):
                    continue
                inside = depth[min(max(offset, 0), len(depth) - 1)]
                weight = (1.0 if 0 <= offset < end - start else
                          1.0 - _smooth_fade((offset + ease + 1) / (ease + 1)) if offset < 0 else
                          _smooth_fade((offset - (end - start) + 1) / (ease + 1)))
                drop[frame] = max(drop[frame], weight * inside)
        residual = 0.0
        for frame in np.flatnonzero(drop > 1e-5):
            goal = positions[frame, ankle] - np.array([0.0, drop[frame], 0.0])
            residual = max(residual, _reach(pose, rotations, positions, skin, frame,
                                            hip, knee, ankle, goal))
        report[side] = {"max_drop_m": round(float(drop.max()), 4),
                        "frames": int((drop > 1e-5).sum()), "max_unreached_m": round(residual, 4)}
    return pose.astype(np.float32), report


#: A foot this far under the floor is raised (eased as ``DIP_EASE_S``).
DIP_TOLERANCE_M = 0.002


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
        raise_m[:, leg] = _ease_hold(dip, ease)
    report = {"max_dip_m": round(float(raise_m.max()), 4),
              "raised_frames": int((raise_m > 1e-6).any(axis=1).sum())}
    if report["raised_frames"]:
        pose = raise_seated_legs(pose, trans, skin, raise_m)
    return pose.astype(np.float32), report


#: An upright body whose lowest foot is over this above the floor floats.
FLOAT_M = 0.003


def ground_upright_floats(pose, trans, skin: SMPLSkin, fps: float, *, floor: float = 0.0
                          ) -> tuple[np.ndarray, dict]:
    """``(trans, report)``: an upright body (pelvis over ``UPRIGHT_PELVIS_M``)
    whose lowest foot is off the floor (Kimodo's walk to an armchair floated
    3-6 cm through its first steps and turn) is lowered onto it, the drop
    held around each float and eased in and out over ``DIP_EASE_S``; never
    raised (a dipping toe is :func:`lift_dipping_feet`'s)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(len(trans), -1, 3)
    trans = np.array(trans, dtype=np.float64, copy=True).reshape(-1, 3)
    if len(trans) < 3:
        return trans.astype(np.float32), {}
    dominant = skin.skin_indices[np.arange(len(skin.skin_indices)), skin.skin_weights.argmax(axis=1)]
    feet = np.flatnonzero(np.isin(dominant, (7, 8, 10, 11)))
    low = posed_vertices(pose, trans, skin, feet)[:, :, 1].min(axis=1)
    _rotations, positions = _world_joints(pose, trans, skin)
    upright = positions[:, 0, 1] > floor + UPRIGHT_PELVIS_M
    lift = np.where(upright & (low - floor > FLOAT_M), low - floor, 0.0)
    ease = max(1, int(round(DIP_EASE_S * fps)))
    drop = _ease_hold(lift, ease)
    # Eased drops must not push a non-floating frame's feet through the floor.
    drop = np.minimum(drop, np.maximum(low - floor, 0.0))
    trans[:, 1] -= drop
    return trans.astype(np.float32), {"max_drop_m": round(float(drop.max()), 4),
                                      "frames": int((drop > 1e-5).sum())}


#: A bone turning faster than this between frames (deg per 50 ms) is
#: smoothed: the fall's left wrist flicked 74 deg in one 50 ms frame.
MAX_TURN_DEG_PER_50MS = 35.0


def limit_bone_turn_rate(pose, fps: float, *, max_deg_per_50ms: float = MAX_TURN_DEG_PER_50MS,
                         bones: Sequence[int] | None = None, max_rounds: int = 20
                         ) -> tuple[np.ndarray, dict]:
    """``(pose, report)``: each bone (not the root) whose rotation steps over
    ``max_deg_per_50ms`` between frames is smoothed around the step (a
    weighted quaternion average of its neighbours, sign aligned) until it
    no longer does; other frames are untouched."""
    pose = np.array(pose, dtype=np.float64, copy=True)
    frames = pose.shape[0]
    # Per frame step allowed (a 20 fps frame is 50 ms).
    limit = max_deg_per_50ms * (20.0 / fps) if fps > 0 else max_deg_per_50ms
    report = {}
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    for bone in (bones if bones is not None else range(1, pose.shape[1])):
        before = None
        for _round in range(max_rounds):
            turns = Rotation.from_rotvec(pose[:, bone])
            step = np.degrees((turns[:-1].inv() * turns[1:]).magnitude())
            if before is None:
                before = float(step.max()) if len(step) else 0.0
            bad = np.flatnonzero(step > limit)
            if not len(bad):
                break
            quats = turns.as_quat()
            for index in range(1, frames):  # one hemisphere
                if np.dot(quats[index], quats[index - 1]) < 0:
                    quats[index] = -quats[index]
            smooth = quats.copy()
            region = sorted({f for i in bad for f in range(i - 1, i + 3) if 0 < f < frames - 1})
            for frame in region:
                lo, hi = max(0, frame - 2), min(frames, frame + 3)
                w = weights[2 - (frame - lo):2 + (hi - frame)]
                mean = (quats[lo:hi] * w[:, None]).sum(axis=0)
                smooth[frame] = mean / np.linalg.norm(mean)
            pose[:, bone] = Rotation.from_quat(smooth).as_rotvec()
        turns = Rotation.from_rotvec(pose[:, bone])
        after = float(np.degrees((turns[:-1].inv() * turns[1:]).magnitude()).max()) if frames > 1 else 0.0
        if before is not None and before > limit:
            report[bone] = {"before_deg": round(before, 1), "after_deg": round(after, 1)}
    return pose.astype(np.float32), report


#: A first frame whose root turns more than this to the second (and more
#: than twice the next frame's turn) is Kimodo's start glitch.
FIRST_FRAME_JUMP_DEG = 12.0


def settle_first_frame(pose, trans) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(pose, trans, report)``: Kimodo's route walks start with the root
    turned 17-34 deg off the second frame (the body snapped 23 deg in the
    first 50 ms on the cross walk). Such a first frame takes the second
    frame's pose (root height and place kept)."""
    pose = np.array(pose, dtype=np.float64, copy=True).reshape(len(trans), -1, 3)
    trans = np.array(trans, dtype=np.float64, copy=True).reshape(-1, 3)
    if len(trans) < 3:
        return pose.astype(np.float32), trans.astype(np.float32), {"settled": False}
    root = Rotation.from_rotvec(pose[:3, 0])
    first = float(np.degrees((root[0].inv() * root[1]).magnitude()))
    second = float(np.degrees((root[1].inv() * root[2]).magnitude()))
    settled = first > FIRST_FRAME_JUMP_DEG and first > 2.0 * second
    if settled:
        pose[0] = pose[1]
    return pose.astype(np.float32), trans.astype(np.float32), {
        "settled": settled, "first_turn_deg": round(first, 1), "next_turn_deg": round(second, 1)}
