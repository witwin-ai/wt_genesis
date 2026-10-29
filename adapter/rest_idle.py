"""A person at rest keeps living until the clip ends: breathing, a look round.

A Kimodo clip ends where its prompt does (a sit, a stop); a scene that runs
on (the cat still jumping onto the bed 2 s later) held the last pose, and
the user saw a person sitting as a perfect statue for 3.4 s. The rest idle
continues that last pose with small upper-body motion only: breathing in
the spine and chest, a slow look round with the neck and head, the
shoulders following the breath. The pelvis, the legs and the feet do not
move (a seated person keeps its seat, a standing one its footing), so
nothing can slide; it eases in from the held pose over ``EASE_S``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

#: SMPL joints the idle moves: spine1, spine2, chest (spine3), neck, head,
#: collars.
SPINE, CHEST, NECK, HEAD, COLLARS = (3, 6), 9, 12, 15, (13, 14)
BREATH_HZ = 0.25
BREATH_DEG = 1.6
LOOK_DEG = 18.0
LOOK_TILT_DEG = 4.0
EASE_S = 0.6


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def rest_idle_poses(last_pose, seconds: float, fps: float, *, seed: int = 0) -> np.ndarray:
    """(frames, J, 3) rotation vectors continuing ``last_pose`` (J, 3) at rest
    for ``seconds`` (the first frame is one step after the last pose)."""
    last = np.asarray(last_pose, dtype=np.float64).reshape(-1, 3)
    frames = max(0, int(math.ceil(seconds * fps - 1e-9)))
    if not frames:
        return np.zeros((0, *last.shape))
    rng = np.random.default_rng(seed)
    # A look one way, then the other (or back), at a person's pace.
    look_to = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.0) * math.radians(LOOK_DEG)
    look_back = -look_to * rng.uniform(0.3, 0.8)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    out = np.repeat(last[None], frames, axis=0)
    for index in range(frames):
        t = (index + 1) / fps
        ease = _ease(t / EASE_S)
        breath = math.sin(2.0 * math.pi * BREATH_HZ * t + phase) * math.radians(BREATH_DEG) * ease
        # The look turns over ~1.5 s, holds, and comes part way back.
        span = max(seconds, 1e-6)
        first = _ease((t - 0.3) / 1.5)
        second = _ease((t - 0.3 - 0.55 * span) / 1.5) if span > 3.0 else 0.0
        yaw = look_to * first + (look_back - look_to) * second
        tilt = math.radians(LOOK_TILT_DEG) * math.sin(math.pi * min(t / span, 1.0)) * ease
        extra = {
            SPINE[0]: [-0.5 * breath, 0.15 * yaw, 0.0],
            SPINE[1]: [-0.5 * breath, 0.15 * yaw, 0.0],
            CHEST: [-breath, 0.15 * yaw, 0.0],
            NECK: [0.4 * tilt, 0.25 * yaw, 0.0],
            HEAD: [0.6 * tilt, 0.30 * yaw, 0.0],
            COLLARS[0]: [0.0, 0.0, 0.5 * breath],
            COLLARS[1]: [0.0, 0.0, -0.5 * breath],
        }
        for joint, rotvec in extra.items():
            if joint < len(last):
                out[index, joint] = (Rotation.from_rotvec(last[joint])
                                     * Rotation.from_rotvec(rotvec)).as_rotvec()
    return out
