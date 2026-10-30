"""A person at rest keeps living until the clip ends: a look round, a settle, breathing.

A Kimodo clip ends where its prompt does (a sit, a stop); a scene that runs
on (the cat still jumping onto the bed 2 s later) held the last pose, and
the user saw a person sitting as a perfect statue for 3.4 s. The rest idle
continues that last pose with upper-body motion only. The pelvis, the legs
and the feet do not move (a seated person keeps its seat, a standing one its
footing), so nothing can slide.

Two styles:

* ``"lively"`` (the default): a chain a reviewer sees in Studio. The person
  looks to one side (~40 deg: head, neck and upper spine together), holds,
  looks across to the other side (~32 deg), holds, then settles: a sigh (the
  shoulders rise and drop), a small forward lean and a nod down, the head
  coming back toward the front with a slight tilt. A longer rest keeps going
  with smaller looks and settles. Every beat moves some bone faster than
  0.3 deg per frame at 30 fps for well under a second of each second (the
  Studio render gate flags a pose unchanged for over 1 s as STILL).
* ``"subtle"``: the first version (breathing 1.6 deg and a slow 18 deg look
  spread over spine, neck and head, then a hold). It read as a statue in
  Studio; kept for callers that want it.
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
STYLES = ("lively", "subtle")

#: Lively style: how a look (yaw), a lean (forward flexion), a nod (head
#: pitch, + down) and a tilt (head roll) are shared down the chain.
LOOK_SHARE = {SPINE[0]: 0.08, SPINE[1]: 0.10, CHEST: 0.12, NECK: 0.28, HEAD: 0.42}
LEAN_SHARE = {SPINE[0]: 0.40, SPINE[1]: 0.35, CHEST: 0.25}
NOD_SHARE = {NECK: 0.4, HEAD: 0.6}
TILT_SHARE = {NECK: 0.4, HEAD: 0.6}
LIVELY_BREATH_DEG = 2.0
#: The opening chain, ``(time s, look, lean, nod, tilt, shrug)`` in degrees;
#: look + is toward ``side``. Between keys each channel follows a minimum-jerk
#: curve (it starts and ends at rest, the way a person turns their head).
OPENING = (
    (0.00, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.95, 40.0, 0.0, -3.0, 2.0, 0.0),     # look to the side
    (1.35, 42.0, 0.0, -1.0, 6.0, 1.0),     # hold: the head keeps tilting a little
    (2.35, -32.0, 1.5, 1.0, -3.0, 0.0),    # look across to the other side
    (2.70, -34.0, 2.0, 2.5, -6.0, 0.0),    # hold
    (3.10, -20.0, 4.0, 5.0, -4.0, 6.0),    # sigh: the shoulders rise ...
    (3.60, -6.0, 7.0, 9.0, 3.0, -1.5),     # ... and drop; lean forward, nod, tilt
)
#: After the opening: further beats ``(duration s, look, lean, nod, tilt,
#: shrug)``, scaled by a random 0.7-1.0 and mirrored per round.
LATER = (
    (0.45, -8.0, 6.0, 7.0, 5.0, -1.0),     # settle in the lean
    (0.90, 24.0, 3.0, 0.0, 2.0, 0.0),      # a smaller look to the first side
    (0.40, 26.0, 3.0, -2.0, 6.0, 0.0),     # hold
    (0.90, -14.0, 1.0, 2.0, -3.0, 0.0),    # back across
    (0.45, -4.0, 2.0, 5.0, -5.0, 5.0),     # a breath in
    (0.50, 0.0, 5.0, 8.0, 2.0, -1.0),      # and out, leaning
)


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def lively_channels(seconds: float, fps: float, *, seed: int = 0) -> np.ndarray:
    """``(frames, 5)`` look, lean, nod, tilt, shrug in degrees for a rest of
    ``seconds`` (frame ``i`` at ``(i + 1) / fps``)."""
    frames = max(0, int(math.ceil(seconds * fps - 1e-9)))
    rng = np.random.default_rng(seed)
    side = 1.0 if seed % 2 == 0 else -1.0
    keys = [(t, np.array(v, dtype=np.float64) * [side, 1, 1, side, 1])
            for t, *v in OPENING]
    end = frames / fps
    flip = side
    while keys[-1][0] < end:
        flip = -flip
        scale = rng.uniform(0.7, 1.0)
        for duration, *values in LATER:
            value = np.array(values, dtype=np.float64) * [flip * scale, scale, scale, flip, scale]
            keys.append((keys[-1][0] + duration, value))
    times = np.array([key[0] for key in keys])
    values = np.stack([key[1] for key in keys])
    t = (np.arange(frames) + 1) / fps
    out = np.zeros((frames, values.shape[1]))
    for index, now in enumerate(t):
        seg = min(int(np.searchsorted(times, now, side="right")) - 1, len(times) - 2)
        u = (now - times[seg]) / (times[seg + 1] - times[seg])
        out[index] = values[seg] + (values[seg + 1] - values[seg]) * _min_jerk(u)
    return out


def _lively(last, seconds, fps, seed):
    frames = max(0, int(math.ceil(seconds * fps - 1e-9)))
    channels = lively_channels(seconds, fps, seed=seed)
    phase = np.random.default_rng(seed + 1).uniform(0.0, 2.0 * math.pi)
    out = np.repeat(last[None], frames, axis=0)
    for index in range(frames):
        t = (index + 1) / fps
        look, lean, nod, tilt, shrug = np.radians(channels[index])
        breath = math.sin(2.0 * math.pi * BREATH_HZ * t + phase) * math.radians(LIVELY_BREATH_DEG) \
            * _ease(t / EASE_S)
        extra = {joint: np.zeros(3) for joint in (*SPINE, CHEST, NECK, HEAD, *COLLARS)}
        for joint, share in LOOK_SHARE.items():
            extra[joint][1] += share * look
        for joint, share in LEAN_SHARE.items():
            extra[joint][0] += share * lean
        for joint, share in NOD_SHARE.items():
            extra[joint][0] += share * nod
        for joint, share in TILT_SHARE.items():
            extra[joint][2] += share * tilt
        extra[SPINE[0]][0] -= 0.5 * breath
        extra[SPINE[1]][0] -= 0.5 * breath
        extra[CHEST][0] -= breath
        # A collar's +z raises the left shoulder, -z the right one.
        extra[COLLARS[0]][2] += shrug + 0.5 * breath
        extra[COLLARS[1]][2] -= shrug + 0.5 * breath
        for joint, rotvec in extra.items():
            if joint < len(last):
                out[index, joint] = (Rotation.from_rotvec(last[joint])
                                     * Rotation.from_rotvec(rotvec)).as_rotvec()
    return out


def _subtle(last, seconds, fps, seed):
    frames = max(0, int(math.ceil(seconds * fps - 1e-9)))
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


def rest_idle_poses(last_pose, seconds: float, fps: float, *, seed: int = 0,
                    style: str = "lively") -> np.ndarray:
    """(frames, J, 3) rotation vectors continuing ``last_pose`` (J, 3) at rest
    for ``seconds`` (the first frame is one step after the last pose).
    ``style``: ``"lively"`` (default) or ``"subtle"`` (module doc)."""
    if style not in STYLES:
        raise ValueError(f"rest idle style {style!r}: one of {STYLES}")
    last = np.asarray(last_pose, dtype=np.float64).reshape(-1, 3)
    frames = max(0, int(math.ceil(seconds * fps - 1e-9)))
    if not frames:
        return np.zeros((0, *last.shape))
    return (_lively if style == "lively" else _subtle)(last, seconds, fps, seed)
