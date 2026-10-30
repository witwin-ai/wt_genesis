"""A person at rest keeps living to the clip end (a seated statue for 3.4 s was rejected)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def test_the_subtle_rest_idle_moves_the_upper_body_only_smoothly_from_the_held_pose():
    from wt_genesis.adapter.rest_idle import rest_idle_poses

    last = np.zeros((24, 3))
    last[1, 0] = last[2, 0] = -1.5  # seated: hips flexed, knees bent
    last[4, 0] = last[5, 0] = 1.5
    poses = rest_idle_poses(last, 3.4, 20.0, seed=3, style="subtle")
    assert poses.shape == (68, 24, 3)
    # Pelvis, legs and feet stay as held: no seat or floor contact changes.
    for joint in (0, 1, 2, 4, 5, 7, 8, 10, 11):
        assert np.allclose(poses[:, joint], last[joint])
    # It eases in: the first frame is the held pose within a fraction of a degree.
    first = Rotation.from_rotvec(poses[0, 15]) * Rotation.from_rotvec(last[15]).inv()
    assert np.degrees(first.magnitude()) < 0.5
    # And it is alive: the head turns visibly, never faster than a calm look.
    head = Rotation.from_rotvec(poses[:, 15])
    turned = np.degrees((head[0].inv() * head).magnitude()).max()
    step = np.degrees((head[:-1].inv() * head[1:]).magnitude()).max()
    assert turned > 3.0 and step < 3.0


def _seated():
    last = np.zeros((24, 3))
    last[1, 0] = last[2, 0] = -1.5  # seated: hips flexed, knees bent
    last[4, 0] = last[5, 0] = 1.5
    return last


def test_the_lively_rest_idle_looks_round_visibly_and_never_holds_a_statue():
    """Studio's render gate flags any 1 s with no bone over 0.3 deg per 30 fps frame."""
    from wt_genesis.adapter.rest_idle import rest_idle_poses

    last = _seated()
    for seconds, fps, seed in ((3.45, 20.0, 0), (3.45, 30.0, 1), (8.0, 20.0, 2)):
        poses = rest_idle_poses(last, seconds, fps, seed=seed)
        assert poses.shape == (int(np.ceil(seconds * fps - 1e-9)), 24, 3)
        # Pelvis, legs and feet stay as held; so do the arms (no new contact).
        for joint in (0, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23):
            assert np.allclose(poses[:, joint], last[joint])
        seq = np.concatenate([last[None], poses])
        change = np.zeros(len(seq) - 1)
        for joint in range(24):
            r = Rotation.from_rotvec(seq[:, joint])
            change = np.maximum(change, np.degrees((r[1:] * r[:-1].inv()).magnitude()))
        # Never a turn faster than a person's quick look (< 80 deg/s on any bone) ...
        assert change.max() * fps < 80.0
        # ... and never still for 1 s (0.3 deg per 1/30 s = 9 deg/s).
        still, run = change * fps < 9.0, 0
        longest = 0
        for flag in still:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        assert longest / fps < 0.6
        # The look round is a real one: the head turns > 30 deg against the
        # pelvis (head, neck and upper spine together), both ways.
        chain = Rotation.from_rotvec(seq[:, 3])
        for joint in (6, 9, 12, 15):
            chain = chain * Rotation.from_rotvec(seq[:, joint])
        forward = chain.apply([0.0, 0.0, 1.0])
        yaw = np.degrees(np.arctan2(forward[:, 0], forward[:, 2]))
        assert yaw.max() > 30.0 and yaw.min() < -25.0
