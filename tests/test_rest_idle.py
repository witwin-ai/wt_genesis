"""A person at rest keeps living to the clip end (a seated statue for 3.4 s was rejected)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def test_the_rest_idle_moves_the_upper_body_only_smoothly_from_the_held_pose():
    from wt_genesis.adapter.rest_idle import rest_idle_poses

    last = np.zeros((24, 3))
    last[1, 0] = last[2, 0] = -1.5  # seated: hips flexed, knees bent
    last[4, 0] = last[5, 0] = 1.5
    poses = rest_idle_poses(last, 3.4, 20.0, seed=3)
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
