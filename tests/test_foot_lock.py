"""A planted SMPL foot is held still (the SOMA -> SMPL retarget slid it 6-15 cm)."""
from __future__ import annotations

import numpy as np
import pytest


def _skin():
    from wt_genesis.adapter import kimodo_backend as backend

    skin = backend.smpl_skin()
    if skin is None:
        pytest.skip("licensed SMPL model files are not available")
    return skin


def test_pivot_corrections_hold_the_heel_then_the_toe_and_fade_out():
    from wt_genesis.adapter.foot_lock import pivot_corrections

    frames, fps = 30, 20.0
    heel = np.stack([np.linspace(0.0, 0.29, frames), np.zeros(frames)], axis=1)
    toe = heel + [0.2, 0.0]
    heel_down = np.zeros(frames, bool)
    toe_down = np.zeros(frames, bool)
    heel_down[0:10] = True
    toe_down[8:15] = True
    delta = pivot_corrections(heel, toe, heel_down, toe_down, fps, release_s=0.25)
    held_heel = heel[:10] + delta[:10]
    assert np.allclose(held_heel, heel[0])
    # Rolled onto the toe: it stays where it was at the roll, no jump.
    held_toe = toe[10:15] + delta[10:15]
    assert np.allclose(held_toe, toe[10] + delta[9])
    # Lifted: the correction fades out within release_s.
    assert np.linalg.norm(delta[15]) < np.linalg.norm(delta[14])
    assert np.allclose(delta[15 + 5:], 0.0)


def test_a_sliding_planted_foot_is_held_by_the_leg_and_the_root_is_untouched():
    from wt_genesis.adapter.foot_lock import lock_planted_feet, sole_points
    from wt_genesis.adapter.kimodo_poses import posed_vertices

    skin = _skin()
    frames, fps = 20, 20.0
    pose = np.zeros((frames, 24, 3))
    # Knees a little bent, as a walk stands (a straight leg cannot reach further).
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
        pose[:, hip, 0], pose[:, knee, 0], pose[:, ankle, 0] = -0.3, 0.6, -0.3
    # Standing, the body creeps 6 cm forward: both feet slide with it.
    trans = np.zeros((frames, 3))
    trans[:, 1] = 1.1
    trans[:, 2] = np.linspace(0.0, 0.06, frames)
    locked, report = lock_planted_feet(pose, trans, skin, fps)
    for side, (heel_ids, _toe_ids) in sole_points(skin).items():
        before = posed_vertices(pose, trans, skin, heel_ids)[:, :, [0, 2]].mean(axis=1)
        after = posed_vertices(locked, trans, skin, heel_ids)[:, :, [0, 2]].mean(axis=1)
        assert np.linalg.norm(before[-1] - before[0]) > 0.05
        assert np.linalg.norm(after - after[0], axis=1).max() < 0.005
        assert report[side]["max_unreached_m"] < 0.005
    # Only the legs move: the pelvis and the upper body keep their rotations.
    untouched = [joint for joint in range(24) if joint not in (1, 2, 4, 5, 7, 8)]
    assert np.allclose(locked[:, untouched], pose[:, untouched])


def test_a_toe_dipping_through_the_floor_is_raised_by_its_leg_not_the_body():
    from wt_genesis.adapter.foot_lock import lift_dipping_feet
    from wt_genesis.adapter.kimodo_poses import posed_vertices

    skin = _skin()
    frames, fps = 20, 20.0
    pose = np.zeros((frames, 24, 3))
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
        pose[:, hip, 0], pose[:, knee, 0], pose[:, ankle, 0] = -0.3, 0.6, -0.3
    trans = np.zeros((frames, 3))
    lowest = posed_vertices(pose[:1], trans[:1], skin)[0, :, 1].min()
    trans[:, 1] = -lowest  # standing on y = 0
    trans[8:12, 1] -= 0.012  # the body dips 1.2 cm for 0.2 s
    lifted, report = lift_dipping_feet(pose, trans, skin, fps)
    low = posed_vertices(lifted, trans, skin)[:, :, 1].min(axis=1)
    assert low.min() > -0.0025
    assert report["max_dip_m"] == pytest.approx(0.012, abs=0.002)
    # Eased: the legs change gradually, not in one frame.
    step = np.abs(np.diff(lifted[:, 4, 0]))
    assert step.max() < 0.5 * np.abs(lifted[:, 4, 0] - pose[:, 4, 0]).max() + 1e-9
