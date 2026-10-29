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


def test_pivot_corrections_hold_the_heel_the_flat_foot_and_the_toe_and_fade_out():
    from wt_genesis.adapter.foot_lock import _turn_xz, pivot_corrections

    frames, fps = 30, 20.0
    # The foot slides forward (+x) and turns 0.3 rad while it stands.
    ankle = np.stack([np.linspace(0.0, 0.29, frames), np.zeros(frames)], axis=1)
    yaw = np.linspace(0.0, 0.3, frames)
    heel = ankle + np.stack([_turn_xz(angle, [0.0, -0.05]) for angle in yaw])
    toe = ankle + np.stack([_turn_xz(angle, [0.0, 0.15]) for angle in yaw])
    heel_down = np.zeros(frames, bool)
    toe_down = np.zeros(frames, bool)
    heel_down[0:10] = True
    toe_down[6:15] = True
    shift, turn = pivot_corrections(ankle, heel, toe, yaw, heel_down, toe_down, fps,
                                    release_s=0.25)

    def placed(points, frame):
        return ankle[frame] + shift[frame] + _turn_xz(turn[frame], points[frame] - ankle[frame])

    heel_held = [placed(heel, f) for f in range(0, 6)]
    assert np.allclose(heel_held, heel_held[0])
    # Held where the foot was on average over the contact, not at touchdown:
    # the correction is centred (the leg reaches it).
    assert np.allclose(shift[:15].mean(axis=0), 0.0, atol=1e-9)
    # Flat (both down): heel and toe both stay, the heading too.
    flat_heel = [placed(heel, f) for f in range(6, 10)]
    flat_toe = [placed(toe, f) for f in range(6, 10)]
    assert np.allclose(flat_heel, flat_heel[0], atol=1e-9)
    assert np.allclose(flat_toe, flat_toe[0], atol=1e-9)
    assert np.allclose(yaw[6:10] + turn[6:10], yaw[6] + turn[6])
    # Rolled onto the toe: it stays where it was, no jump.
    toe_held = [placed(toe, f) for f in range(9, 15)]
    assert np.allclose(toe_held, toe_held[0], atol=1e-9)
    # Lifted: the correction fades out within release_s.
    assert np.linalg.norm(shift[15]) < np.linalg.norm(shift[14])
    assert np.allclose(shift[15 + 5:], 0.0) and np.allclose(turn[15 + 5:], 0.0)


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
    locked, locked_trans, report = lock_planted_feet(pose, trans, skin, fps)
    # In reach: the root is not touched.
    assert np.allclose(locked_trans, trans) and report["max_root_drop_m"] == 0.0
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


def test_a_held_foot_out_of_reach_brings_the_root_down_instead_of_sliding():
    from wt_genesis.adapter.foot_lock import lock_planted_feet, sole_points
    from wt_genesis.adapter.kimodo_poses import posed_vertices

    skin = _skin()
    frames, fps = 20, 20.0
    pose = np.zeros((frames, 24, 3))
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
        pose[:, hip, 0], pose[:, knee, 0], pose[:, ankle, 0] = -0.1, 0.2, -0.1
    trans = np.zeros((frames, 3))
    trans[:, 1] = 1.1
    # Nearly straight legs; the body creeps 40 cm: the held feet are out of reach.
    trans[:, 2] = np.linspace(0.0, 0.4, frames)
    locked, locked_trans, report = lock_planted_feet(pose, trans, skin, fps)
    assert report["max_root_drop_m"] > 0.005
    assert np.allclose(locked_trans[:, [0, 2]], trans[:, [0, 2]])
    assert (locked_trans[:, 1] <= trans[:, 1] + 1e-6).all()
    for side, (heel_ids, _toe_ids) in sole_points(skin).items():
        after = posed_vertices(locked, locked_trans, skin, heel_ids)[:, :, [0, 2]].mean(axis=1)
        assert np.linalg.norm(after - after[0], axis=1).max() < 0.005


def test_an_upright_body_floating_over_its_feet_comes_down_and_is_never_raised():
    from wt_genesis.adapter.foot_lock import ground_upright_floats
    from wt_genesis.adapter.kimodo_poses import posed_vertices

    skin = _skin()
    frames, fps = 40, 20.0
    pose = np.zeros((frames, 24, 3))
    trans = np.zeros((frames, 3))
    lowest = posed_vertices(pose[:1], trans[:1], skin)[0, :, 1].min()
    trans[:, 1] = -lowest
    trans[15:25, 1] += 0.05  # floats 5 cm for 0.5 s
    grounded, report = ground_upright_floats(pose, trans, skin, fps)
    low = posed_vertices(pose, grounded, skin)[:, :, 1].min(axis=1)
    assert np.abs(low).max() < 0.004
    assert (grounded[:, 1] <= trans[:, 1] + 1e-6).all()
    assert report["max_drop_m"] == pytest.approx(0.05, abs=0.005)


def test_a_wrist_flick_is_smoothed_under_the_turn_rate_gate():
    from scipy.spatial.transform import Rotation
    from wt_genesis.adapter.foot_lock import limit_bone_turn_rate

    frames, fps = 30, 20.0
    pose = np.zeros((frames, 24, 3))
    pose[:, 20, 2] = np.where(np.arange(frames) < 15, 0.0, np.radians(74.0))  # one-frame flick
    pose[:, 4, 0] = np.linspace(0.0, 0.5, frames)  # a slow knee stays as it is
    smoothed, report = limit_bone_turn_rate(pose, fps)
    turns = Rotation.from_rotvec(smoothed[:, 20])
    assert np.degrees((turns[:-1].inv() * turns[1:]).magnitude()).max() <= 35.0 + 1e-6
    assert report[20]["before_deg"] == pytest.approx(74.0, abs=0.1)
    assert np.allclose(smoothed[:, 4], pose[:, 4], atol=1e-6)
    assert np.allclose(smoothed[:8, 20], 0.0, atol=1e-6)  # far from the flick: untouched


def test_a_first_frame_snapping_the_root_takes_the_second_frames_pose():
    from wt_genesis.adapter.foot_lock import settle_first_frame

    frames = 10
    pose = np.zeros((frames, 24, 3))
    pose[:, 0, 1] = np.r_[0.11, np.linspace(0.51, 0.6, frames - 1)]  # 23 deg, then 1 deg
    pose[:, 4, 0] = 0.2
    trans = np.zeros((frames, 3))
    settled, _trans, report = settle_first_frame(pose, trans)
    assert report["settled"] and report["first_turn_deg"] > 20
    assert np.allclose(settled[0], pose[1]) and np.allclose(settled[1:], pose[1:])
    # A clip that starts turning smoothly is left alone.
    smooth = pose.copy()
    smooth[:, 0, 1] = np.linspace(0.0, 0.9, frames)
    kept, _trans, report = settle_first_frame(smooth, trans)
    assert not report["settled"] and np.allclose(kept, smooth)
