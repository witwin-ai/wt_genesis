"""Kimodo clips are grounded once: SMPL soles on the floor, by one constant.

Kimodo's SOMA hip position is used as the SMPL root translation, but the
SMPL pelvis sits 0.22 m below its root and its legs differ: in the R47
person trial every Kimodo clip (walk, sit, in place) had its SMPL ankles
0.13-0.14 m below the floor from the first frame. The validator's floor
lift then raised walks in steps by up to 0.32 m (the hip rising mid-walk).
"""
from __future__ import annotations

import numpy as np
import pytest


def _toy_skeleton():
    from wt_genesis.adapter.kimodo_motion import SMPLRestSkeleton

    # pelvis -> knee -> ankle; the sole is 0.02 m under the ankle joint.
    return SMPLRestSkeleton(
        joints=np.array([[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, -1.0, 0.0]]),
        parents=[-1, 0, 1], foot_joints=(2,), sole_below_feet_m=0.02)


def _motion(trans_y, knee_bend=None):
    frames = len(trans_y)
    pose = np.zeros((1, frames, 24, 3), dtype=np.float32)
    if knee_bend is not None:
        pose[0, :, 1, 0] = knee_bend
    trans = np.zeros((1, frames, 3), dtype=np.float32)
    trans[0, :, 1] = trans_y
    trans[0, :, 0] = np.linspace(0.0, 1.0, frames)
    return {"smpl_pose": pose, "smpl_betas": np.zeros((1, frames, 10), np.float32),
            "smpl_trans": trans}


def test_a_sunk_clip_is_raised_so_the_soles_touch_the_floor():
    from wt_genesis.adapter.kimodo_motion import ground_smpl_motion

    grounded, lift = ground_smpl_motion(_motion([0.87] * 20), _toy_skeleton())

    # ankle at 0.87 - 1.0 = -0.13; the sole 0.02 lower must end at 0.
    assert lift == pytest.approx(0.15, abs=1e-6)
    assert grounded["smpl_trans"][0, :, 1] == pytest.approx(np.full(20, 1.02), abs=1e-6)
    assert grounded["smpl_trans"][0, :, 0] == pytest.approx(np.linspace(0.0, 1.0, 20))


def test_one_offset_for_the_whole_clip_keeps_a_sit_a_sit():
    from wt_genesis.adapter.kimodo_motion import ground_smpl_motion

    # Standing for 10 frames, then the hips drop 0.4 m (sitting) with the
    # knee bent so the ankle stays on the floor.
    hips = [0.87] * 10 + [0.47] * 10
    bend = [0.0] * 10 + [float(np.arccos(0.2))] * 10  # 0.5 * (1 - cos b) = 0.4
    grounded, lift = ground_smpl_motion(_motion(hips, bend), _toy_skeleton())
    after = grounded["smpl_trans"][0, :, 1]

    assert after[0] - after[-1] == pytest.approx(0.4, abs=1e-6)
    assert lift == pytest.approx(0.15, abs=1e-6)


def test_a_few_low_frames_do_not_lift_the_whole_clip():
    from wt_genesis.adapter.kimodo_motion import ground_smpl_motion

    hips = [0.87] * 40
    hips[7] = 0.60  # one glitch frame: an ankle 0.27 m lower
    _grounded, lift = ground_smpl_motion(_motion(hips), _toy_skeleton())

    assert lift == pytest.approx(0.15, abs=0.01)


def test_the_backend_grounds_kimodo_clips(tmp_path, monkeypatch):
    import wt_genesis.adapter.kimodo_backend as backend
    from tests.test_kimodo_text_cache import _fake_runtime

    backend, _commands = _fake_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "smpl_rest_skeleton", lambda: _toy_skeleton())

    motion = backend.generate_smpl_motion(["a person waves"], motion_length=2.0, seed=3)

    # The fake clip's root is at y 0 (ankle at -1.0): lifted by 1.02.
    assert motion["smpl_trans"][0, :, 1] == pytest.approx([1.02, 1.02], abs=1e-6)
    assert motion["grounding_lift_m"] == pytest.approx(1.02, abs=1e-6)


def test_the_real_smpl_rest_skeleton_has_feet_and_soles():
    import wt_genesis.adapter.kimodo_backend as backend

    skeleton = backend.smpl_rest_skeleton()
    if skeleton is None:
        pytest.skip("licensed SMPL model files are not available")
    assert skeleton.joints.shape == (24, 3)
    assert set(skeleton.foot_joints) == {7, 8, 10, 11}
    assert 0.0 < skeleton.sole_below_feet_m < 0.06
    pelvis_to_feet = skeleton.joints[0, 1] - skeleton.joints[list(skeleton.foot_joints), 1].min()
    assert 0.8 < pelvis_to_feet < 1.0
