"""Bone rotation keys stay smooth in Studio's per-component Euler lerp.

A sit-down bends the knees through X = 90 deg, the ZXY singularity: the
meeting sit samples' knee keys jumped 1.4-1.6 rad apart and the legs whipped
between keys in Studio (user: a ghost glide, a huge first stride).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _deep_sit(frames=40):
    poses = np.zeros((frames, 72), dtype=np.float32)
    bend = np.linspace(0.0, 1.9, frames)  # past 90 deg, as a sitter's knee
    for knee in (4, 5):
        poses[:, knee * 3] = bend
        poses[:, knee * 3 + 1] = 0.08 * np.sin(np.linspace(0, 3, frames))
        poses[:, knee * 3 + 2] = 0.03
    for hip in (1, 2):
        poses[:, hip * 3] = -bend
        poses[:, hip * 3 + 2] = 0.05
    return poses


def test_no_euler_component_steps_more_than_0_05_rad_on_a_deep_sit():
    from wt_genesis.adapter.scene_animation import MAX_EULER_STEP_RAD, bone_rotation_keys

    poses, fps = _deep_sit(), 20.0
    assert MAX_EULER_STEP_RAD <= 0.05
    for bone in (1, 2, 4, 5):
        keys = bone_rotation_keys(poses, bone, fps)
        values = np.asarray([value for _time, value in keys])
        times = np.asarray([time for time, _value in keys])
        assert np.abs(np.diff(values, axis=0)).max() <= 0.05 + 1e-9
        assert (np.diff(times) > 0.0001).all()
        # Every source frame keeps its key, with its true rotation.
        for frame in range(len(poses)):
            index = int(np.argmin(np.abs(times - frame / fps)))
            assert abs(times[index] - frame / fps) < 1e-9
            got = Rotation.from_euler("ZXY", values[index])
            want = Rotation.from_rotvec(poses[frame, bone * 3:bone * 3 + 3])
            assert (got.inv() * want).magnitude() < 1e-5


def test_keys_are_unwrapped_across_pi():
    from wt_genesis.adapter.scene_animation import bone_rotation_keys

    frames = 30
    poses = np.zeros((frames, 72), dtype=np.float32)
    # A forearm turning past 180 deg about its twist axis.
    poses[:, 18 * 3 + 1] = np.linspace(2.8, 3.6, frames)
    keys = bone_rotation_keys(poses, 18, 20.0)
    values = np.asarray([value for _time, value in keys])
    assert np.abs(np.diff(values, axis=0)).max() <= 0.05 + 1e-9


def test_a_walk_keeps_one_key_per_frame():
    from wt_genesis.adapter.scene_animation import bone_rotation_keys

    frames = 20
    poses = np.zeros((frames, 72), dtype=np.float32)
    poses[:, 4 * 3] = 0.3 + 0.01 * np.arange(frames)
    assert len(bone_rotation_keys(poses, 4, 20.0)) == frames


def test_a_falling_root_is_densified_and_every_sub_key_keeps_the_pelvis_on_smpls_path():
    from wt_genesis.adapter.scene_animation import root_motion_keys

    frames, fps = 20, 20.0
    pelvis = np.array([0.0, -0.22, 0.03])
    pitch = np.linspace(0.0, 1.7, frames)  # forward, past lying face down
    rotvecs = np.stack([pitch, 0.1 * np.ones(frames), 0.05 * np.ones(frames)], axis=1)
    translation = np.stack([np.zeros(frames), np.linspace(1.17, 0.3, frames),
                            np.linspace(0.0, 0.8, frames)], axis=1)
    positions = translation + pelvis - Rotation.from_rotvec(rotvecs).apply(pelvis)
    keys = root_motion_keys({"positions": positions.tolist(), "root_rotvecs": rotvecs},
                            pelvis, fps)
    values = np.asarray([value for _t, _p, value in keys])
    assert len(keys) > frames
    assert np.abs(np.diff(values, axis=0)).max() <= 0.05 + 1e-9
    for time, position, value in keys:
        # The scene pelvis (object + R J0) is SMPL's (translation + J0), lerped.
        share = time * fps
        low = min(int(np.floor(share + 1e-9)), frames - 2)
        expected = (translation[low] + (share - low) * (translation[low + 1] - translation[low])
                    + pelvis)
        got = np.asarray(position) + Rotation.from_euler("ZXY", value).apply(pelvis)
        assert np.allclose(got, expected, atol=1e-6)
