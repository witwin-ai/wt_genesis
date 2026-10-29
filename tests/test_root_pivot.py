"""The scene body turns about the SMPL pelvis joint, as SMPL does.

SMPL rotates the root about the pelvis joint J0 (0.22 m under the root
translation); the scene body object turns about its own origin, which the
root motion tracks put at the SMPL translation. Upright the two agree (J0
is nearly vertical); a body lying down hovered ~0.2 m over the floor and a
person lying on a bed 0.2-0.3 m over the mattress (R47 pose review, found
by the person + cat session).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _skin_with_pelvis_offset():
    from wt_humanoid.adapter.smpl_skinned import SMPLSkinAsset

    return SMPLSkinAsset(
        vertices=np.array([[0.0, -0.3, 0.0], [0.0, 0.5, 0.0], [0.1, -0.3, 0.0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        skin_indices=np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64),
        skin_weights=np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=np.float32),
        joint_positions=np.array([[0.0, -0.2, 0.0], [0.0, 0.3, 0.0]], dtype=np.float32),
        parent_indices=[-1, 0],
        inverse_bind_matrices=np.array([np.eye(4), np.eye(4)], dtype=np.float32),
    )


def _lying_backend(prompts, **_kwargs):
    # Frame 0 upright; frame 1 turned 90 deg about x (lying on the back).
    pose = np.zeros((1, 2, 24, 3), dtype=np.float32)
    pose[0, 1, 0] = Rotation.from_euler("x", 90, degrees=True).as_rotvec()
    trans = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    return {"smpl_pose": pose, "smpl_betas": np.zeros((1, 2, 10), np.float32), "smpl_trans": trans}


def test_a_lying_body_keeps_its_pelvis_where_smpl_puts_it(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset",
                        lambda *args, **kwargs: _skin_with_pelvis_offset())
    scene = Scene(scene_id="main")
    result = create_humanoid_animation(
        scene=scene, project_dir=tmp_path, prompt="a person lies down", motion_length=0.1,
        device=-1, position=[2.0, 0.0, 3.0], playback_mode="baked_transforms",
        backend=_lying_backend)

    track = scene.timeline_manager.clip.get_track(result.body_object_id, "Transform", "position")
    # The 90 deg turn in one frame gets sub-frame keys; frame 1 is the last key.
    upright, lying = np.asarray(track.keyframes[0].value), np.asarray(track.keyframes[-1].value)
    # SMPL pelvis = trans + J0 = (2, 0.8, 3) in both frames; the body object
    # turns about its origin, so it must sit at pelvis - R J0.
    assert upright == pytest.approx([2.0, 1.0, 3.0], abs=1e-6)
    rotated = Rotation.from_euler("x", 90, degrees=True).apply([0.0, -0.2, 0.0])
    assert lying == pytest.approx(np.array([2.0, 0.8, 3.0]) - rotated, abs=1e-5)


def test_the_saved_body_stands_where_its_root_track_starts(tmp_path, monkeypatch):
    """Studio shows the static transform until the timeline first applies:
    left at the placement's y = 0 the pelvis was 1.17 m into the floor on
    frame 0 (the user saw a sunk person)."""
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset",
                        lambda *args, **kwargs: _skin_with_pelvis_offset())
    scene = Scene(scene_id="main")
    result = create_humanoid_animation(
        scene=scene, project_dir=tmp_path, prompt="a person lies down", motion_length=0.1,
        device=-1, position=[2.0, 0.0, 3.0], playback_mode="baked_transforms",
        backend=_lying_backend)

    track = scene.timeline_manager.clip.get_track(result.body_object_id, "Transform", "position")
    static = scene.get_object(result.body_object_id).get_component("Transform").position
    assert np.asarray(static, dtype=float) == pytest.approx(np.asarray(track.keyframes[0].value), abs=1e-6)
