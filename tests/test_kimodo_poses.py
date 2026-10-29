"""A Kimodo person that sits on a seat, lies down on a bed, or falls.

The first sits sank 0.12-0.17 m into sofas: Kimodo sits at its own seat
height, and only the root (x, z) and heading were constrained. Kimodo can
pin the pelvis itself (its ``Hips`` end effector: position and orientation)
on the last frames; what remains after the SOMA -> SMPL retarget is fixed by
grounding the posed skin on its support (seat, mattress or floor).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_an_upright_pelvis_faces_its_heading():
    from wt_genesis.adapter.kimodo_poses import upright_pelvis

    facing = Rotation.from_rotvec(upright_pelvis(math.pi / 2)).apply([0.0, 0.0, 1.0])
    assert facing == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_a_supine_pelvis_points_the_head_along_the_bed_and_the_face_up():
    from wt_genesis.adapter.kimodo_poses import supine_pelvis

    rotation = Rotation.from_rotvec(supine_pelvis((0.0, -1.0)))
    assert rotation.apply([0.0, 1.0, 0.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)  # head
    assert rotation.apply([0.0, 0.0, 1.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)  # face
    assert np.linalg.det(rotation.as_matrix()) == pytest.approx(1.0)


def test_an_end_pose_pins_the_hips_on_the_last_frames_instead_of_the_root_path():
    from wt_genesis.adapter.kimodo_path import plan_root_path
    from wt_genesis.adapter.kimodo_poses import EndPose, upright_pelvis

    pose = EndPose(kind="sit", hips=(2.0, 0.55, -0.6), rotvec=tuple(upright_pelvis(math.pi)),
                   hold_s=0.5)
    plan = plan_root_path([(0.0, 0.0), (2.0, 0.0)], speed_mps=1.0, end_hold_s=3.0,
                          sit_at=(2.0, -0.6), end_pose=pose)
    constraints = plan.kimodo_constraints()

    hips = [item for item in constraints if item["type"] == "end-effector"]
    assert len(hips) == 1 and hips[0]["joint_names"] == ["Hips"]
    frames = hips[0]["frame_indices"]
    assert frames[-1] == plan.frame_count - 1 and len(frames) == round(0.5 * plan.fps)
    assert np.asarray(hips[0]["local_joints_rot"]).shape == (len(frames), 30, 3)
    assert hips[0]["local_joints_rot"][0][0] == pytest.approx(list(pose.rotvec))
    assert hips[0]["local_joints_rot"][0][5] == pytest.approx([0.0, 0.0, 0.0])
    # Relative to the route start, like the root path; y is the pelvis height.
    assert hips[0]["root_positions"][-1] == pytest.approx([2.0, 0.55, -0.6])
    assert hips[0]["smooth_root_2d"][-1] == pytest.approx([2.0, -0.6])
    root2d = {index for item in constraints if item["type"] == "root2d"
              for index in item["frame_indices"]}
    assert not root2d & set(frames)


def _tiny_skin():
    from wt_genesis.adapter.kimodo_poses import SMPLSkin

    # Two joints (pelvis, knee) 0.5 m apart; a "buttock" vertex 0.1 m under
    # the pelvis bound to it, a "foot" vertex 0.5 m under the knee.
    return SMPLSkin(
        vertices=np.array([[0.0, -0.1, 0.0], [0.0, -1.0, 0.0]]),
        joints=np.array([[0.0, 0.0, 0.0], [0.0, -0.5, 0.0]]),
        parents=[-1, 0],
        skin_indices=np.array([[0, 0, 0, 0], [1, 0, 0, 0]]),
        skin_weights=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )


def test_posed_vertices_follow_their_bones():
    from wt_genesis.adapter.kimodo_poses import posed_vertices

    pose = np.zeros((2, 24, 3))
    pose[1, 1] = [math.pi / 2, 0.0, 0.0]  # knee turned 90 deg about x: foot goes to -z
    trans = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    vertices = posed_vertices(pose, trans, _tiny_skin())

    assert np.allclose(vertices[0], [[0.0, 0.9, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(vertices[1], [[1.0, 0.9, 0.0], [1.0, 0.5, -0.5]], atol=1e-9)


def _clip(hips_y, knee_bend=None):
    frames = len(hips_y)
    motion = {"smpl_pose": np.zeros((1, frames, 24, 3), dtype=np.float32),
              "smpl_betas": np.zeros((1, frames, 10), dtype=np.float32),
              "smpl_trans": np.zeros((1, frames, 3), dtype=np.float32)}
    motion["smpl_trans"][0, :, 1] = hips_y
    if knee_bend is not None:
        motion["smpl_pose"][0, :, 1, 0] = knee_bend
    return motion


def test_the_settled_pose_rests_on_its_support_and_the_walk_is_untouched():
    from wt_genesis.adapter.kimodo_poses import ground_end_pose

    # 20 fps: walk for 1 s with the "foot" on the floor (root 1.0), then the
    # pelvis drops; at rest the buttock vertex is at 0.40 but the seat is 0.45.
    hips = [1.0] * 20 + list(np.linspace(1.0, 0.5, 20)) + [0.5] * 20
    # The knee bends as the pelvis drops, so the foot stays on the floor.
    bend = np.arccos(np.clip((np.asarray(hips) - 0.5) / 0.5, -1.0, 1.0))
    grounded, report = ground_end_pose(
        _clip(hips, bend), _tiny_skin(), support_height=0.45, region_joints=(0,),
        transition=(1.0, 2.0), rest=(2.0, 2.95), fps=20.0)
    after = grounded["smpl_trans"][0, :, 1]

    assert after[:20] == pytest.approx(hips[:20])          # walk unchanged
    assert after[-1] == pytest.approx(0.55)                # buttock 0.45 on the seat
    assert after[30] == pytest.approx(hips[30] + 0.025, abs=1e-6)  # half-blended
    assert report["rest_gap_before_m"] == pytest.approx(-0.05)
    assert report["rest_correction_m"] == pytest.approx(0.05)


def test_no_frame_of_a_fall_goes_through_the_floor():
    from wt_genesis.adapter.kimodo_poses import ground_end_pose

    # Falling: the root drops so far that the foot vertex would go to -0.2 m.
    hips = [1.0] * 20 + list(np.linspace(1.0, 0.8, 20)) + [0.8] * 20
    grounded, report = ground_end_pose(
        _clip(hips), _tiny_skin(), support_height=0.0, region_joints=None,
        transition=(1.0, 2.0), rest=(2.0, 2.95), fps=20.0)
    lowest = grounded["smpl_trans"][0, :, 1] - 1.0  # foot vertex height

    assert lowest.min() >= -1e-6
    assert lowest[-1] == pytest.approx(0.0, abs=1e-6)       # at rest on the floor
    assert report["frames_lifted_off_floor"] == 0 or lowest.min() >= -1e-6
