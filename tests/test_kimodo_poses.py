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


def test_a_fall_is_left_to_kimodo_after_the_walk():
    from wt_genesis.adapter.kimodo_path import plan_root_path
    from wt_genesis.adapter.kimodo_poses import EndPose

    fall = EndPose(kind="fall", support_height=0.0, region_joints=None)
    plan = plan_root_path([(0.0, 0.0), (2.0, 0.0)], speed_mps=1.0, end_hold_s=3.0,
                          end_pose=fall)
    constraints = plan.kimodo_constraints()
    pinned = {index for item in constraints for index in item["frame_indices"]}
    arrival = int(round(plan.arrival_s * plan.fps))

    assert all(item["type"] == "root2d" for item in constraints)
    assert set(range(arrival + 1)) <= pinned
    assert not pinned & set(range(arrival + 1, plan.frame_count))


def _seat_only_skin():
    from wt_genesis.adapter.kimodo_poses import SMPLSkin

    skin = _tiny_skin()
    return SMPLSkin(vertices=skin.vertices[:1], joints=skin.joints, parents=skin.parents,
                    skin_indices=skin.skin_indices[:1], skin_weights=skin.skin_weights[:1])


def test_a_path_animation_rests_its_end_pose_on_the_support(tmp_path, monkeypatch):
    from tests.test_kimodo_path import ROUTE, _route_following_generator, _scene_with_tiny_skin
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation
    from wt_genesis.adapter.kimodo_poses import EndPose, upright_pelvis

    scene = _scene_with_tiny_skin(monkeypatch)
    pose = EndPose(kind="sit", hips=(3.0, 0.55, 3.9), rotvec=tuple(upright_pelvis(math.pi)),
                   support_height=0.45, region_joints=(0,))
    result = create_humanoid_path_animation(
        scene=scene, project_dir=tmp_path, prompt="a person walks over and sits down",
        waypoints=ROUTE, sit_at=(3.0, 3.9), end_heading=math.pi, end_hold_s=2.0,
        end_pose=pose, skin=_seat_only_skin(), generator=_route_following_generator([]))

    grounding = result.end_pose_grounding
    # The fake clip keeps its root at 0.92: the buttock vertex at 0.82 is
    # 0.37 m above the 0.45 m seat, and is lowered onto it.
    assert grounding["rest_gap_before_m"] == pytest.approx(0.37, abs=1e-3)
    track = scene.timeline_manager.clip.get_track(
        result.animation.body_object_id, "Transform", "position")
    assert track.keyframes[-1].value[1] == pytest.approx(0.55, abs=1e-3)
    assert track.keyframes[0].value[1] == pytest.approx(0.92, abs=1e-3)
    assert result.to_dict()["end_pose_grounding"] == grounding


def test_the_real_smpl_skin_stands_on_its_soles():
    import wt_genesis.adapter.kimodo_backend as backend
    from wt_genesis.adapter.kimodo_poses import BACK_REGION_JOINTS, SEAT_REGION_JOINTS, posed_vertices

    skin = backend.smpl_skin()
    if skin is None:
        pytest.skip("licensed SMPL model files are not available")
    rest = posed_vertices(np.zeros((1, 24, 3)), np.zeros((1, 3)), skin)[0]
    assert rest.shape == skin.vertices.shape
    assert np.allclose(rest, skin.vertices, atol=1e-6)  # rest pose is the bind pose
    assert len(skin.region(SEAT_REGION_JOINTS)) > 200 and len(skin.region(BACK_REGION_JOINTS)) > 200
    # The buttocks (pelvis-bound skin) sit about 0.1 m under the pelvis joint;
    # the seat region also holds the thighs, which rest on a seat when sitting.
    buttocks = skin.vertices[skin.region((0,)), 1].min()
    assert 0.05 < skin.joints[0, 1] - buttocks < 0.2


def test_the_walk_before_the_end_pose_stands_on_the_floor():
    # bench_7105 armchair: the clip-wide foot grounding was set by the seated
    # frames (feet lower after the retarget), so the walk floated 6-7 cm.
    from wt_genesis.adapter.kimodo_poses import ground_end_pose

    hips = [1.07] * 20 + list(np.linspace(1.07, 0.5, 20)) + [0.5] * 20  # foot at +0.07 while walking
    bend = np.arccos(np.clip((np.asarray(hips) - 0.5) / 0.5, -1.0, 1.0))
    bend[:20] = 0.0
    grounded, report = ground_end_pose(
        _clip(hips, bend), _tiny_skin(), support_height=0.45, region_joints=(0,),
        transition=(1.0, 2.0), rest=(2.0, 2.95), fps=20.0, foot_joints=(1,), smooth_s=0.0)
    after = grounded["smpl_trans"][0, :, 1]

    assert after[:20] == pytest.approx(np.full(20, 1.0), abs=1e-6)   # foot on the floor
    assert report["walk_correction_m"] == pytest.approx(-0.07, abs=1e-6)
    assert after[-1] == pytest.approx(0.55)                           # still seated on 0.45


def test_path_error_covers_the_walk_not_the_free_end_of_a_fall(tmp_path, monkeypatch):
    # Lying down and falling leave the route on purpose after arriving; the
    # bench refuses walks that stray > 0.15 m, so the free end must not count.
    from tests.test_kimodo_path import ROUTE, _route_following_generator, _scene_with_tiny_skin
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation
    from wt_genesis.adapter.kimodo_poses import EndPose

    follow = _route_following_generator([])

    def stumble(prompts, **kwargs):
        motion = follow(prompts, **kwargs)
        frames = motion["smpl_trans"].shape[1]
        motion["smpl_trans"][0, frames - 20:, 0] += np.linspace(0.0, 0.8, 20)
        return motion

    result = create_humanoid_path_animation(
        scene=_scene_with_tiny_skin(monkeypatch), project_dir=tmp_path, prompt="a person falls",
        waypoints=ROUTE, end_hold_s=2.0, end_pose=EndPose(kind="fall"), generator=stumble)

    assert result.path_following["timed_max_error_m"] < 0.05


def test_the_report_measures_the_settled_pose_after_grounding():
    # bench_7105 #1: Kimodo sat 3.9 cm high; lowering it onto the seat put the
    # feet through the floor, the floor guard lifted it back, and the buttocks
    # hovered. The report shows what the settled pose ends as, for a gate.
    from wt_genesis.adapter.kimodo_poses import ground_end_pose

    hips = [1.0] * 20 + list(np.linspace(1.0, 0.5, 20)) + [0.5] * 20
    bend = np.arccos(np.clip((np.asarray(hips) - 0.5) / 0.5, -1.0, 1.0))
    # The seat is 0.43: lowering the buttock (0.40) is fine, but a seat at
    # 0.35 would need the body 0.05 lower than the foot allows.
    _grounded, fine = ground_end_pose(_clip(hips, bend), _tiny_skin(), support_height=0.43,
                                      region_joints=(0,), transition=(1.0, 2.0),
                                      rest=(2.0, 2.95), fps=20.0, foot_joints=(1,))
    _grounded, hover = ground_end_pose(_clip(hips, bend), _tiny_skin(), support_height=0.35,
                                       region_joints=(0,), transition=(1.0, 2.0),
                                       rest=(2.0, 2.95), fps=20.0, foot_joints=(1,))

    assert fine["rest_support_gap_m"] == pytest.approx(0.0, abs=1e-6)
    assert fine["rest_feet_lowest_m"] == pytest.approx(0.03, abs=1e-6)
    assert hover["rest_support_gap_m"] == pytest.approx(0.05, abs=1e-6)   # held up by the floor
    assert hover["rest_feet_lowest_m"] == pytest.approx(0.0, abs=1e-6)


def test_no_frame_sinks_into_the_bed_on_the_way_to_lying_down():
    # bench_7107: getting onto the bed, the body went 16 cm into a 0.35 m
    # mattress; only the settled pose was put on the support.
    from wt_genesis.adapter.kimodo_poses import ground_end_pose

    # The buttock vertex passes over a 0.45 m support (x in [0.5, 2]) at 0.25 m
    # in the middle of the clip, then settles at 0.45 m; the knee bend keeps
    # the foot vertex on the floor.
    frames = 60
    hips = np.r_[np.full(20, 1.0), np.full(20, 0.35), np.full(20, 0.55)]
    clip = _clip(hips, np.arccos(np.clip((hips - 0.5) / 0.5, -1.0, 1.0)))
    clip["smpl_trans"][0, :, 0] = np.r_[np.zeros(20), np.full(40, 1.0)]
    clip["smpl_pose"][0, 20:40, 1, 0] = 0.0          # a straight leg in the bed's frames...
    clip["smpl_trans"][0, 20:40, 1] = 0.35           # ...the foot vertex would be at -0.65
    skin = _seat_only_skin()                          # judge the buttock alone

    grounded, report = ground_end_pose(
        clip, skin, support_height=0.45, region_joints=(0,), transition=(1.0, 2.0),
        rest=(2.0, 2.95), fps=20.0, support_rect=(0.5, -1.0, 2.0, 1.0), smooth_s=0.0)
    buttock = grounded["smpl_trans"][0, :, 1] - 0.1

    assert buttock[20:40].min() >= 0.45 - 0.02 - 1e-6   # never more than 2 cm into it
    assert buttock[:20] == pytest.approx(np.full(20, 0.9))  # off the support: untouched
    assert report["frames_lifted_onto_support"] > 0


def test_a_support_rect_in_the_scene_is_moved_into_the_clip_frame():
    # The clip is placed with its Kimodo origin on the route start, so clip
    # (x, z) = scene (x, z) - route start.
    from wt_genesis.adapter.kimodo_path import _clip_rect

    assert _clip_rect((3.0, 1.0, 5.0, 2.5), (1.0, 2.0)) == pytest.approx((2.0, -1.0, 4.0, 0.5))
    assert _clip_rect(None, (1.0, 2.0)) is None


def test_feet_on_the_floor_under_a_seat_edge_do_not_lift_the_body():
    # bench_7105: the guard looked at all skin over the armchair's footprint,
    # including feet on the floor under its front edge, and lifted the seated
    # person 30 cm. Only the support region (buttocks, back) is held up.
    from wt_genesis.adapter.kimodo_poses import ground_end_pose

    hips = [1.0] * 20 + list(np.linspace(1.0, 0.5, 20)) + [0.5] * 20
    bend = np.arccos(np.clip((np.asarray(hips) - 0.5) / 0.5, -1.0, 1.0))
    clip = _clip(hips, bend)
    # Everything stands inside the seat's footprint, the foot on the floor.
    grounded, report = ground_end_pose(
        clip, _tiny_skin(), support_height=0.40, region_joints=(0,), transition=(1.0, 2.0),
        rest=(2.0, 2.95), fps=20.0, support_rect=(-1.0, -1.0, 1.0, 1.0), smooth_s=0.0,
        foot_joints=(1,))

    assert report["rest_support_gap_m"] == pytest.approx(0.0, abs=1e-6)
    assert report["rest_feet_lowest_m"] == pytest.approx(0.0, abs=1e-6)
