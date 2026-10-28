"""A Kimodo human that walks a planned 2D route through the scene.

Text alone does not control where a generated human goes: a "few slow small
steps" prompt walked 11.9 m in 9 s. Kimodo accepts dense root (x, z) and
heading constraints; these tests pin the scene -> Kimodo path contract, the
placement of the generated motion back on the planned route, and the timed
occupancy a later cat + human script reads.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest


ROUTE = [(1.0, 2.0), (3.0, 2.0), (3.0, 3.5)]


def test_constant_speed_plan_starts_and_ends_on_the_route():
    from wt_genesis.adapter.kimodo_path import plan_root_path

    plan = plan_root_path(ROUTE, speed_mps=1.0, start_hold_s=0.5, end_hold_s=1.0, ease_s=0.4)

    assert plan.origin == pytest.approx((1.0, 2.0))
    assert plan.length_m == pytest.approx(3.5)
    assert plan.duration_s == pytest.approx(0.5 + 3.5 + 0.4 + 1.0, abs=1.0 / plan.fps)
    assert plan.position_at(0.0) == pytest.approx((1.0, 2.0))
    assert plan.position_at(0.5) == pytest.approx((1.0, 2.0))
    assert plan.position_at(plan.duration_s) == pytest.approx((3.0, 3.5))
    assert plan.position_at(plan.duration_s - 0.9) == pytest.approx((3.0, 3.5))
    steps = np.linalg.norm(np.diff(plan.frames_xz, axis=0), axis=1) * plan.fps
    assert steps.max() == pytest.approx(1.0, abs=0.02)
    assert plan.arrival_s == pytest.approx(0.5 + 3.5 + 0.4, abs=1.0 / plan.fps)


def test_kimodo_constraints_are_relative_to_the_route_start():
    from wt_genesis.adapter.kimodo_path import plan_root_path

    plan = plan_root_path(ROUTE, speed_mps=1.0)
    constraints = plan.kimodo_constraints()

    frames = sorted(index for item in constraints for index in item["frame_indices"])
    assert frames == list(range(plan.frame_count))
    assert all(item["type"] == "root2d" for item in constraints)
    points = {index: point for item in constraints
              for index, point in zip(item["frame_indices"], item["smooth_root_2d"])}
    assert points[0] == pytest.approx([0.0, 0.0])
    assert points[plan.frame_count - 1] == pytest.approx([2.0, 1.5])
    json.dumps(constraints)
    # Kimodo makes int(duration * fps) frames: the passed duration must give
    # exactly the constrained frame count, not one short.
    assert int(float(plan.kimodo_duration_s) * plan.fps) == plan.frame_count


def test_waypoint_times_place_the_human_on_time_and_repeat_means_pause():
    from wt_genesis.adapter.kimodo_path import plan_root_path

    plan = plan_root_path(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        waypoint_times=[0.0, 2.0, 4.0, 6.0], start_hold_s=0.0, end_hold_s=0.5)

    assert plan.position_at(1.0) == pytest.approx((1.0, 0.0), abs=0.05)
    assert plan.position_at(2.0) == pytest.approx((2.0, 0.0))
    assert plan.position_at(3.0) == pytest.approx((2.0, 0.0))
    assert plan.position_at(5.0) == pytest.approx((2.0, 1.0), abs=0.05)
    assert plan.duration_s == pytest.approx(6.5, abs=1.0 / plan.fps)


def test_end_heading_is_a_kimodo_heading_constraint_on_the_final_frames():
    from wt_genesis.adapter.kimodo_path import heading_from_direction, plan_root_path

    # Kimodo heading 0 faces +Z; facing +X is +pi/2 (atan2(dx, dz)).
    assert heading_from_direction((1.0, 0.0)) == pytest.approx(math.pi / 2)
    assert heading_from_direction((0.0, 1.0)) == pytest.approx(0.0)

    plan = plan_root_path(ROUTE, end_heading=math.pi / 2, end_hold_s=1.5)
    headed = [item for item in plan.kimodo_constraints() if "global_root_heading" in item]

    # The start faces along the first leg (+X); the end faces the requested way.
    assert len(headed) == 2
    start, end = sorted(headed, key=lambda item: item["frame_indices"][0])
    assert start["frame_indices"][0] == 0
    assert start["global_root_heading"][0] == pytest.approx([0.0, 1.0], abs=1e-6)
    assert end["frame_indices"][-1] == plan.frame_count - 1
    assert end["global_root_heading"][-1] == pytest.approx([0.0, 1.0], abs=1e-6)
    indices = [index for item in plan.kimodo_constraints() for index in item["frame_indices"]]
    assert len(indices) == len(set(indices))


def test_sitting_leaves_the_turn_and_sit_to_kimodo_and_pins_the_seat():
    """The walk ends in front of the seat; Kimodo turns and sits on its own.

    Only the last frame is pinned: the root over the seat, facing away from
    it. Pinning the frames between would drag a standing body into the seat.
    """
    from wt_genesis.adapter.kimodo_path import plan_root_path

    plan = plan_root_path([(0.0, 0.0), (2.0, 0.0)], speed_mps=1.0, end_hold_s=3.0,
                          sit_at=(2.0, -0.6), end_heading=math.pi)
    constraints = plan.kimodo_constraints()
    pinned = {index for item in constraints for index in item["frame_indices"]}
    arrival = int(round(plan.arrival_s * plan.fps))

    assert plan.sit_at == pytest.approx((2.0, -0.6))
    assert set(range(arrival + 1)) <= pinned
    assert not pinned & set(range(arrival + 1, plan.frame_count - 1))
    last = next(item for item in constraints
                if item["frame_indices"] == [plan.frame_count - 1])
    assert last["smooth_root_2d"] == [pytest.approx([2.0, -0.6])]
    assert last["global_root_heading"] == [pytest.approx([-1.0, 0.0], abs=1e-6)]
    assert plan.position_at(plan.duration_s) == pytest.approx((2.0, -0.6))
    assert plan.final_xz == pytest.approx((2.0, -0.6))


def test_plan_rejects_routes_it_cannot_walk():
    from wt_genesis.adapter.kimodo_path import plan_root_path

    with pytest.raises(ValueError):
        plan_root_path([(0.0, 0.0)])
    with pytest.raises(ValueError):
        plan_root_path([(0.0, 0.0), (1.0, float("nan"))])
    with pytest.raises(ValueError):
        plan_root_path([(0.0, 0.0), (1.0, 0.0)], waypoint_times=[0.0, 0.0])
    with pytest.raises(ValueError):
        plan_root_path([(0.0, 0.0), (30.0, 0.0)], speed_mps=1.0, max_duration_s=10.0)


def test_kimodo_backend_passes_the_constraint_list_to_the_cli(tmp_path, monkeypatch):
    import wt_genesis.adapter.kimodo_backend as backend

    fake_python = tmp_path / "python.exe"
    fake_python.touch()
    source = tmp_path / "kimodo"
    (source / "kimodo" / "scripts").mkdir(parents=True)
    (source / "kimodo" / "scripts" / "generate.py").touch()
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "token").write_text("not-a-real-token", encoding="utf-8")
    monkeypatch.setenv("WITWIN_KIMODO_PYTHON", str(fake_python))
    monkeypatch.setenv("WITWIN_KIMODO_ROOT", str(source))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(
        backend, "_probe_runtime",
        lambda python, root: {"importable": True, "cuda_available": True, "detail": "ok"})
    seen = []

    def run(command, **_kwargs):
        written = None
        if "--constraints" in command:
            with open(command[command.index("--constraints") + 1], encoding="utf-8") as handle:
                written = json.load(handle)
        seen.append((command, written))
        output = command[command.index("--output") + 1]
        rotations = np.repeat(np.eye(3)[None, None], 2 * 77, axis=0).reshape(2, 77, 3, 3)
        np.savez(f"{output}.npz", local_rot_mats=rotations, root_positions=np.zeros((2, 3)))

    monkeypatch.setattr(backend.subprocess, "run", run)
    constraints = [{"type": "root2d", "frame_indices": [0, 1], "smooth_root_2d": [[0, 0], [0, 1]]}]

    backend.generate_smpl_motion(["a person walks"], motion_length=2.0, seed=1,
                                 constraints=constraints)
    backend.generate_smpl_motion(["a person waves"], motion_length=2.0, seed=1)

    command, written = seen[0]
    assert written == constraints
    assert command[command.index("--duration") + 1] == "2.0"
    assert "--constraints" not in seen[1][0]
    with pytest.raises(ValueError):
        # Constraint frames index one clip; Kimodo's sentence segments would
        # shift them, so a constrained prompt must be a single sentence.
        backend.generate_smpl_motion(["a person walks. a person stops."], motion_length=2.0,
                                     seed=1, constraints=constraints)


def _route_following_generator(calls, *, start=(0.0, 0.92, 0.0), wobble=0.0):
    """A stand-in for Kimodo that walks exactly the constrained root path."""
    def generate(prompts, *, motion_length, seed, device=0, constraints=None, **_kwargs):
        calls.append({"prompts": list(prompts), "motion_length": motion_length,
                      "constraints": constraints, "seed": seed})
        points = {}
        for item in constraints:
            for index, point in zip(item["frame_indices"], item["smooth_root_2d"]):
                points[index] = point
        count = max(points) + 1
        known = sorted(points)
        source = np.array([[np.interp(index, known, [points[k][axis] for k in known])
                            for axis in (0, 1)] for index in range(count)], dtype=np.float64)
        times = np.arange(count) / 30.0
        out_times = np.arange(0.0, times[-1] + 1e-9, 1.0 / 20.0)
        xz = np.column_stack([np.interp(out_times, times, source[:, axis]) for axis in (0, 1)])
        frames = len(out_times)
        trans = np.zeros((1, frames, 3), dtype=np.float32)
        trans[0, :, 0] = xz[:, 0] + start[0] + wobble * np.sin(out_times * 7.0)
        trans[0, :, 1] = start[1]
        trans[0, :, 2] = xz[:, 1] + start[2]
        return {"smpl_pose": np.zeros((1, frames, 24, 3), dtype=np.float32),
                "smpl_betas": np.zeros((1, frames, 10), dtype=np.float32),
                "smpl_trans": trans}

    return generate


def _scene_with_tiny_skin(monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from tests.test_scene_animation import _tiny_skin_asset

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset",
                        lambda *args, **kwargs: _tiny_skin_asset())
    return Scene(scene_id="main")


def test_path_animation_lands_the_generated_walk_on_the_planned_route(tmp_path, monkeypatch):
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation

    scene = _scene_with_tiny_skin(monkeypatch)
    calls = []
    result = create_humanoid_path_animation(
        scene=scene, project_dir=tmp_path, prompt="A person walks. Then stops.",
        waypoints=[(1.0, 0.1, 2.0), (3.0, 0.1, 2.0), (3.0, 0.1, 3.5)],
        speed_mps=1.0, seed=7, generator=_route_following_generator(calls))

    assert calls[0]["prompts"] == ["A person walks; Then stops."]
    assert calls[0]["constraints"][0]["smooth_root_2d"][0] == pytest.approx([0.0, 0.0])
    track = scene.timeline_manager.clip.get_track(
        result.animation.body_object_id, "Transform", "position")
    first, last = track.keyframes[0].value, track.keyframes[-1].value
    assert [first[0], first[2]] == pytest.approx([1.0, 2.0], abs=1e-4)
    assert [last[0], last[2]] == pytest.approx([3.0, 3.5], abs=1e-4)
    assert first[1] == pytest.approx(0.1 + 0.92, abs=1e-4)
    following = result.path_following
    assert following["timed_max_error_m"] < 0.01  # 30 -> 20 fps resampling only
    assert following["route_max_error_m"] < 0.01
    assert following["end_error_m"] < 0.01
    assert following["samples"] == len(track.keyframes)
    assert result.to_dict()["plan"]["waypoints"][0] == pytest.approx([1.0, 2.0])


def test_path_following_reports_a_drifting_walk(tmp_path, monkeypatch):
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_humanoid_path_animation(
        scene=scene, project_dir=tmp_path, prompt="a person walks",
        waypoints=ROUTE, generator=_route_following_generator([], wobble=0.2))

    assert 0.15 < result.path_following["timed_max_error_m"] <= 0.2 + 1e-3
    assert result.path_following["route_max_error_m"] <= 0.2 + 1e-3


def test_kimodo_root_offset_from_the_route_is_kept_and_reported(tmp_path, monkeypatch):
    """Kimodo's hip root starts a few cm off the constrained smooth root.

    Placing the clip by its first frame would shift the whole walk by that
    offset; the Kimodo origin goes on the route start instead.
    """
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_humanoid_path_animation(
        scene=scene, project_dir=tmp_path, prompt="a person walks", waypoints=ROUTE,
        generator=_route_following_generator([], start=(0.03, 0.92, -0.02)))

    track = scene.timeline_manager.clip.get_track(
        result.animation.body_object_id, "Transform", "position")
    first = track.keyframes[0].value
    assert [first[0], first[2]] == pytest.approx([1.03, 1.98], abs=1e-4)
    assert result.path_following["timed_max_error_m"] == pytest.approx(0.036, abs=1e-3)


def test_timed_occupancy_is_the_humans_root_over_time(tmp_path, monkeypatch):
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation, humanoid_occupancy

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_humanoid_path_animation(
        scene=scene, project_dir=tmp_path, prompt="a person walks",
        waypoints=ROUTE, speed_mps=1.0, start_hold_s=0.0, ease_s=0.0, end_hold_s=0.5,
        generator=_route_following_generator([]))

    occupancy = humanoid_occupancy(scene, result.animation.body_object_id, fps=10.0)

    assert occupancy["radius_m"] > 0.2
    assert occupancy["samples"][0]["t"] == pytest.approx(0.0)
    assert occupancy["samples"][-1]["t"] == pytest.approx(result.animation.duration, abs=0.1)
    at_one = min(occupancy["samples"], key=lambda item: abs(item["t"] - 1.0))
    assert (at_one["x"], at_one["z"]) == pytest.approx((2.0, 2.0), abs=0.06)
    assert result.occupancy["body_object_id"] == result.animation.body_object_id
    assert len(result.occupancy["samples"]) >= len(occupancy["samples"])


def test_seat_contact_is_support_only_for_the_named_seat(tmp_path, monkeypatch):
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation
    from wt_genesis.adapter.trajectory_clearance import HumanoidTrajectoryBlocked

    def seated(obstacle):
        def validate(*_args):
            return {"status": "overlap_detected", "samples": 10, "affected_samples": 3,
                    "contacts": [{"time": 4.0, "body_part": "pelvis", "obstacle_id": obstacle,
                                  "obstacle": obstacle, "kind": "conservative_convex_overlap",
                                  "penetration_m": 0.04}]}
        return validate

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_humanoid_path_animation(
        scene=scene, project_dir=tmp_path, prompt="a person walks to the chair and sits down",
        waypoints=ROUTE, sit_support_ids=["chair_1"], body_validator=seated("chair_1"),
        generator=_route_following_generator([]))
    assert result.animation.trajectory_clearance["seat_support_contacts"] == 1

    with pytest.raises(HumanoidTrajectoryBlocked):
        create_humanoid_path_animation(
            scene=_scene_with_tiny_skin(monkeypatch), project_dir=tmp_path,
            prompt="a person walks to the chair and sits down", waypoints=ROUTE,
            sit_support_ids=["chair_1"], body_validator=seated("table_2"),
            generator=_route_following_generator([]))


def _contact(obstacle, depth, time=4.0):
    return {"time": time, "body_part": "pelvis", "obstacle_id": obstacle, "obstacle": obstacle,
            "kind": "conservative_convex_overlap", "penetration_m": depth}


def test_seat_depth_is_judged_after_the_body_is_lifted_onto_the_floor(tmp_path, monkeypatch):
    # bench_3007: Kimodo bodies start ~0.13 m sunk (SOMA hips used as the SMPL
    # root). The floor lift only runs when the floor is the sole blocker, and
    # the sunk sitting body also overlapped the bed by 0.15-0.2 m: every sit
    # was refused before it was lifted to where it actually sits.
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation

    reports = [
        {"status": "overlap_detected", "samples": 10, "affected_samples": 4,
         "contacts": [_contact("floor", .13, 0.2), _contact("bed_1", .2)]},
        {"status": "overlap_detected", "samples": 10, "affected_samples": 1,
         "contacts": [_contact("bed_1", .05)]},
    ]
    result = create_humanoid_path_animation(
        scene=_scene_with_tiny_skin(monkeypatch), project_dir=tmp_path,
        prompt="a person walks over and sits down", waypoints=ROUTE, sit_at=(3.0, 3.9),
        end_heading=0.0, sit_support_ids=["bed_1"], body_validator=lambda *_a: reports.pop(0),
        generator=_route_following_generator([]))

    clearance = result.animation.trajectory_clearance
    assert clearance["floor_support_correction"]["total_lift_m"] == pytest.approx(.131)
    assert clearance["seat_support_contacts"] == 1


def test_a_seat_still_too_deep_after_the_lift_is_refused(tmp_path, monkeypatch):
    from wt_genesis.adapter.kimodo_path import create_humanoid_path_animation
    from wt_genesis.adapter.trajectory_clearance import HumanoidTrajectoryBlocked

    reports = [
        {"status": "overlap_detected", "samples": 10, "affected_samples": 4,
         "contacts": [_contact("floor", .13, 0.2), _contact("bed_1", .3)]},
        {"status": "overlap_detected", "samples": 10, "affected_samples": 1,
         "contacts": [_contact("bed_1", .17)]},
    ]
    with pytest.raises(HumanoidTrajectoryBlocked):
        create_humanoid_path_animation(
            scene=_scene_with_tiny_skin(monkeypatch), project_dir=tmp_path,
            prompt="a person walks over and sits down", waypoints=ROUTE, sit_at=(3.0, 3.9),
            end_heading=0.0, sit_support_ids=["bed_1"], body_validator=lambda *_a: reports.pop(0),
            generator=_route_following_generator([]))
