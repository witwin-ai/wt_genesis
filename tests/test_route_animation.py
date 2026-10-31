"""A person walks a planned route to a target and ends in a pose.

The agent audit found that agents could only give a text prompt and a start
position, so "walk to the sofa and sit" never worked: the bed's coordinates
went into the prompt. The route animation composes the pieces the meeting
samples were made with (Kimodo route + end pose, foot lock, the pelvis-pivot
binding, a lively rest to the clip end) behind one call. These tests use a
stand-in for Kimodo that walks exactly the constrained root path.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.test_kimodo_path import ROUTE, _route_following_generator, _scene_with_tiny_skin


def _bone_ids(scene, body_id):
    return list(scene.get_object(body_id).get_component("SkinnedMesh").bone_ids)


def _body_tracks(scene, body_id):
    ids = {body_id, *_bone_ids(scene, body_id)}
    return [track for track in scene.timeline_manager.clip.tracks.values()
            if str(track.object_id) in ids and track.keyframes]


def test_a_standing_route_rests_alive_to_the_clip_end(tmp_path, monkeypatch):
    from wt_genesis.adapter.route_animation import create_route_animation

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, end="stand",
        speed_mps=1.0, rest_until_s=8.0, foot_lock=False,
        generator=_route_following_generator([]))

    body = result.body_object_id
    # Every track of the person ends on the clip end: nobody freezes early.
    ends = {round(max(float(key.time) for key in track.keyframes), 4)
            for track in _body_tracks(scene, body)}
    assert ends == {8.0}
    assert result.rest["until_s"] == pytest.approx(8.0)
    assert result.rest["frames"] > 0
    # The rest continues the last pose with the upper body only: the clip
    # file's pelvis, legs and root stay as they were, the head moves.
    data = np.load(result.path.animation.motion_asset.motion_path)
    pose, root = data["smpl_pose"], data["root"]
    rest = slice(len(pose) - result.rest["frames"], len(pose))
    assert pose.shape[0] == root.shape[0]
    assert data["shape"].shape == (10,)
    for joint in (0, 1, 2, 4, 5, 7, 8, 10, 11):
        assert np.abs(pose[rest, joint] - pose[rest.start - 1, joint]).max() < 1e-6
    assert np.abs(root[rest] - root[rest.start - 1]).max() < 1e-6
    assert np.abs(pose[rest, 15] - pose[rest.start - 1, 15]).max() > math.radians(5)


def test_the_rest_goes_to_the_scene_clip_end_by_default(tmp_path, monkeypatch):
    from witwin_server.features.timeline.data import InterpolationType
    from wt_genesis.adapter.route_animation import create_route_animation
    from wt_genesis.adapter.scene_animation import _ensure_timeline

    scene = _scene_with_tiny_skin(monkeypatch)
    timeline = _ensure_timeline(scene)
    timeline.record_keyframes([
        {"object_id": "cat", "component_name": "Transform", "field_name": "position",
         "value": [0.0, 0.0, 0.0], "time": time, "interpolation": InterpolationType.LINEAR}
        for time in (0.0, 9.0)])
    timeline.duration = 9.0

    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, speed_mps=1.0, foot_lock=False,
        generator=_route_following_generator([]))

    assert result.rest["until_s"] == pytest.approx(9.0)
    assert timeline.duration == pytest.approx(9.0)


def test_the_feet_are_cleaned_on_the_grounded_clip_before_it_is_bound(tmp_path, monkeypatch):
    import wt_genesis.adapter.route_animation as route

    calls = []

    def clean(pose, root, skin, fps):
        calls.append((pose.shape, root.shape, fps))
        return pose, root - np.array([0.0, 0.01, 0.0]), {"held": 3}

    monkeypatch.setattr(route, "foot_lock_clip", clean)
    scene = _scene_with_tiny_skin(monkeypatch)
    result = route.create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, speed_mps=1.0, rest_until_s=None,
        foot_lock=True, skin=object(), generator=_route_following_generator([]))

    assert len(calls) == 1
    assert result.foot_lock == {"held": 3}
    motion = result.path.animation.motion_asset.motion_path
    before = np.load(motion.replace(".npz", ".prelock.npz"))["root"]
    after = np.load(motion)["root"]
    assert after[:, 1] == pytest.approx(before[:, 1] - 0.01)
    track = scene.timeline_manager.clip.get_track(result.body_object_id, "Transform", "position")
    assert track.keyframes[0].value[1] == pytest.approx(0.92 - 0.01, abs=1e-4)


def test_the_body_turns_about_its_pelvis_joint(tmp_path, monkeypatch):
    # The pelvis-pivot binding (wt_genesis 2590eec): root = translation +
    # J0 - R J0. A body bound the old way swayed 7-10 cm when tilted.
    from wt_genesis.adapter.route_animation import create_route_animation

    follow = _route_following_generator([])

    def lean(prompts, **kwargs):
        motion = follow(prompts, **kwargs)
        frames = motion["smpl_pose"].shape[1]
        motion["smpl_pose"][0, :, 0, 0] = np.linspace(0.0, 1.2, frames)  # tips forward
        return motion

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, speed_mps=1.0, rest_until_s=None,
        foot_lock=False, generator=lean)

    assert result.binding["pivot"] == "smpl_pelvis_j0"
    assert result.binding["frames_checked"] == result.path.animation.motion_asset.frame_count
    assert result.binding["max_error_m"] < 2e-3


def test_the_result_reports_the_interval_route_and_arrival(tmp_path, monkeypatch):
    from wt_genesis.adapter.route_animation import create_route_animation

    scene = _scene_with_tiny_skin(monkeypatch)
    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, speed_mps=1.0, start_time=1.0,
        rest_until_s=None, foot_lock=False, generator=_route_following_generator([]))
    payload = result.to_dict()

    assert payload["end"] == "stand"
    assert payload["body_object_id"] == result.body_object_id
    assert payload["interval_s"][0] == pytest.approx(1.0)
    assert payload["route"]["length_m"] == pytest.approx(3.5)
    assert payload["key_events"]["arrival"] == pytest.approx(1.0 + result.path.plan.arrival_s)
    assert payload["path_following"]["timed_max_error_m"] < 0.01
    assert payload["attempts"] == [{"seed": 10, "status": "kept"}]


def _box(object_id, low, high, *, name=None):
    """A scene_geometry item: a box's corners and its top as two triangles."""
    (x0, y0, z0), (x1, y1, z1) = low, high
    points = [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    vertices = [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]
    return {"object_id": object_id, "name": name or object_id, "points": points,
            "vertices": vertices, "faces": [[0, 1, 2], [0, 2, 3]]}


SOFA = _box("sofa1", (2.5, 0.0, 4.0), (3.5, 0.42, 4.9))


def _seat_skin():
    from tests.test_kimodo_poses import _seat_only_skin

    return _seat_only_skin()


def _pins(calls):
    return [item for item in calls[0]["constraints"] if item["type"] == "end-effector"]


def test_a_sit_on_a_target_finds_its_seat_heading_and_height_from_the_scene(tmp_path, monkeypatch):
    from wt_genesis.adapter.route_animation import create_route_animation

    calls = []
    scene = _scene_with_tiny_skin(monkeypatch)
    # The route ends 0.5 m in front of the sofa's z = 4.0 edge.
    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, end="sit", target_object_id="sofa1",
        speed_mps=1.0, rest_until_s=None, foot_lock=False, skin=_seat_skin(),
        geometry_reader=lambda _scene: [SOFA], generator=_route_following_generator(calls))

    plan = result.path.plan
    assert plan.sit_at == pytest.approx((3.0, 4.16))
    assert plan.end_heading == pytest.approx(math.pi)          # faces out of the seat (-z)
    pins = _pins(calls)
    assert len(pins) == 1 and pins[0]["joint_names"][0] == "Hips"
    # Pelvis 0.24 m over the 0.42 m seat top, relative to the route start (1, 2).
    assert pins[0]["root_positions"][0] == pytest.approx([2.0, 0.66, 2.16])
    assert result.end_pose["support_height_m"] == pytest.approx(0.42)
    assert result.path.animation.trajectory_clearance.get("seat_support_ids") is None  # no validator
    payload = result.to_dict()
    assert payload["target"] == {"object_id": "sofa1", "support_height_m": 0.42,
                                 "height_source": "mesh_top", "support_rect": [2.5, 4.0, 3.5, 4.9],
                                 "sit_at": [3.0, 4.16], "end_heading": pytest.approx(math.pi)}
    assert "seat_contact" in payload["key_events"]


def test_a_sit_needs_the_route_to_end_in_front_of_the_seat(tmp_path, monkeypatch):
    from wt_genesis.adapter.route_animation import RouteTargetUnsupported, create_route_animation

    calls = []
    scene = _scene_with_tiny_skin(monkeypatch)
    for route, words in (([(1.0, 2.0), (3.0, 4.5)], "inside"),
                         ([(1.0, 2.0), (3.0, 2.0)], "from the seat")):
        with pytest.raises(RouteTargetUnsupported, match=words):
            create_route_animation(
                scene=scene, project_dir=tmp_path, waypoints=route, end="sit",
                target_object_id="sofa1", geometry_reader=lambda _scene: [SOFA],
                foot_lock=False, generator=_route_following_generator(calls))
    with pytest.raises(RouteTargetUnsupported, match="no geometry"):
        create_route_animation(
            scene=scene, project_dir=tmp_path, waypoints=ROUTE, end="sit",
            target_object_id="armchair9", geometry_reader=lambda _scene: [SOFA],
            foot_lock=False, generator=_route_following_generator(calls))
    tall = _box("bar_stool", (2.5, 0.0, 4.0), (3.5, 0.75, 4.9))
    with pytest.raises(RouteTargetUnsupported, match="too high"):
        create_route_animation(
            scene=scene, project_dir=tmp_path, waypoints=ROUTE, end="sit",
            target_object_id="bar_stool", geometry_reader=lambda _scene: [tall],
            foot_lock=False, generator=_route_following_generator(calls))
    assert calls == []  # refused before Kimodo runs


def test_a_lie_pins_the_pelvis_along_the_bed_from_its_head_end(tmp_path, monkeypatch):
    from wt_genesis.adapter.route_animation import RouteTargetUnsupported, create_route_animation

    bed = _box("bed1", (2.0, 0.0, 4.0), (4.0, 0.5, 5.0))
    calls = []
    scene = _scene_with_tiny_skin(monkeypatch)
    with pytest.raises(RouteTargetUnsupported, match="head_direction"):
        create_route_animation(
            scene=scene, project_dir=tmp_path, waypoints=ROUTE, end="lie", target_object_id="bed1",
            geometry_reader=lambda _scene: [bed], foot_lock=False,
            generator=_route_following_generator(calls))
    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=ROUTE, end="lie", target_object_id="bed1",
        head_direction=(1.0, 0.0), rest_until_s=None, foot_lock=False, skin=_seat_skin(),
        geometry_reader=lambda _scene: [bed], generator=_route_following_generator(calls))

    pins = _pins(calls)
    # Head end at x = 4.0, pelvis 0.95 m from it on the bed's centre line,
    # 0.20 m over the 0.5 m mattress.
    assert pins[0]["root_positions"][0] == pytest.approx([3.05 - 1.0, 0.70, 4.5 - 2.0])
    assert result.path.plan.sit_at is None
    assert result.to_dict()["target"]["head_direction"] == pytest.approx([1.0, 0.0])


def test_a_fall_needs_clear_floor_ahead_of_the_last_step(tmp_path, monkeypatch):
    from wt_genesis.adapter.route_animation import FallZoneBlocked, create_route_animation

    route = [(0.0, 0.0), (2.0, 0.0)]
    table = _box("table1", (3.0, 0.0, -0.5), (3.6, 0.75, 0.5))
    rug = _box("floor_rug", (-1.0, 0.0, -1.0), (6.0, 0.01, 1.0))
    calls = []
    scene = _scene_with_tiny_skin(monkeypatch)
    with pytest.raises(FallZoneBlocked, match="table1") as blocked:
        create_route_animation(
            scene=scene, project_dir=tmp_path, waypoints=route, end="fall",
            geometry_reader=lambda _scene: [table, rug], foot_lock=False,
            generator=_route_following_generator(calls))
    assert blocked.value.blockers[0]["object_id"] == "table1"
    assert calls == []

    far = _box("table1", (4.5, 0.0, -0.5), (5.1, 0.75, 0.5))
    result = create_route_animation(
        scene=scene, project_dir=tmp_path, waypoints=route, end="fall", rest_until_s=None,
        geometry_reader=lambda _scene: [far, rug], foot_lock=False, skin=_seat_skin(),
        generator=_route_following_generator(calls))
    assert result.to_dict()["target"]["fall_zone"] == {
        "status": "clear", "reach_m": 1.8, "body_radius_m": 0.3}
    assert _pins(calls) == []  # a fall is left to Kimodo after the walk
