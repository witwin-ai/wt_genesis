"""License-free interface checks: synthetic motion/skin, real Scene/Timeline.

These validate authoring transactions, not MDM inference or licensed SMPL shape.
"""
import copy

import numpy as np
import pytest

from witwin_server.core import SceneObject
from witwin_server.core.scene import Scene
from witwin_server.features.timeline.manager import TimelineManager

from wt_genesis.adapter import scene_animation as adapter
from test_scene_animation import _animated_backend_for_frames, _tiny_skin_asset


def _existing_scene():
    scene = Scene(scene_id="isolated_authoring_test")
    existing = SceneObject(name="Existing cat", mesh_type="Empty")
    scene.add_object(existing)
    timeline = TimelineManager(scene)
    scene.timeline_manager = timeline
    timeline.duration = 20.
    timeline.fps = 30.
    timeline.is_looping = False
    timeline.record_keyframes([
        {"object_id": existing.id, "component_name": "Transform",
         "field_name": "position", "time": 0., "value": [0., 0., 0.]},
        {"object_id": existing.id, "component_name": "Transform",
         "field_name": "position", "time": 20., "value": [20., 0., 0.]},
    ])
    timeline.set_time(4., apply_to_scene=True)
    timeline.is_playing = True
    return scene, existing


def _snapshot(scene, existing):
    return copy.deepcopy((existing.to_dict(), scene.timeline_manager.to_dict(),
                          scene.timeline_manager.is_playing, list(scene.objects), list(scene.root_order)))


def _create(scene, tmp_path, **kwargs):
    return adapter.create_humanoid_animation(
        scene=scene, project_dir=tmp_path, prompt="a person waves",
        motion_length=.2, device=-1, backend=_animated_backend_for_frames(4), **kwargs)


@pytest.fixture(autouse=True)
def synthetic_skin(monkeypatch):
    monkeypatch.setattr(adapter, "build_smpl_skin_asset", lambda *_args, **_kwargs: _tiny_skin_asset())


def test_missing_licensed_model_does_not_create_scene_objects_or_touch_timeline(tmp_path, monkeypatch):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("licensed SMPL_NEUTRAL.pkl is unavailable")

    monkeypatch.setattr(adapter, "build_smpl_skin_asset", missing)
    with pytest.raises(FileNotFoundError, match="SMPL_NEUTRAL"):
        _create(scene, tmp_path)
    assert _snapshot(scene, existing) == before
    # Preserve the generated asset for diagnosis/reuse; no user files removed.
    assert len(list((tmp_path / "assets/motions").glob("*.npz"))) == 1


@pytest.mark.parametrize("recording", [False, True])
def test_short_human_animation_preserves_existing_timeline_and_current_cat_pose(tmp_path, recording):
    scene, existing = _existing_scene()
    scene.timeline_manager.is_recording = recording
    old_object = copy.deepcopy(existing.to_dict())
    old_track = copy.deepcopy(next(iter(scene.timeline_manager.clip.tracks.values())).to_dict())
    result = _create(scene, tmp_path, loop=True, autoplay=False)
    timeline = scene.timeline_manager
    assert result.ok is True
    assert result.duration == pytest.approx(.15)
    assert timeline.duration == 20.
    assert timeline.fps == 30.
    assert timeline.is_looping is False
    assert timeline.current_time == 4.
    assert timeline.is_playing is True
    assert timeline.is_recording is recording
    assert existing.to_dict() == old_object
    assert timeline.clip.get_track(existing.id, "Transform", "position").to_dict() == old_track
    assert len(scene.objects) == 5  # retained cat + body/controller + two synthetic bones
    assert all(key.time <= result.duration + 1e-8
               for track in timeline.clip.tracks.values() if track.object_id != existing.id
               for key in track.keyframes)


@pytest.mark.parametrize("phase", ["attach_skin", "partial_keyframes", "broadcast"])
def test_authoring_failure_rolls_back_only_new_animation(tmp_path, monkeypatch, phase):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)
    if phase == "attach_skin":
        original = adapter.attach_smpl_skinned_mesh

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected attachment failure")

        monkeypatch.setattr(adapter, "attach_smpl_skinned_mesh", fail)
    elif phase == "partial_keyframes":
        original = scene.timeline_manager.record_keyframes

        def fail(records):
            original(records[:2])
            raise RuntimeError("injected partial-keyframe failure")

        monkeypatch.setattr(scene.timeline_manager, "record_keyframes", fail)
    else:
        def fail(**_kwargs):
            raise RuntimeError("injected broadcast failure")

        monkeypatch.setattr(adapter, "_broadcast_live_scene_update", fail)

    with pytest.raises(RuntimeError, match="injected"):
        _create(scene, tmp_path)
    assert _snapshot(scene, existing) == before


def test_collision_with_existing_bone_id_preserves_that_object(tmp_path):
    scene, existing = _existing_scene()
    conflict_id = f"{scene.allocate_id('New Human')}-smpl-bone-00"
    conflict = SceneObject(id=conflict_id, name="User object", mesh_type="Empty")
    scene.add_object(conflict)
    before = _snapshot(scene, existing)
    with pytest.raises(ValueError, match="collide"):
        _create(scene, tmp_path, body_name="New Human")
    assert scene.get_object(conflict.id) is conflict
    assert _snapshot(scene, existing) == before


def test_failure_on_empty_scene_restores_absent_timeline_and_callbacks(tmp_path, monkeypatch):
    scene = Scene(scene_id="empty_failure_test")
    callbacks = list(scene._callbacks["component_updated"])

    def fail(**_kwargs):
        raise RuntimeError("injected last-step failure")

    monkeypatch.setattr(adapter, "_broadcast_live_scene_update", fail)
    with pytest.raises(RuntimeError, match="injected"):
        _create(scene, tmp_path)
    assert not scene.objects
    assert scene.timeline_manager is None
    assert scene._callbacks["component_updated"] == callbacks


@pytest.mark.parametrize("position", [[0., float("nan"), 0.], [0., float("inf"), 0.], [0., 0.]])
def test_invalid_position_is_rejected_before_generating_asset(tmp_path, monkeypatch, position):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)
    monkeypatch.setattr(adapter, "generate_motion_asset",
                        lambda *_args, **_kwargs: pytest.fail("must validate input before generation"))
    with pytest.raises(ValueError, match="position"):
        _create(scene, tmp_path, position=position)
    assert _snapshot(scene, existing) == before


def test_replay_or_other_write_guard_fails_before_generation(tmp_path, monkeypatch):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)

    def deny(_operation):
        raise RuntimeError("scene is not in authoring mode")

    monkeypatch.setattr(scene, "_require_author_write", deny)
    monkeypatch.setattr(adapter, "generate_motion_asset",
                        lambda *_args, **_kwargs: pytest.fail("must guard before generation"))
    with pytest.raises(RuntimeError, match="authoring"):
        _create(scene, tmp_path)
    assert _snapshot(scene, existing) == before


def test_floor_support_correction_allows_bounded_iterative_convergence(monkeypatch):
    calls = 0

    def validator(_scene, _object_id, _start_time, _end_time):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return {
                "status": "overlap_detected",
                "contacts": [{
                    "obstacle_id": "floor",
                    "obstacle": "floor",
                    "time": 4.5,
                    "penetration_m": 0.03,
                    "kind": "conservative_convex_overlap",
                }],
            }
        return {"status": "clear_at_samples", "samples": 120}

    monkeypatch.setattr(adapter, "_lift_root_track_suffix", lambda *_args, **_kwargs: None)
    result = adapter._validate_with_floor_support_correction(
        object(), "human", 0.0, 6.0, validator,
    )

    assert calls == 5
    assert result["floor_support_correction"]["attempts"] == 4


@pytest.mark.parametrize("key", ["smpl_pose", "smpl_betas", "smpl_trans"])
def test_nonfinite_generated_body_is_rejected_before_scene_mutation(tmp_path, monkeypatch, key):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)
    original = _animated_backend_for_frames(4)

    def invalid(*args, **kwargs):
        data = original(*args, **kwargs)
        data[key].flat[0] = np.nan
        return data

    with pytest.raises(ValueError, match="nonfinite"):
        adapter.create_humanoid_animation(scene=scene, project_dir=tmp_path,
                                          prompt="a person waves", backend=invalid)
    assert _snapshot(scene, existing) == before


def test_baked_animation_authors_one_pose_authority_and_preserves_existing_tracks(tmp_path):
    scene, existing = _existing_scene()
    old_object = copy.deepcopy(existing.to_dict())
    old_track = copy.deepcopy(next(iter(scene.timeline_manager.clip.tracks.values())).to_dict())
    result = _create(scene, tmp_path, playback_mode="baked_transforms")
    timeline = scene.timeline_manager
    assert result.playback["pose_authority"] == "baked_root_and_bone_transforms"
    assert result.playback["source_controller_keyed"] is False
    assert result.playback["parametric_smpl_animation_export_supported"] is False
    assert result.to_dict()["playback"] == result.playback
    assert timeline.clip.get_track(result.motion_object_id, "SMPLMotion", "current_frame") is None
    assert scene.get_object(result.motion_object_id).get_component("SMPLMotion") is not None
    assert result.timeline_track == {
        "path": f"{result.body_object_id}/Transform/position",
        "object_id": result.body_object_id,
        "component_name": "Transform", "field_name": "position",
        "keyframe_count": len(timeline.clip.get_track(result.body_object_id, "Transform",
                                                      "position").keyframes),
    }
    # Four source frames keyed (plus sub-frame keys where the fake clip turns fast).
    rate = result.motion_asset.fps
    assert sum(abs(key.time * rate - round(key.time * rate)) < 1e-6 for key in
               timeline.clip.get_track(result.body_object_id, "Transform", "position").keyframes) >= 4
    assert timeline.current_time == 4. and timeline.fps == 30. and timeline.duration == 20.
    assert timeline.clip.get_track(existing.id, "Transform", "position").to_dict() == old_track
    assert existing.to_dict() == old_object
    body = scene.get_object(result.body_object_id)
    bone = scene.get_object(body.get_component("SkinnedMesh").bone_ids[1])
    before = bone.get_component("Transform").get_quaternion().tolist()
    timeline.set_time(.075, apply_to_scene=True)
    assert bone.get_component("Transform").get_quaternion().tolist() != pytest.approx(before)
    # Controller stays as source metadata; it does not overwrite the baked mesh.
    assert scene.get_object(result.motion_object_id).get_component("SMPLMotion").current_frame == 0


def test_invalid_playback_mode_is_rejected_before_generation(tmp_path, monkeypatch):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)
    monkeypatch.setattr(adapter, "generate_motion_asset",
                        lambda *_args, **_kwargs: pytest.fail("must validate mode before generation"))
    with pytest.raises(ValueError, match="playback_mode"):
        _create(scene, tmp_path, playback_mode="guess")
    assert _snapshot(scene, existing) == before


def test_rejected_partial_bake_rolls_back_scene_and_preserves_timeline(tmp_path, monkeypatch):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)
    original = adapter._build_skeleton_rotation_keyframes

    def incomplete(*args, **kwargs):
        count, keys = original(*args, **kwargs)
        return count, keys[:-1]

    monkeypatch.setattr(adapter, "_build_skeleton_rotation_keyframes", incomplete)
    with pytest.raises(ValueError, match="every source frame"):
        _create(scene, tmp_path, playback_mode="baked_transforms")
    assert _snapshot(scene, existing) == before


def test_silently_partial_recording_is_detected_and_rolled_back(tmp_path, monkeypatch):
    scene, existing = _existing_scene()
    before = _snapshot(scene, existing)
    original = scene.timeline_manager.record_keyframes

    def incomplete(records):
        original(records[:-1])
        return len(records)  # A nominal success is not proof the keys survived.

    monkeypatch.setattr(scene.timeline_manager, "record_keyframes", incomplete)
    with pytest.raises(ValueError, match="complete baked track"):
        _create(scene, tmp_path, playback_mode="baked_transforms")
    assert _snapshot(scene, existing) == before
