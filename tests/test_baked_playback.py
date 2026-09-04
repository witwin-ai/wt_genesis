"""License-free producer and Studio playback contracts; no human/RF claims."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from wt_genesis.adapter.baked_playback import select_playback_keyframes, verify_recorded_baked_playback


def _bundle(frame_count=3):
    keys = [dict(object_id="source", component_name="SMPLMotion", field_name="current_frame",
                 value=frame, time=time, interpolation="linear")
            for frame, time in [(0, 0.), (frame_count-1, max(1, frame_count-1)/20.)]]
    for object_id, field in [("human", "position"), ("human", "rotation"),
                             ("human-smpl-bone-00", "rotation"), ("human-smpl-bone-01", "rotation")]:
        for frame in range(frame_count):
            value = [0., 0., 0.]
            if object_id == "human" and field == "position":
                value[0] = frame / 20.
            if object_id == "human-smpl-bone-01":
                value[0] = frame * .6  # Studio Euler order is ZXY.
            keys.append(dict(object_id=object_id, component_name="Transform", field_name=field,
                             value=value, time=frame/20., interpolation="linear"))
    return keys


def _select(keys, **overrides):
    args = dict(playback_mode="baked_transforms", body_object_id="human", motion_object_id="source",
                bone_ids=["human-smpl-bone-00", "human-smpl-bone-01"], frame_count=3, fps=20.)
    args.update(overrides)
    return select_playback_keyframes(keys, **args)


def test_baked_bundle_has_single_authority_without_mutating_input():
    source = _bundle()
    untouched = deepcopy(source)
    keys, receipt = _select(source)
    assert source == untouched
    assert len(keys) == 12
    assert all(key["component_name"] == "Transform" for key in keys)
    assert receipt["pose_authority"] == "baked_root_and_bone_transforms"
    assert receipt["source_component_retained"] is True
    assert receipt["source_controller_keyed"] is False
    assert receipt["parametric_smpl_animation_export_supported"] is False
    keys[0]["value"][0] = 99.
    assert source == untouched


def test_legacy_is_default_and_preserves_bundle():
    keys, receipt = select_playback_keyframes(_bundle(), body_object_id="human", motion_object_id="source",
                                            bone_ids=[], frame_count=3, fps=20.)
    assert keys == _bundle()
    assert receipt["mode"] == "legacy_controller"
    assert receipt["source_controller_keyed"] is True


def test_single_frame_bake_is_valid():
    keys, receipt = _select(_bundle(1), frame_count=1)
    assert len(keys) == 4
    assert receipt["source_frame_count"] == 1


@pytest.mark.parametrize("fps", [10000., 100000.])
def test_host_would_merge_dense_source_keys_so_bundle_is_rejected(fps):
    keys = _bundle()
    for key in keys:
        key["time"] *= 20. / fps
    # Demonstrate this is the host's real storage limit, not a guessed FPS cap.
    scene, _, _ = _procedural_scene()
    scene.timeline_manager.record_keyframes(keys[2:])
    assert sum(len(track.keyframes) for track in scene.timeline_manager.clip.tracks.values()) < 12
    with pytest.raises(ValueError, match="time resolution"):
        _select(keys, fps=fps)


@pytest.mark.parametrize("failure", [None, "missing", "merged", "time", "value", "interpolation", "controller"])
def test_recorded_bake_verified_not_only_proposed_bundle(failure):
    from witwin_server.features.timeline import InterpolationType
    scene, _, _ = _procedural_scene()
    timeline = scene.timeline_manager
    keys, receipt = _select(_bundle())
    timeline.record_keyframes(keys)
    track = timeline.clip.get_track("human", "Transform", "position")
    if failure == "missing":
        timeline.clip.tracks.pop(track.path)
    elif failure == "merged":
        track.keyframes.pop()
    elif failure == "time":
        track.keyframes[-1].time += .01
    elif failure == "value":
        # The host may retain references to input values; mimic an independent
        # replacement so the expected producer bundle stays unchanged.
        track.keyframes[-1].value = [99., 0., 0.]
    elif failure == "interpolation":
        track.keyframes[-1].interpolation = InterpolationType.STEP
    elif failure == "controller":
        timeline.record_keyframes(_bundle()[:2])
    if failure:
        with pytest.raises(ValueError, match="Studio"):
            verify_recorded_baked_playback(timeline, keys, receipt)
    else:
        verify_recorded_baked_playback(timeline, keys, receipt)


@pytest.mark.parametrize("change", [
    "missing_bone", "missing_root", "missing_frame", "duplicate_frame", "wrong_grid",
    "unknown_track", "wrong_source", "nonlinear", "nonfinite", "wrong_vector",
    "missing_controller", "wrong_controller_value", "wrong_controller_time",
])
def test_malformed_or_unrelated_tracks_are_not_silently_discarded(change):
    keys = _bundle()
    if change == "missing_bone":
        keys = [key for key in keys if key["object_id"] != "human-smpl-bone-01"]
    elif change == "missing_root":
        keys = [key for key in keys if key["object_id"] != "human"]
    elif change == "missing_frame":
        keys.pop()
    elif change == "duplicate_frame":
        keys.append(deepcopy(keys[-1]))
    elif change == "wrong_grid":
        keys[-1]["time"] += .01
    elif change == "unknown_track":
        keys.append(dict(keys[-1], object_id="someone_else"))
    elif change == "wrong_source":
        keys[0]["object_id"] = "different_source"
    elif change == "nonlinear":
        keys[-1]["interpolation"] = "step"
    elif change == "nonfinite":
        keys[-1]["value"][0] = float("nan")
    elif change == "wrong_vector":
        keys[-1]["value"] = [0., 0.]
    elif change == "missing_controller":
        keys = keys[2:]
    elif change == "wrong_controller_value":
        keys[1]["value"] = 3
    elif change == "wrong_controller_time":
        keys[1]["time"] = .2
    with pytest.raises(ValueError):
        _select(keys)


@pytest.mark.parametrize("args", [
    {"playback_mode": "ignore_unknown"}, {"fps": 0.}, {"fps": float("nan")},
    {"frame_count": 0}, {"frame_count": True}, {"bone_ids": []},
    {"bone_ids": ["bone", "bone"]}, {"bone_ids": ["source"]},
    {"motion_object_id": "human"},
])
def test_invalid_contract_rejected(args):
    with pytest.raises(ValueError):
        _select(_bundle(), **args)


def _procedural_scene():
    """An invented 2-bone triangle, not SMPL model geometry or neural output."""
    from witwin_server.core import Scene, SceneObject
    from witwin_server.core.components import PlatformGeometryComponent
    from witwin_server.features.timeline import TimelineManager
    from wt_humanoid.adapter.smpl_motion import SMPLMotionData
    from wt_humanoid.adapter.smpl_skinned import SMPLSkinAsset, attach_smpl_skinned_mesh
    from wt_humanoid.components.smpl import SMPLBodyComponent, SMPLMotionComponent

    scene = Scene(scene_id="license_free_playback_contract")
    target = SceneObject(id="human", name="Procedural contract proxy", mesh_type="Empty")
    body = target.add_component(SMPLBodyComponent())
    target.add_component(PlatformGeometryComponent())
    scene.add_object(target)
    matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    matrices[1, 1, 3] = -1.
    asset = SMPLSkinAsset(
        vertices=np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        skin_indices=np.array([[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        skin_weights=np.array([[.75, .25, 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]], dtype=np.float32),
        joint_positions=np.array([[0., 0., 0.], [0., 1., 0.]], dtype=np.float32),
        parent_indices=[-1, 0], inverse_bind_matrices=matrices)
    attach_smpl_skinned_mesh(scene, target, body, asset=asset)
    source = SceneObject(id="source", name="In-memory pose source", mesh_type="Empty")
    motion = source.add_component(SMPLMotionComponent())
    scene.add_object(source)
    motion.target_body = {"object_id": "human", "component_type": "SMPLBody", "object_name": target.name}
    motion.motion_path = "synthetic-in-memory-no-licensed-data"
    poses = np.zeros((3, 72), dtype=np.float32)
    poses[:, 5] = [0., .6, 1.2]
    motion._motion_cache = SMPLMotionData(poses, np.zeros(10, dtype=np.float32),
                                        np.zeros((3, 3), dtype=np.float32), 20., motion.motion_path)
    motion._motion_cache_key = (motion.motion_path, None, None, None)
    motion.loop = "off"
    motion.apply_current_frame(body)
    scene.timeline_manager = TimelineManager(scene)
    scene.timeline_manager.duration = .1
    scene.timeline_manager.fps = 20.
    return scene, target, source


def _vertices(scene, target, times):
    result = []
    for time in times:
        scene.timeline_manager.set_time(time, apply_to_scene=True)
        skin = target.get_component("SkinnedMesh")
        result.append(skin.compute_skinned_vertices(in_local_space=False).detach().cpu().numpy().copy())
    return np.stack(result)


def test_baked_subframes_match_existing_controller_first_mesh_not_controller_last():
    times = np.linspace(0., .1, 9)
    scene, target, _ = _procedural_scene()
    timeline = scene.timeline_manager
    timeline.record_keyframes(_bundle())
    legacy = _vertices(scene, target, times)
    controller = timeline.clip.get_track("source", "SMPLMotion", "current_frame")
    timeline.clip.tracks.pop(controller.path)
    baked = _vertices(scene, target, times)
    np.testing.assert_allclose(legacy, baked, atol=2e-6, rtol=0.)
    timeline.clip.tracks[controller.path] = controller
    reordered = _vertices(scene, target, times)
    assert np.max(np.linalg.norm(reordered - baked, axis=-1)) > .6


def test_baked_source_component_remains_unkeyed_and_parameter_pose_is_not_live():
    scene, target, source = _procedural_scene()
    keys, _ = _select(_bundle())
    scene.timeline_manager.record_keyframes(keys)
    initial_pose = list(target.get_component("SMPLBody").pose)
    initial_geometry_pose = list(target.get_component("PlatformGeometry").pose)
    frames = _vertices(scene, target, [0., .075])
    assert not np.allclose(frames[0], frames[1])
    assert source.get_component("SMPLMotion").current_frame == 0
    assert target.get_component("SMPLBody").pose == initial_pose
    assert target.get_component("PlatformGeometry").pose == initial_geometry_pose
    assert scene.timeline_manager.clip.get_track("source", "SMPLMotion", "current_frame") is None


def test_current_unmodified_radar_sampler_accepts_baked_proxy_only():
    sampler_module = pytest.importorskip("wt_radar.adapter.studio_motion")
    scene, target, _ = _procedural_scene()
    timeline = scene.timeline_manager
    timeline.record_keyframes(_bundle())
    with pytest.raises(ValueError, match="unsupported track source/SMPLMotion/current_frame"):
        sampler_module.StudioSkinSampler(scene, target.id)
    timeline.remove_object_tracks("source")
    sampler = sampler_module.StudioSkinSampler(scene, target.id)
    positions, velocities = sampler.sample(.025)
    assert np.isfinite(positions).all() and np.isfinite(velocities).all()
    assert np.linalg.norm(velocities) > .5
    expected = _vertices(scene, target, [.025])[0, sampler.vertex_ids]
    np.testing.assert_allclose(positions, expected, atol=2e-6, rtol=0.)


def test_explicit_producer_baked_mode_reaches_existing_mesh_sampler(tmp_path, monkeypatch):
    """Real authoring/loading/baking path with invented motion+skin, no RF solve."""
    from witwin_server.core import Scene
    from wt_genesis.adapter import scene_animation
    from test_scene_animation import _animated_backend_for_frames, _tiny_skin_asset

    sampler_module = pytest.importorskip("wt_radar.adapter.studio_motion")
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *_args, **_kw: _tiny_skin_asset())
    scene = Scene(scene_id="new_human_producer_contract")
    result = scene_animation.create_humanoid_animation(
        scene=scene, project_dir=tmp_path, prompt="a person waves", motion_length=.2,
        playback_mode="baked_transforms", backend=_animated_backend_for_frames(4))
    assert result.ok
    assert result.playback["source_controller_keyed"] is False
    sampler = sampler_module.StudioSkinSampler(scene, result.body_object_id)
    samples = [sampler.sample(time) for time in (0., .025, .075, .1)]
    assert all(np.isfinite(positions).all() and np.isfinite(velocities).all()
               for positions, velocities in samples)
    assert not np.allclose(samples[0][0], samples[-1][0])
