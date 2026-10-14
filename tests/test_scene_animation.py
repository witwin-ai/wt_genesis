"""Scene and timeline integration for wt_genesis."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest


def _animated_backend(prompts, **kwargs):
    assert prompts == ["a person waves"]
    pose = np.zeros((1, 4, 24, 3), dtype=np.float32)
    for frame in range(4):
        pose[0, frame, :, :] = float(frame)
    betas = np.ones((1, 4, 10), dtype=np.float32)
    trans = np.array(
        [[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0]]],
        dtype=np.float32,
    )
    return {"smpl_pose": pose, "smpl_betas": betas, "smpl_trans": trans}


def test_generate_humanoid_animation_tool_creates_scene_timeline_and_playback(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.tools import make_generate_humanoid_animation_tool

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="main")
    tool = make_generate_humanoid_animation_tool(_FakeContext(scene), backend=_animated_backend)

    result = asyncio.run(
        tool.call(
            {
                "prompt": "a person waves",
                "motion_length": 0.2,
                "device": -1,
                "output_name": "wave scene",
                "body_name": "Generated SMPL Body",
                "motion_name": "Generated SMPL Motion",
                "position": [1.0, 0.0, 2.0],
                "autoplay": True,
                "loop": True,
            }
        )
    )

    assert result["ok"] is True
    assert result["frame_count"] == 4
    assert result["fps"] == 20.0

    body_obj = scene.get_object(result["body_object_id"])
    motion_obj = scene.get_object(result["motion_object_id"])
    assert body_obj is not None
    assert motion_obj is not None
    assert body_obj.get_component("Transform").position.tolist() == pytest.approx([1.0, 0.0, 2.0])

    body = body_obj.get_component("SMPLBody")
    geom = body_obj.get_component("PlatformGeometry")
    motion = motion_obj.get_component("SMPLMotion")
    assert body is not None
    assert geom is not None
    assert motion is not None
    assert geom.kind == "smpl"
    assert geom.model_root
    assert motion.motion_path == result["motion_path"]
    assert motion.pose_key == "smpl_pose"
    assert motion.betas_key == "shape"
    assert motion.trans_key == "root"
    assert motion.target_body["object_id"] == body_obj.id
    assert motion.target_body["component_type"] == "SMPLBody"
    assert motion.frame_count == 4
    assert motion.loop == "on"
    assert body_obj.get_component("SkinnedMesh") is not None
    assert body_obj.get_component("GenesisSMPLMesh") is None

    timeline = scene.timeline_manager
    assert timeline is not None
    assert timeline.fps == 20.0
    assert timeline.duration == pytest.approx(3.0 / 20.0)
    assert timeline.is_looping is True
    assert timeline.is_playing is True

    track = timeline.clip.get_track(motion_obj.id, "SMPLMotion", "current_frame")
    assert track is not None
    assert track.path == result["timeline_track"]["path"]
    assert [keyframe.value for keyframe in track.keyframes] == [0, 3]
    assert [keyframe.interpolation.value for keyframe in track.keyframes] == ["linear", "linear"]

    assert body.pose == pytest.approx([0.0] * 72)
    timeline.set_time(2.0 / 20.0, apply_to_scene=True)
    assert int(motion.current_frame) == 2
    assert body.pose == pytest.approx([2.0] * 72)
    child_bone = scene.get_object(body_obj.get_component("SkinnedMesh").bone_ids[1])
    assert child_bone.get_component("Transform").get_quaternion().tolist() != pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_apply_humanoid_motion_preset_materializes_reproducible_asset(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    import wt_genesis.tools as genesis_tools

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    monkeypatch.setattr(
        genesis_tools,
        "_motion_matching_body_validation",
        lambda *_args: {"ok": True, "status": "clear_at_samples", "contacts": []},
    )
    scene = Scene(scene_id="main")
    tool = genesis_tools.make_apply_humanoid_motion_preset_tool(_FakeContext(scene))

    result = asyncio.run(tool.call({
        "preset": "accidental_slip_v1",
        "scene_id": "main",
        "output_name": "curated slip",
        "position": [1.0, 0.0, 2.0],
        "playback_mode": "baked_transforms",
        "loop": False,
    }))

    assert result["ok"] is True
    assert result["frame_count"] == 120
    assert result["project_relative_path"] == "assets/motions/curated_slip.npz"
    assert result["playback"]["mode"] == "baked_transforms"
    assert scene.get_object(result["body_object_id"]) is not None
    assert (tmp_path / "assets" / "motions" / "curated_slip.npz").is_file()


def test_generate_humanoid_animation_rejects_blocked_root_sweep_before_scene_write(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    import wt_genesis.tools as genesis_tools
    from wt_genesis.tools import make_generate_humanoid_animation_tool

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    monkeypatch.setattr(genesis_tools, "_motion_matching_body_validation", lambda *_args: {
        "ok": True,
        "status": "overlap_detected",
        "samples": 5,
        "affected_samples": 2,
        "contacts": [{
            "time": 0.1,
            "body_part": "torso",
            "obstacle_id": "bed",
            "obstacle": "bed",
            "kind": "conservative_convex_overlap",
            "penetration_m": 0.2,
        }],
        "limitations": ["sampled, not CCD"],
    })
    scene = Scene(scene_id="main")
    tool = make_generate_humanoid_animation_tool(_FakeContext(scene), backend=_animated_backend)

    with pytest.raises(Exception) as raised:
        asyncio.run(tool.call({
            "prompt": "a person waves",
            "motion_length": 0.2,
            "device": -1,
            "position": [0.0, 0.0, 0.0],
        }))

    assert getattr(raised.value, "code", None) == "humanoid_trajectory_blocked"
    assert scene.objects == {}
    assert scene.timeline_manager is None


def test_generate_humanoid_animation_tool_saves_and_broadcasts_final_scene(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.tools import make_generate_humanoid_animation_tool

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="main")
    server = _FakeServer(scene)
    tool = make_generate_humanoid_animation_tool(_FakeContext(scene, server=server), backend=_animated_backend)

    result = asyncio.run(
        tool.call(
            {
                "prompt": "a person waves",
                "motion_length": 0.2,
                "device": -1,
                "output_name": "wave scene",
                "autoplay": False,
            }
        )
    )

    assert result["ok"] is True
    assert server.persistence.save_scene_calls == [True]
    assert any(msg["type"] == "objects_batch_created" for msg in server.messages)
    timeline_state = next(msg for msg in server.messages if msg["type"] == "timeline_state")
    assert any(track["path"] == result["timeline_track"]["path"] for track in timeline_state["tracks"])


def test_generate_humanoid_animation_persists_skinned_mesh_for_cold_load(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    from witwin_server.persistence import PersistenceManager
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.tools import make_generate_humanoid_animation_tool

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="persistent_human")

    class PersistentServer(_FakeServer):
        def __init__(self, active_scene):
            super().__init__(active_scene)
            self.verbose = False
            self.scenes = {active_scene.scene_id: active_scene}
            self.persistence = PersistenceManager(self)

    server = PersistentServer(scene)
    tool = make_generate_humanoid_animation_tool(
        _FakeContext(scene, server=server), backend=_animated_backend
    )

    result = asyncio.run(
        tool.call(
            {
                "prompt": "a person waves",
                "motion_length": 0.2,
                "device": -1,
                "output_name": "persistent wave",
                "autoplay": False,
            }
        )
    )

    cold_scene = server.persistence.load_scene_by_id(scene.scene_id)
    cold_body = cold_scene.get_object(result["body_object_id"])
    cold_mesh = cold_body.get_component("SkinnedMesh")
    cold_data = cold_mesh.get_mesh_data()

    assert cold_mesh.source["storage"] == "scene_binary"
    assert cold_data["vertices"]
    assert cold_data["faces"]
    assert cold_data["skin_indices"]
    assert cold_data["skin_weights"]


def test_generate_humanoid_animation_keys_root_motion_on_body_transform(tmp_path, monkeypatch):
    from scipy.spatial.transform import Rotation
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="main")
    result = create_humanoid_animation(
        scene=scene,
        project_dir=tmp_path,
        prompt="a person walks and turns",
        motion_length=0.15,
        device=-1,
        output_name="root motion",
        position=[1.0, 0.0, 2.0],
        backend=_root_motion_backend,
    )

    body_obj = scene.get_object(result.body_object_id)
    assert body_obj is not None
    transform = body_obj.get_component("Transform")

    position_track = scene.timeline_manager.clip.get_track(body_obj.id, "Transform", "position")
    rotation_track = scene.timeline_manager.clip.get_track(body_obj.id, "Transform", "rotation")
    assert position_track is not None
    assert rotation_track is not None
    expected_positions = [[1.0, 0.0, 2.0], [1.5, 0.0, 2.0], [2.0, 0.0, 3.0]]
    for keyframe, expected in zip(position_track.keyframes, expected_positions):
        assert keyframe.value == pytest.approx(expected)
    assert rotation_track.keyframes[1].value == pytest.approx(
        Rotation.from_rotvec([0.0, 0.0, np.pi / 2.0]).as_euler("ZXY", degrees=False).tolist()
    )

    scene.timeline_manager.set_time(2.0 / result.motion_asset.fps, apply_to_scene=True)
    assert transform.position.tolist() == pytest.approx([2.0, 0.0, 3.0])
    assert transform.rotation.tolist() == pytest.approx(rotation_track.keyframes[2].value)


def test_generate_humanoid_animation_offsets_all_baked_tracks(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(
        scene_animation,
        "build_smpl_skin_asset",
        lambda *args, **kwargs: _tiny_skin_asset(),
    )
    scene = Scene(scene_id="main")
    validated_intervals = []

    def validate(_scene, _object_id, start, end):
        validated_intervals.append((start, end))
        return {
            "status": "clear_at_samples",
            "samples": 3,
            "affected_samples": 0,
            "contacts": [],
        }

    result = create_humanoid_animation(
        scene=scene,
        project_dir=tmp_path,
        prompt="a person falls",
        motion_length=0.15,
        device=-1,
        output_name="delayed fall",
        start_time=4.5,
        playback_mode="baked_transforms",
        backend=_root_motion_backend,
        body_validator=validate,
    )

    assert result.start_time == pytest.approx(4.5)
    assert scene.timeline_manager.duration == pytest.approx(
        4.5 + result.duration,
    )
    body = scene.get_object(result.body_object_id)
    bone_ids = body.get_component("SkinnedMesh").bone_ids
    relevant = [
        track
        for track in scene.timeline_manager.clip.tracks.values()
        if track.object_id in {result.body_object_id, *bone_ids}
    ]
    assert relevant
    assert all(track.keyframes[0].time == pytest.approx(4.5) for track in relevant)
    assert validated_intervals == [(4.5, pytest.approx(4.5 + result.duration))]


def test_floor_penetration_is_lifted_and_revalidated_without_ignoring_obstacles(
    tmp_path, monkeypatch,
):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(
        scene_animation,
        "build_smpl_skin_asset",
        lambda *args, **kwargs: _tiny_skin_asset(),
    )
    scene = Scene(scene_id="main")
    reports = [
        {
            "status": "overlap_detected",
            "samples": 6,
            "affected_samples": 2,
            "contacts": [{
                "time": 1.1,
                "body_part": "torso",
                "obstacle_id": "floor",
                "obstacle": "floor",
                "kind": "conservative_convex_overlap",
                "penetration_m": 0.08,
            }],
        },
        {
            "status": "clear_at_samples",
            "samples": 6,
            "affected_samples": 0,
            "contacts": [],
            "limitations": [
                "Finite samples are not CCD.",
                "No automatic support-contact exemption or motion correction.",
            ],
        },
    ]

    def validate(*_args):
        return reports.pop(0)

    result = create_humanoid_animation(
        scene=scene,
        project_dir=tmp_path,
        prompt="a person falls",
        motion_length=0.15,
        device=-1,
        start_time=1.0,
        playback_mode="baked_transforms",
        backend=_root_motion_backend,
        body_validator=validate,
    )

    correction = result.trajectory_clearance["floor_support_correction"]
    assert correction["status"] == "applied_and_revalidated"
    assert correction["total_lift_m"] == pytest.approx(0.081)
    assert result.trajectory_clearance["limitations"] == [
        "Finite samples are not CCD.",
    ]
    track = scene.timeline_manager.clip.get_track(
        result.body_object_id, "Transform", "position",
    )
    assert track.keyframes[-1].value[1] == pytest.approx(0.081)


def test_generate_humanoid_animation_preserves_initial_vertical_root_translation(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="main")
    result = create_humanoid_animation(
        scene=scene,
        project_dir=tmp_path,
        prompt="a person walks upright",
        motion_length=0.15,
        device=-1,
        output_name="root height",
        position=[1.0, 0.25, 2.0],
        backend=_root_height_backend,
    )

    body_obj = scene.get_object(result.body_object_id)
    assert body_obj is not None
    transform = body_obj.get_component("Transform")
    position_track = scene.timeline_manager.clip.get_track(body_obj.id, "Transform", "position")
    assert position_track is not None

    assert position_track.keyframes[0].value == pytest.approx([1.0, 1.45, 2.0])
    assert position_track.keyframes[1].value == pytest.approx([1.5, 1.5, 2.25])
    assert transform.position.tolist() == pytest.approx([1.0, 1.45, 2.0])


def test_generate_humanoid_animation_builds_skinned_mesh_visual(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="main")

    result = create_humanoid_animation(
        scene=scene,
        project_dir=tmp_path,
        prompt="a person waves",
        motion_length=0.15,
        device=-1,
        output_name="visible wave",
        backend=_animated_backend_for_frames(3),
    )

    body_obj = scene.get_object(result.body_object_id)
    visual = body_obj.get_component("SkinnedMesh")
    assert visual is not None
    assert body_obj.get_component("Material") is not None
    assert body_obj.get_component("GenesisSMPLMesh") is None
    assert visual.bone_count == 2
    assert visual.get_mesh_data()["skin_indices"][0] == [0, 1, 0, 0]

    visual_track = scene.timeline_manager.clip.get_track(body_obj.id, "GenesisSMPLMesh", "current_frame")
    assert visual_track is None
    assert not scene.timeline_manager.mesh_cache.is_cached(body_obj.id)

    root_track = scene.timeline_manager.clip.get_track(visual.bone_ids[0], "Transform", "rotation")
    child_track = scene.timeline_manager.clip.get_track(visual.bone_ids[1], "Transform", "rotation")
    assert root_track is not None
    assert child_track is not None
    assert len(root_track.keyframes) == 3
    assert len(child_track.keyframes) == 3
    assert root_track.keyframes[2].value == pytest.approx([0.0, 0.0, 0.0])
    assert child_track.keyframes[2].value != pytest.approx([0.0, 0.0, 0.0])

    scene.timeline_manager.set_time(2.0 / result.motion_asset.fps, apply_to_scene=True)
    child_bone = scene.get_object(visual.bone_ids[1])
    assert child_bone.get_component("Transform").get_quaternion().tolist() != pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert result.visual_mesh_cache["mode"] == "skinned_mesh"
    assert result.visual_mesh_cache["cached_object_count"] == 0
    assert result.visual_mesh_cache["bone_count"] == 2
    assert result.visual_mesh_cache["skeleton_track_count"] == 2


def test_generate_humanoid_animation_broadcasts_final_objects_and_timeline(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="world")
    messages: list[dict] = []

    result = create_humanoid_animation(
        scene=scene,
        project_dir=tmp_path,
        prompt="a person waves",
        motion_length=0.15,
        device=-1,
        output_name="live wave",
        backend=_animated_backend_for_frames(3),
        broadcast=messages.append,
        serialize_object=lambda obj: obj.to_dict(),
    )

    batch = next(msg for msg in messages if msg["type"] == "objects_batch_created")
    assert batch["scene_id"] == "world"
    object_ids = {obj["id"] for obj in batch["objects"]}
    assert result.body_object_id in object_ids
    assert result.motion_object_id in object_ids

    body_payload = next(obj for obj in batch["objects"] if obj["id"] == result.body_object_id)
    skinned = next(comp for comp in body_payload["components"] if comp["name"] == "SkinnedMesh")
    assert skinned["bone_ids"]
    assert len(skinned["bone_ids"]) == 2

    timeline_state = next(msg for msg in messages if msg["type"] == "timeline_state")
    assert timeline_state["duration"] == pytest.approx(result.duration)
    assert timeline_state["fps"] == pytest.approx(result.motion_asset.fps)
    assert any(track["path"] == result.timeline_track["path"] for track in timeline_state["tracks"])


def test_generate_humanoid_animation_returns_radar_04_target_contract(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.adapter.scene_animation as scene_animation
    from wt_genesis.tools import make_generate_humanoid_animation_tool
    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scene_animation, "build_smpl_skin_asset", lambda *args, **kwargs: _tiny_skin_asset())
    scene = Scene(scene_id="main")
    tool = make_generate_humanoid_animation_tool(
        _FakeContext(scene), backend=_animated_backend
    )

    result = asyncio.run(
        tool.call(
            {
                "prompt": "a person waves",
                "motion_length": 0.2,
                "device": -1,
                "playback_mode": "baked_transforms",
            }
        )
    )

    body_obj = scene.get_object(result["body_object_id"])
    assert body_obj.get_component("SkinnedMesh") is not None
    assert body_obj.get_component("RadarScatterer") is None
    assert result["scene_binding"]["radar_target_contract"] == {
        "version": "studio_skinned_mesh_baked_transforms_v1",
        "target_object_id": body_obj.id,
        "component": "SkinnedMesh",
        "required_playback_mode": "baked_transforms",
        "site_budget_owner": "witwin.radar.target_site_budget",
    }

def _tiny_skin_asset():
    from wt_humanoid.adapter.smpl_skinned import SMPLSkinAsset

    return SMPLSkinAsset(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        skin_indices=np.array([[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int64),
        skin_weights=np.array([[0.75, 0.25, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        joint_positions=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        parent_indices=[-1, 0],
        inverse_bind_matrices=np.array(
            [
                np.eye(4, dtype=np.float32),
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )


def _animated_backend_for_frames(frame_count):
    def _backend(prompts, **kwargs):
        pose = np.zeros((1, frame_count, 24, 3), dtype=np.float32)
        for frame in range(frame_count):
            pose[0, frame, :, :] = float(frame)
        return {
            "smpl_pose": pose,
            "smpl_betas": np.zeros((1, frame_count, 10), dtype=np.float32),
            "smpl_trans": np.zeros((1, frame_count, 3), dtype=np.float32),
        }

    return _backend


def _root_motion_backend(prompts, **kwargs):
    pose = np.zeros((1, 3, 24, 3), dtype=np.float32)
    pose[0, :, 0, :] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2.0],
            [0.0, 0.0, np.pi],
        ],
        dtype=np.float32,
    )
    trans = np.array(
        [[[10.0, 0.0, 5.0], [10.5, 0.0, 5.0], [11.0, 0.0, 6.0]]],
        dtype=np.float32,
    )
    return {
        "smpl_pose": pose,
        "smpl_betas": np.zeros((1, 3, 10), dtype=np.float32),
        "smpl_trans": trans,
    }


def _root_height_backend(prompts, **kwargs):
    pose = np.zeros((1, 2, 24, 3), dtype=np.float32)
    trans = np.array(
        [[[10.0, 1.2, 5.0], [10.5, 1.25, 5.25]]],
        dtype=np.float32,
    )
    return {
        "smpl_pose": pose,
        "smpl_betas": np.zeros((1, 2, 10), dtype=np.float32),
        "smpl_trans": trans,
    }


@dataclass
class _ToolBucket:
    registered: list

    def register(self, tool):
        self.registered.append(tool)
        return tool


class _FakeContext:
    def __init__(self, scene, server=None):
        self.scene = scene
        self.api = _FakeApi(scene, server) if server is not None else None
        self.tools = _ToolBucket([])


class _FakeApi:
    def __init__(self, scene, server):
        self.scene = scene
        self.server = server


class _FakePersistence:
    def __init__(self):
        self.save_scene_calls: list[bool] = []

    def save_scene(self, force=False):
        self.save_scene_calls.append(bool(force))
        return True


class _FakeServer:
    def __init__(self, scene):
        self.scene = scene
        self.mesh_assets = None
        self.messages: list[dict] = []
        self.persistence = _FakePersistence()

    def _safe_broadcast(self, message):
        self.messages.append(message)

    def _serialize_object(self, obj, scene):
        return obj.to_dict()
