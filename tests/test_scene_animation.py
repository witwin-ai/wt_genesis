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
    from wt_genesis.tools import make_generate_humanoid_animation_tool

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
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
    assert motion.motion_path == result["motion_path"]
    assert motion.pose_key == "smpl_pose"
    assert motion.betas_key == "shape"
    assert motion.trans_key == "root"
    assert motion.target_body["object_id"] == body_obj.id
    assert motion.target_body["component_type"] == "SMPLBody"
    assert motion.frame_count == 4
    assert motion.loop == "on"

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


def test_generate_humanoid_animation_keys_root_motion_on_body_transform(tmp_path):
    from scipy.spatial.transform import Rotation
    from witwin_server.core.scene import Scene
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

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


def test_generate_humanoid_animation_builds_visible_mesh_cache(tmp_path, monkeypatch):
    from witwin_server.core.scene import Scene
    import wt_genesis.components.smpl_mesh as smpl_mesh
    from wt_genesis.adapter.scene_animation import create_humanoid_animation

    bake_modes = []

    def fake_bake_sequence(*args, **kwargs):
        bake_modes.append(kwargs.get("root_motion_mode", args[6] if len(args) > 6 else None))
        vertices = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        return vertices, faces

    monkeypatch.setattr(smpl_mesh, "bake_smpl_sequence", fake_bake_sequence)
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
    visual = body_obj.get_component("GenesisSMPLMesh")
    assert visual is not None
    assert body_obj.get_component("Material") is not None
    assert visual.motion_path == result.motion_asset.motion_path
    assert visual.frame_count == 3

    visual_track = scene.timeline_manager.clip.get_track(body_obj.id, "GenesisSMPLMesh", "current_frame")
    assert visual_track is not None
    assert [keyframe.value for keyframe in visual_track.keyframes] == [0, 1, 2]
    assert bake_modes
    assert set(bake_modes) == {"transform"}
    assert scene.timeline_manager.mesh_cache.is_cached(body_obj.id)

    mesh = scene.timeline_manager.get_mesh_at_time(body_obj.id, 2.0 / result.motion_asset.fps)
    assert mesh is not None
    assert mesh["faces"] == [[0, 1, 2]]
    assert mesh["vertices"][0] == pytest.approx([2.0, 0.0, 0.0])
    assert result.visual_mesh_cache["cached_object_count"] == 1


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


@dataclass
class _ToolBucket:
    registered: list

    def register(self, tool):
        self.registered.append(tool)
        return tool


class _FakeContext:
    def __init__(self, scene):
        self.scene = scene
        self.tools = _ToolBucket([])
