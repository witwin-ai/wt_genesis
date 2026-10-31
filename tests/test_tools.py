"""Agent tool registration for wt_genesis."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import numpy as np
from types import SimpleNamespace


def _backend(prompts, **kwargs):
    assert prompts == ["a person waves"]
    return {
        "smpl_pose": np.zeros((1, 4, 24, 3), dtype=np.float32),
        "smpl_betas": np.ones((1, 4, 10), dtype=np.float32),
        "smpl_trans": np.zeros((1, 4, 3), dtype=np.float32),
    }


def test_generate_smpl_motion_tool_writes_project_motion_asset(tmp_path, monkeypatch):
    from wt_genesis.tools import make_generate_smpl_motion_tool

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    tool = make_generate_smpl_motion_tool(_FakeContext(), backend=_backend)

    result = asyncio.run(
        tool.call(
            {
                "prompt": "a person waves",
                "motion_length": 2.0,
                "device": -1,
                "output_name": "wave demo",
            }
        )
    )

    assert result["ok"] is True
    assert result["frame_count"] == 4
    assert result["fps"] == 20.0
    assert result["project_relative_path"] == "assets/motions/wave_demo.npz"
    assert os.path.exists(result["motion_path"])
    assert result["pose_key"] == "smpl_pose"
    assert result["betas_key"] == "shape"
    assert result["trans_key"] == "root"


def test_generate_smpl_motion_tool_defaults_to_gpu_device_zero(tmp_path, monkeypatch):
    from wt_genesis.tools import make_generate_smpl_motion_tool

    captured = {}

    def backend(prompts, **kwargs):
        captured.update(kwargs)
        return _backend(prompts, **kwargs)

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    tool = make_generate_smpl_motion_tool(_FakeContext(), backend=backend)

    asyncio.run(tool.call({"prompt": "a person waves"}))

    assert captured["device"] == 0


def test_humanoid_collision_bridge_delegates_relocation_to_motion_matching(monkeypatch):
    import wt_genesis.tools as genesis_tools

    captured = {}

    def validate(scene, args, project_root):
        captured.update(scene=scene, args=args, project_root=project_root)
        return {"status": "clear_at_samples", "automatic_relocation": {"status": "applied"}}

    module = SimpleNamespace(body_collision_report_with_relocation=validate)
    monkeypatch.setattr(genesis_tools.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(genesis_tools, "default_project_dir", lambda: "project-root")
    scene = object()

    result = genesis_tools._motion_matching_body_validation(scene, "human", 1.0, 3.0)

    assert result["automatic_relocation"]["status"] == "applied"
    assert captured["scene"] is scene
    assert captured["project_root"] == "project-root"
    assert captured["args"]["object_id"] == "human"
    assert captured["args"]["start"] == 1.0
    assert captured["args"]["end"] == 3.0


def test_check_genesis_runtime_tool_returns_structured_status(monkeypatch):
    import wt_genesis.tools as genesis_tools

    class Status:
        def to_dict(self):
            return {"ready": False, "missing_smpl_files": ["SMPL_NEUTRAL.pkl"]}

    monkeypatch.setattr(genesis_tools, "inspect_runtime", lambda: Status())
    monkeypatch.setattr(
        genesis_tools,
        "inspect_kimodo_runtime",
        lambda: {"ready": False, "issues": ["missing auth"]},
    )
    monkeypatch.setattr(
        genesis_tools,
        "_inspect_radar_pipeline_runtime",
        lambda: {"ready": False, "issues": ["CUDA is unavailable"]},
    )
    tool = genesis_tools.make_check_genesis_runtime_tool(_FakeContext())

    result = asyncio.run(tool.call({}))

    assert result["ready"] is False
    assert result["missing_smpl_files"] == ["SMPL_NEUTRAL.pkl"]
    assert result["pipeline_ready"] is False
    assert result["readiness_scope"] == "packages_models_native_imports_and_cuda_only"
    assert result["human_radar_validation"] == "not_run"
    assert result["radar_runtime"]["issues"] == ["CUDA is unavailable"]
    assert result["active_backend"] == "mdm_tiny"


def test_generate_tool_uses_configured_kimodo_backend(tmp_path, monkeypatch):
    import wt_genesis.tools as genesis_tools

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "kimodo")
    monkeypatch.setattr(genesis_tools, "configured_backend", lambda: _backend)
    tool = genesis_tools.make_generate_smpl_motion_tool(_FakeContext())

    result = asyncio.run(tool.call({"prompt": "a person waves"}))

    assert result["ok"] is True
    assert result["frame_count"] == 4


def test_radar_pipeline_doctor_separates_versions_native_imports_and_cuda(monkeypatch):
    import torch
    import wt_genesis.tools as genesis_tools

    versions = {
        "witwin-radar": "0.2.9",
        "witwin-channel": "0.5.0",
        "rayd": "0.8.0",
    }
    monkeypatch.setattr(genesis_tools.metadata, "version", versions.__getitem__)

    def import_module(name):
        if name == "rayd.torch":
            raise OSError("native DLL missing")
        return object()

    monkeypatch.setattr(genesis_tools.importlib, "import_module", import_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    result = genesis_tools._inspect_radar_pipeline_runtime()

    assert result["ready"] is False
    assert result["versions"]["witwin-radar"] == "0.2.9"
    assert "witwin-radar 0.2.9 is incompatible" in result["issues"][0]
    assert result["issues"][-1] == "CUDA is unavailable for Radar sequence solving"
    assert result["import_errors"] == {"rayd.torch": "OSError: native DLL missing"}

def test_register_adds_namespaced_tool_to_context():
    import wt_genesis.tools as genesis_tools

    ctx = _FakeContext()
    genesis_tools.register(ctx)

    assert [tool.name for tool in ctx.tools.registered] == [
        "check_genesis_runtime",
        "generate_smpl_motion",
        "generate_humanoid_animation",
        "generate_humanoid_route_animation",
    ]
    animation = next(tool for tool in ctx.tools.registered
                     if tool.name == "generate_humanoid_animation")
    assert "pack:witwin.scene" in animation.tags
    assert "word-intent:person" in animation.tags
    assert "hint-explicit-intent:someone walking" in animation.tags
    assert "hint-requires-pack" in animation.tags


@dataclass
class _ToolBucket:
    registered: list

    def register(self, tool):
        self.registered.append(tool)
        return tool


class _FakeContext:
    def __init__(self):
        self.tools = _ToolBucket([])
