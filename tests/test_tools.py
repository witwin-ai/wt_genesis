"""Agent tool registration for wt_genesis."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import numpy as np


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


def test_register_adds_namespaced_tool_to_context():
    import wt_genesis.tools as genesis_tools

    ctx = _FakeContext()
    genesis_tools.register(ctx)

    assert [tool.name for tool in ctx.tools.registered] == [
        "generate_smpl_motion",
        "generate_humanoid_animation",
    ]


@dataclass
class _ToolBucket:
    registered: list

    def register(self, tool):
        self.registered.append(tool)
        return tool


class _FakeContext:
    def __init__(self):
        self.tools = _ToolBucket([])
