"""Motion export behavior for wt_genesis."""
from __future__ import annotations

import numpy as np


def _fake_backend(prompts, **kwargs):
    assert prompts == ["walk forward"]
    assert kwargs["motion_length"] == 1.5
    pose = np.arange(1 * 3 * 24 * 3, dtype=np.float32).reshape(1, 3, 24, 3)
    betas = np.arange(1 * 3 * 10, dtype=np.float32).reshape(1, 3, 10)
    trans = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [2.0, 0.0, 1.0]]], dtype=np.float32)
    return {"smpl_pose": pose, "smpl_betas": betas, "smpl_trans": trans}


def test_generate_motion_asset_exports_humanoid_compatible_npz(tmp_path):
    from wt_genesis.adapter.motion_export import generate_motion_asset

    result = generate_motion_asset(
        tmp_path,
        prompt="walk forward",
        motion_length=1.5,
        seed=7,
        device=-1,
        guidance_param=2.0,
        smplify_iters=3,
        output_name="Walk Forward!",
        backend=_fake_backend,
    )

    assert result.ok is True
    assert result.frame_count == 3
    assert result.fps == 20.0
    assert result.project_relative_path == "assets/motions/walk_forward.npz"

    with np.load(result.motion_path, allow_pickle=False) as data:
        assert set(data.files) >= {"smpl_pose", "shape", "root", "fps", "prompt"}
        assert data["smpl_pose"].shape == (3, 24, 3)
        assert data["shape"].shape == (10,)
        assert data["root"].shape == (3, 3)
        assert float(data["fps"]) == 20.0
        assert str(data["prompt"]) == "walk forward"
        assert data["shape"].tolist() == np.arange(10, dtype=np.float32).tolist()


def test_output_name_is_sanitized_and_does_not_escape_assets_dir(tmp_path):
    from wt_genesis.adapter.motion_export import generate_motion_asset

    result = generate_motion_asset(
        tmp_path,
        prompt="turn left",
        output_name="../Bad Name??",
        backend=_fake_backend_for_prompt("turn left"),
    )

    assert result.project_relative_path == "assets/motions/bad_name.npz"
    assert result.motion_path == str(tmp_path / "assets" / "motions" / "bad_name.npz")


def _fake_backend_for_prompt(prompt):
    def _backend(prompts, **kwargs):
        assert prompts == [prompt]
        return {
            "smpl_pose": np.zeros((1, 2, 24, 3), dtype=np.float32),
            "smpl_betas": np.zeros((1, 2, 10), dtype=np.float32),
            "smpl_trans": np.zeros((1, 2, 3), dtype=np.float32),
        }

    return _backend
