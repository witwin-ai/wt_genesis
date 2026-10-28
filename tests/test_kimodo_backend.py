from __future__ import annotations

import numpy as np


def test_kimodo_backend_runs_isolated_cli_and_returns_genesis_contract(tmp_path, monkeypatch):
    import wt_genesis.adapter.kimodo_backend as backend

    fake_python = tmp_path / "python.exe"
    fake_python.touch()
    source = tmp_path / "kimodo"
    (source / "kimodo" / "scripts").mkdir(parents=True)
    (source / "kimodo" / "scripts" / "generate.py").touch()
    token_home = tmp_path / "hf"
    token_home.mkdir()
    (token_home / "token").write_text("not-a-real-token", encoding="utf-8")
    monkeypatch.setenv("WITWIN_KIMODO_PYTHON", str(fake_python))
    monkeypatch.setenv("WITWIN_KIMODO_ROOT", str(source))
    monkeypatch.setenv("HF_HOME", str(token_home))
    monkeypatch.setattr(
        backend, "_probe_runtime",
        lambda python, root: {"importable": True, "cuda_available": True, "detail": "ok"},
    )
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        output = command[command.index("--output") + 1]
        rotations = np.repeat(np.eye(3)[None, None], 2 * 77, axis=0).reshape(2, 77, 3, 3)
        np.savez(f"{output}.npz", local_rot_mats=rotations, root_positions=np.zeros((2, 3)))

    monkeypatch.setattr(backend.subprocess, "run", run)

    result = backend.generate_smpl_motion(
        ["A person loses balance and falls."], motion_length=4, seed=41, device=0
    )

    assert result["smpl_pose"].shape == (1, 2, 24, 3)
    assert captured["command"][1].endswith("kimodo_cached_generate.py")
    assert "Kimodo-SOMA-RP-v1.1" in captured["command"]
    assert captured["kwargs"]["env"]["TEXT_ENCODER_DEVICE"] == "cpu"


def test_kimodo_runtime_reports_missing_auth_separately(tmp_path, monkeypatch):
    import wt_genesis.adapter.kimodo_backend as backend

    python = tmp_path / "python.exe"
    python.touch()
    source = tmp_path / "kimodo"
    (source / "kimodo" / "scripts").mkdir(parents=True)
    (source / "kimodo" / "scripts" / "generate.py").touch()
    monkeypatch.setenv("WITWIN_KIMODO_PYTHON", str(python))
    monkeypatch.setenv("WITWIN_KIMODO_ROOT", str(source))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(backend.Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
    monkeypatch.setattr(
        backend, "_probe_runtime",
        lambda python, root: {"importable": True, "cuda_available": True, "detail": "ok"},
    )

    status = backend.inspect_kimodo_runtime()

    assert status["ready"] is False
    assert status["huggingface_authenticated"] is False
    assert "authentication is missing" in status["issues"][0]


def test_backend_selection_is_opt_in(monkeypatch):
    import wt_genesis.adapter.kimodo_backend as backend

    monkeypatch.delenv("WITWIN_GENESIS_BACKEND", raising=False)
    assert backend.configured_backend() is None
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "kimodo")
    assert backend.configured_backend() is backend.generate_smpl_motion
