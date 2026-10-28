"""Kimodo's text encoder (LLM2Vec Llama-3-8B, bf16 on CPU) costs ~18 GB and
~40 s to load for every generated clip. In the R47 trial a clip was killed
while loading it next to other sessions' processes. Batch prompts repeat, so
embeddings are cached on disk and the encoder is loaded only on a miss.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


class _FakeEncoder:
    loads = 0

    def __init__(self):
        type(self).loads += 1
        self.seen = []

    def __call__(self, texts):
        import torch

        self.seen.extend(texts)
        rows = [[float(len(text)), 1.0, 2.0, 3.0] for text in texts]
        return torch.tensor(rows)[:, None], [1] * len(texts)


def test_embeddings_are_cached_and_the_encoder_loads_only_on_a_miss(tmp_path):
    from wt_genesis.adapter.kimodo_cached_generate import CachedTextEncoder

    _FakeEncoder.loads = 0
    first = CachedTextEncoder(tmp_path, factory=_FakeEncoder)
    feat, lengths = first(["a person walks.", "a person sits."])

    assert _FakeEncoder.loads == 1
    assert tuple(feat.shape) == (2, 1, 4) and lengths == [1, 1]
    assert feat[0, 0, 0].item() == len("a person walks.")
    assert len(list(tmp_path.glob("*.npy"))) == 2

    second = CachedTextEncoder(tmp_path, factory=_FakeEncoder)
    again, _ = second(["a person sits.", "a person walks."])
    assert _FakeEncoder.loads == 1
    assert again[1, 0, 0].item() == len("a person walks.")


def test_an_empty_text_needs_no_encoder(tmp_path):
    from wt_genesis.adapter.kimodo_cached_generate import CachedTextEncoder

    _FakeEncoder.loads = 0
    encoder = CachedTextEncoder(tmp_path, factory=_FakeEncoder, dim=4)
    feat, lengths = encoder([""])

    assert _FakeEncoder.loads == 0
    assert np.allclose(feat.numpy(), 0.0) and lengths == [1]


def _fake_runtime(tmp_path, monkeypatch):
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
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        output = command[command.index("--output") + 1]
        rotations = np.repeat(np.eye(3)[None, None], 2 * 77, axis=0).reshape(2, 77, 3, 3)
        np.savez(f"{output}.npz", local_rot_mats=rotations, root_positions=np.zeros((2, 3)))

    monkeypatch.setattr(backend.subprocess, "run", run)
    return backend, commands


def test_the_backend_generates_through_the_cached_runner(tmp_path, monkeypatch):
    backend, commands = _fake_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("WITWIN_KIMODO_TEXT_CACHE", str(tmp_path / "cache"))

    backend.generate_smpl_motion(["a person walks"], motion_length=2.0, seed=3,
                                 constraints=[{"type": "root2d", "frame_indices": [0],
                                               "smooth_root_2d": [[0.0, 0.0]]}])

    command = commands[-1]
    assert command[1].endswith("kimodo_cached_generate.py")
    assert command[command.index("--text-cache") + 1] == str(tmp_path / "cache")
    for flag in ("--model", "--duration", "--seed", "--output", "--constraints"):
        assert flag in command


def test_the_text_cache_can_be_switched_off(tmp_path, monkeypatch):
    backend, commands = _fake_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("WITWIN_KIMODO_TEXT_CACHE", "off")

    backend.generate_smpl_motion(["a person waves"], motion_length=2.0, seed=3)

    assert commands[-1][1:3] == ["-m", "kimodo.scripts.generate"]


def test_runner_arguments_match_the_kimodo_cli(tmp_path):
    from wt_genesis.adapter.kimodo_cached_generate import parse_args

    constraints = tmp_path / "c.json"
    constraints.write_text(json.dumps([]), encoding="utf-8")
    args = parse_args(["a person walks", "--model", "Kimodo-SOMA-RP-v1.1", "--duration", "4.5",
                       "--num_samples", "1", "--seed", "7", "--output", str(tmp_path / "m"),
                       "--constraints", str(constraints), "--text-cache", str(tmp_path)])

    assert args.prompt == "a person walks" and args.duration == "4.5" and args.seed == 7
    assert args.text_cache == str(tmp_path)


def test_a_ready_runtime_is_probed_once_per_process(tmp_path, monkeypatch):
    # R47 person trial: the import probe (import kimodo, torch) ran before every
    # clip and hit its 30 s timeout under memory pressure, failing a sample.
    import wt_genesis.adapter.kimodo_backend as backend

    backend, _commands = _fake_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "_READY_RUNTIMES", set(), raising=False)
    probes = []
    monkeypatch.setattr(backend, "_probe_runtime", lambda python, root: probes.append(1) or {
        "importable": True, "cuda_available": True, "detail": "ok"})

    for _ in range(3):
        backend.generate_smpl_motion(["a person waves"], motion_length=2.0, seed=3)

    assert len(probes) == 1


def test_the_import_probe_allows_a_loaded_machine(monkeypatch, tmp_path):
    import wt_genesis.adapter.kimodo_backend as backend

    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return type("R", (), {"stdout": "1\n"})()

    monkeypatch.setattr(backend.subprocess, "run", run)
    backend._probe_runtime(tmp_path / "python.exe", tmp_path)

    assert seen["timeout"] >= 120


def test_the_runner_imports_resolve_in_the_kimodo_runtime():
    # The first real run failed: instantiate_from_dict is not exported by
    # kimodo.model (it lives in kimodo.model.loading). Fakes cannot see that.
    import os
    import subprocess
    from pathlib import Path

    import wt_genesis.adapter.kimodo_backend as backend

    python, root = backend._kimodo_python(), backend._kimodo_root()
    if not python.is_file() or not (root / "kimodo").is_dir() or "witwin-kimodo" not in str(python):
        pytest.skip("isolated Kimodo runtime not configured")
    adapter = Path(backend.__file__).parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(adapter)]))
    result = subprocess.run(
        [str(python), "-c", "import kimodo_cached_generate as m; m.kimodo_imports()"],
        capture_output=True, text=True, env=env, timeout=300)

    assert result.returncode == 0, result.stderr[-2000:]
