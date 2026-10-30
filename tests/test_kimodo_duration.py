"""Kimodo splits a prompt at full stops and gives each sentence --duration.

R47 acceptance: a two-sentence 9 s request became an 18 s Kimodo clip whose
second segment kept walking into a wall. The requested motion_length is the
total; it must be divided across Kimodo's sentence segments.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def kimodo(tmp_path, monkeypatch):
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
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        output = command[command.index("--output") + 1]
        rotations = np.repeat(np.eye(3)[None, None], 2 * 77, axis=0).reshape(2, 77, 3, 3)
        np.savez(f"{output}.npz", local_rot_mats=rotations, root_positions=np.zeros((2, 3)))

    monkeypatch.setattr(backend.subprocess, "run", run)

    def durations(prompt, motion_length):
        backend.generate_smpl_motion([prompt], motion_length=motion_length, seed=10, device=0)
        command = commands[-1]
        return command[command.index("--duration") + 1]

    return durations


def test_single_sentence_keeps_the_requested_duration(kimodo):
    assert kimodo("A person stands and waves once.", 9) == "9.0"


def test_multi_sentence_prompt_divides_the_total_duration(kimodo):
    value = kimodo(
        "A person takes a few slow steps, stops and waves one hand. No stairs or furniture contact.", 9)
    assert value == "4.5 4.5"
    assert sum(float(part) for part in value.split()) == pytest.approx(9.0)


def test_three_sentences_sum_to_the_total(kimodo):
    value = kimodo("A person walks forward. The person stops. The person waves.", 10)
    parts = [float(part) for part in value.split()]
    assert len(parts) == 3
    assert sum(parts) == pytest.approx(10.0, abs=1e-6)
