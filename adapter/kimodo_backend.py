"""Isolated process adapter for NVIDIA Kimodo text-to-motion generation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .kimodo_motion import load_kimodo_motion


DEFAULT_MODEL = "Kimodo-SOMA-RP-v1.1"


def configured_backend():
    """Return the configured optional backend, preserving MDM Tiny by default."""
    name = os.environ.get("WITWIN_GENESIS_BACKEND", "mdm_tiny").strip().lower()
    if name in {"", "mdm", "mdm_tiny", "genesis"}:
        return None
    if name == "kimodo":
        return generate_smpl_motion
    raise ValueError(f"unsupported WITWIN_GENESIS_BACKEND: {name}")


def inspect_kimodo_runtime() -> dict[str, Any]:
    root = _kimodo_root()
    python = _kimodo_python()
    token_paths = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token",
        Path.home() / ".huggingface" / "token",
    )
    has_token = bool(os.environ.get("HF_TOKEN")) or any(path.is_file() for path in token_paths)
    issues = []
    if not python.is_file():
        issues.append(f"Kimodo Python executable does not exist: {python}")
    if not (root / "kimodo" / "scripts" / "generate.py").is_file():
        issues.append(f"Kimodo source checkout is missing: {root}")
    import_probe = _probe_runtime(python, root) if python.is_file() else {
        "importable": False,
        "cuda_available": False,
        "detail": "Python executable is missing",
    }
    if not import_probe["importable"]:
        issues.append(f"Kimodo cannot import in its isolated runtime: {import_probe['detail']}")
    elif not import_probe["cuda_available"]:
        issues.append("CUDA is unavailable in the isolated Kimodo runtime")
    if not has_token:
        issues.append("Hugging Face authentication is missing for the gated Llama text encoder")
    return {
        "ready": not issues,
        "model": os.environ.get("WITWIN_KIMODO_MODEL", DEFAULT_MODEL),
        "python": str(python),
        "source_root": str(root),
        "text_encoder_device": os.environ.get("TEXT_ENCODER_DEVICE", "cpu"),
        "huggingface_authenticated": has_token,
        "import_probe": import_probe,
        "issues": issues,
    }


def generate_smpl_motion(
    prompts,
    *,
    motion_length: float,
    seed: int,
    device: int = 0,
    guidance_param: float = 2.5,
    smplify_iters: int = 20,
) -> dict[str, Any]:
    """Run Kimodo out-of-process and adapt its motion to WT's SMPL contract."""
    del guidance_param, smplify_iters
    if len(prompts) != 1 or not str(prompts[0]).strip():
        raise ValueError("Kimodo backend requires exactly one non-empty prompt")
    status = inspect_kimodo_runtime()
    if not status["ready"]:
        raise RuntimeError("Kimodo runtime is not ready: " + "; ".join(status["issues"]))

    model = str(status["model"])
    with tempfile.TemporaryDirectory(prefix="witwin-kimodo-") as work:
        output_stem = Path(work) / "motion"
        command = [
            str(status["python"]), "-m", "kimodo.scripts.generate", str(prompts[0]),
            "--model", model,
            "--duration", str(float(motion_length)),
            "--num_samples", "1",
            "--seed", str(int(seed)),
            "--output", str(output_stem),
        ]
        env = os.environ.copy()
        env.setdefault("TEXT_ENCODER_DEVICE", "cpu")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(status["source_root"]), env.get("PYTHONPATH")) if value
        )
        if int(device) < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif int(device) > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(int(device))
        subprocess.run(command, check=True, env=env, timeout=900)

        generated = (
            Path(f"{output_stem}_amass.npz")
            if "smplx" in model.lower()
            else Path(f"{output_stem}.npz")
        )
        if not generated.is_file():
            raise RuntimeError(f"Kimodo completed without producing expected output: {generated.name}")
        return load_kimodo_motion(generated, target_fps=20.0)


def _kimodo_root() -> Path:
    configured = os.environ.get("WITWIN_KIMODO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "repos" / "kimodo"


def _kimodo_python() -> Path:
    configured = os.environ.get("WITWIN_KIMODO_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    sibling = Path(sys.executable).resolve().parent.parent / "witwin-kimodo" / "python.exe"
    return sibling if sibling.is_file() else Path(sys.executable).resolve()


def _probe_runtime(python: Path, root: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root), env.get("PYTHONPATH")) if value
    )
    try:
        result = subprocess.run(
            [str(python), "-c", "import kimodo, torch; print(int(torch.cuda.is_available()))"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"importable": False, "cuda_available": False, "detail": str(exc)}
    return {
        "importable": True,
        "cuda_available": result.stdout.strip().splitlines()[-1:] == ["1"],
        "detail": "ok",
    }
