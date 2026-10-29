"""Isolated process adapter for NVIDIA Kimodo text-to-motion generation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from .kimodo_motion import SMPLRestSkeleton, ground_smpl_motion, load_kimodo_motion


DEFAULT_MODEL = "Kimodo-SOMA-RP-v1.1"
#: The SMPL clips a Kimodo run is resampled to.
KIMODO_FPS = 20.0
#: (python, source root) pairs whose import probe already succeeded in this
#: process; a batch probes once, not before every clip.
_READY_RUNTIMES: set[tuple[str, str]] = set()


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
    key = (str(python), str(root))
    if key in _READY_RUNTIMES and python.is_file():
        import_probe = {"importable": True, "cuda_available": True,
                        "detail": "ok (probed earlier in this process)"}
    elif python.is_file():
        import_probe = _probe_runtime(python, root)
        if import_probe["importable"] and import_probe["cuda_available"]:
            _READY_RUNTIMES.add(key)
    else:
        import_probe = {
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


def _segment_durations(prompt: str, total_s: float) -> str:
    """Kimodo --duration for a prompt whose full stops start new segments.

    Kimodo's CLI generates every '.'-separated sentence as its own segment of
    --duration seconds, so a two-sentence prompt doubled the clip length. The
    requested motion_length is the total; split it evenly across sentences.
    """
    count = sum(1 for text in prompt.split(".") if text.strip())
    if count <= 1:
        return str(float(total_s))
    part = round(total_s / count, 3)
    parts = [part] * (count - 1) + [round(total_s - part * (count - 1), 3)]
    return " ".join(str(float(value)) for value in parts)


def generate_smpl_motion(
    prompts,
    *,
    motion_length: float,
    seed: int,
    device: int = 0,
    guidance_param: float = 2.5,
    smplify_iters: int = 20,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Kimodo out-of-process and adapt its motion to WT's SMPL contract.

    ``constraints`` is a Kimodo constraint list (``root2d`` paths, headings,
    keyframes) as its CLI loads it from ``--constraints``.
    """
    del guidance_param, smplify_iters
    if len(prompts) != 1 or not str(prompts[0]).strip():
        raise ValueError("Kimodo backend requires exactly one non-empty prompt")
    if constraints and sum(1 for text in str(prompts[0]).split(".") if text.strip()) > 1:
        # Constraint frames index one clip; Kimodo's per-sentence segments
        # (and their transitions) would no longer line up with them.
        raise ValueError("a constrained Kimodo prompt must be a single sentence")
    status = inspect_kimodo_runtime()
    if not status["ready"]:
        raise RuntimeError("Kimodo runtime is not ready: " + "; ".join(status["issues"]))

    model = str(status["model"])
    with tempfile.TemporaryDirectory(prefix="witwin-kimodo-") as work:
        output_stem = Path(work) / "motion"
        text_cache = None if "smplx" in model.lower() else _text_cache_dir()
        entry = (["-m", "kimodo.scripts.generate"] if text_cache is None
                 else [str(Path(__file__).with_name("kimodo_cached_generate.py"))])
        command = [
            str(status["python"]), *entry, str(prompts[0]),
            "--model", model,
            "--duration", _segment_durations(str(prompts[0]), float(motion_length)),
            "--num_samples", "1",
            "--seed", str(int(seed)),
            "--output", str(output_stem),
        ]
        if constraints:
            constraint_path = Path(work) / "constraints.json"
            constraint_path.write_text(json.dumps(constraints), encoding="utf-8")
            command += ["--constraints", str(constraint_path)]
        if text_cache is not None:
            command += ["--text-cache", str(text_cache)]
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
        motion = load_kimodo_motion(generated, target_fps=KIMODO_FPS)
    skeleton = smpl_rest_skeleton()
    if skeleton is not None:
        motion, lift = ground_smpl_motion(motion, skeleton)
        motion["grounding_lift_m"] = lift
    skin = smpl_skin()
    if skin is not None:
        # SOMA's root path on SMPL's legs slides a planted foot 6-15 cm.
        # A swinging toe skimming through the floor is raised by its leg first.
        from .foot_lock import (lift_dipping_feet, lock_planted_feet, set_down_hovering_feet,
                                settle_first_frame)

        pose, _trans, first = settle_first_frame(motion["smpl_pose"][0], motion["smpl_trans"][0])
        pose, lifted = lift_dipping_feet(pose, motion["smpl_trans"][0], skin, KIMODO_FPS)
        pose, set_down = set_down_hovering_feet(pose, motion["smpl_trans"][0], skin, KIMODO_FPS)
        pose, trans, report = lock_planted_feet(pose, motion["smpl_trans"][0], skin, KIMODO_FPS)
        motion["smpl_pose"] = pose[None].astype(np.float32)
        motion["smpl_trans"] = trans[None].astype(np.float32)
        motion["foot_lock"] = {**report, "lifted": lifted, "set_down": set_down,
                               "first_frame": first}
    return motion


#: SMPL joints of the ankles and feet (toes).
SMPL_FOOT_JOINTS = (7, 8, 10, 11)


def _smpl_skin_asset():
    from witwin.genesis.mdm_tiny_runtime import DEFAULT_MODEL_ROOT
    from wt_humanoid.adapter.smpl_skinned import build_smpl_skin_asset

    root = os.environ.get("WITWIN_SMPL_MODEL_ROOT") or str(DEFAULT_MODEL_ROOT)
    return build_smpl_skin_asset(root, gender="neutral", betas=[0.0] * 10)


def smpl_skin():
    """The neutral SMPL body for skinning Genesis motions (zero betas), or
    None when the licensed model files are missing."""
    from .kimodo_poses import SMPLSkin

    try:
        asset = _smpl_skin_asset()
    except Exception:  # noqa: BLE001 - grounding on the skin is skipped, not the clip
        return None
    return SMPLSkin(vertices=np.asarray(asset.vertices, dtype=np.float64),
                    joints=np.asarray(asset.joint_positions, dtype=np.float64),
                    parents=list(asset.parent_indices),
                    skin_indices=np.asarray(asset.skin_indices, dtype=np.int64),
                    skin_weights=np.asarray(asset.skin_weights, dtype=np.float64))


def smpl_rest_skeleton() -> SMPLRestSkeleton | None:
    """The neutral SMPL rest skeleton Genesis bodies use (zero betas, as
    Kimodo clips have), or None when the licensed model files are missing."""
    try:
        from witwin.genesis.mdm_tiny_runtime import DEFAULT_MODEL_ROOT
        from wt_humanoid.adapter.smpl_skinned import build_smpl_skin_asset

        root = os.environ.get("WITWIN_SMPL_MODEL_ROOT") or str(DEFAULT_MODEL_ROOT)
        asset = build_smpl_skin_asset(root, gender="neutral", betas=[0.0] * 10)
    except Exception:  # noqa: BLE001 - grounding is skipped, not the clip
        return None
    joints = np.asarray(asset.joint_positions, dtype=np.float64)
    vertices = np.asarray(asset.vertices, dtype=np.float64)
    feet = float(joints[list(SMPL_FOOT_JOINTS), 1].min())
    return SMPLRestSkeleton(joints=joints, parents=list(asset.parent_indices),
                            foot_joints=SMPL_FOOT_JOINTS,
                            sole_below_feet_m=feet - float(vertices[:, 1].min()))


def _text_cache_dir() -> Path | None:
    """Where prompt embeddings are cached (``WITWIN_KIMODO_TEXT_CACHE``; ``off``
    runs Kimodo's own CLI, which loads its text encoder for every clip)."""
    configured = os.environ.get("WITWIN_KIMODO_TEXT_CACHE", "").strip()
    if configured.lower() in {"off", "0", "false", "none"}:
        return None
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "witwin" / "kimodo_text_embeddings"


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
            # Importing torch takes ~10 s, and minutes on a loaded machine.
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"importable": False, "cuda_available": False, "detail": str(exc)}
    return {
        "importable": True,
        "cuda_available": result.stdout.strip().splitlines()[-1:] == ["1"],
        "detail": "ok",
    }
