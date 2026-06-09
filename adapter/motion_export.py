"""Export MDM text-to-SMPL output as wt_humanoid-compatible motion assets."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import mdm_tiny_runtime


Backend = Callable[..., dict[str, Any]]
FPS = 20.0


@dataclass(frozen=True)
class MotionExportResult:
    ok: bool
    motion_path: str
    project_relative_path: str
    frame_count: int
    fps: float
    prompt: str
    pose_key: str = "smpl_pose"
    betas_key: str = "shape"
    trans_key: str = "root"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "motion_path": self.motion_path,
            "project_relative_path": self.project_relative_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "prompt": self.prompt,
            "pose_key": self.pose_key,
            "betas_key": self.betas_key,
            "trans_key": self.trans_key,
        }


def generate_motion_asset(
    project_dir: str | Path,
    *,
    prompt: str,
    motion_length: float = 6.0,
    seed: int = 10,
    device: int = 0,
    guidance_param: float = 2.5,
    smplify_iters: int = 20,
    output_name: str | None = None,
    backend: Backend | None = None,
) -> MotionExportResult:
    """Generate and save a visible Project motion asset."""
    prompt_value = str(prompt).strip()
    if not prompt_value:
        raise ValueError("prompt must be a non-empty string")

    root = Path(project_dir).resolve()
    output_dir = root / "assets" / "motions"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(output_name or prompt_value)
    output_path = _unique_output_path(output_dir, stem)

    generator = backend or mdm_tiny_runtime.generate_smpl_motion
    motion = generator(
        [prompt_value],
        motion_length=float(motion_length),
        seed=int(seed),
        device=int(device),
        guidance_param=float(guidance_param),
        smplify_iters=int(smplify_iters),
    )

    pose = _normalize_pose(motion.get("smpl_pose"))
    frame_count = int(pose.shape[0])
    shape = _normalize_shape(motion.get("smpl_betas"), frame_count)
    root_translation = _normalize_translation(motion.get("smpl_trans"), frame_count)

    np.savez(
        output_path,
        smpl_pose=pose,
        shape=shape,
        root=root_translation,
        fps=np.array(FPS, dtype=np.float32),
        prompt=np.array(prompt_value),
    )

    relative = output_path.relative_to(root).as_posix()
    return MotionExportResult(
        ok=True,
        motion_path=str(output_path),
        project_relative_path=relative,
        frame_count=frame_count,
        fps=FPS,
        prompt=prompt_value,
    )


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return stem or "smpl_motion"


def _unique_output_path(output_dir: Path, stem: str) -> Path:
    candidate = output_dir / f"{stem}.npz"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = output_dir / f"{stem}_{index}.npz"
        if not candidate.exists():
            return candidate
        index += 1


def _as_numpy(value: Any, label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"MDM output is missing {label}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_pose(value: Any) -> np.ndarray:
    pose = _as_numpy(value, "smpl_pose")
    if pose.ndim == 4 and pose.shape[-2:] == (24, 3):
        pose = pose[0]
    elif pose.ndim == 3 and pose.shape[-2:] == (24, 3):
        pass
    elif pose.ndim == 3 and pose.shape[-1] == 72:
        pose = pose[0].reshape(pose.shape[1], 24, 3)
    elif pose.ndim == 2 and pose.shape[-1] == 72:
        pose = pose.reshape(pose.shape[0], 24, 3)
    else:
        raise ValueError("smpl_pose must have shape (sample, frames, 24, 3), (frames, 24, 3), or frame-major 72D poses")
    if pose.shape[0] <= 0:
        raise ValueError("smpl_pose must contain at least one frame")
    return np.ascontiguousarray(pose, dtype=np.float32)


def _normalize_shape(value: Any, frame_count: int) -> np.ndarray:
    if value is None:
        return np.zeros(10, dtype=np.float32)
    shape = _as_numpy(value, "smpl_betas")
    if shape.ndim == 3:
        shape = shape[0, 0]
    elif shape.ndim == 2:
        shape = shape[0]
    elif shape.ndim != 1:
        raise ValueError("smpl_betas must be a vector, frame-major array, or sample-major frame array")
    if shape.size == 0:
        return np.zeros(10, dtype=np.float32)
    return np.ascontiguousarray(shape.reshape(-1), dtype=np.float32)


def _normalize_translation(value: Any, frame_count: int) -> np.ndarray:
    if value is None:
        return np.zeros((frame_count, 3), dtype=np.float32)
    trans = _as_numpy(value, "smpl_trans")
    if trans.ndim == 3 and trans.shape[-1] == 3:
        trans = trans[0]
    elif trans.ndim == 2 and trans.shape[-1] == 3:
        pass
    elif trans.ndim == 1 and trans.size == 3:
        trans = np.repeat(trans.reshape(1, 3), frame_count, axis=0)
    else:
        raise ValueError("smpl_trans must have shape (sample, frames, 3), (frames, 3), or (3,)")
    if trans.shape[0] == 1 and frame_count > 1:
        trans = np.repeat(trans, frame_count, axis=0)
    if trans.shape != (frame_count, 3):
        raise ValueError(f"smpl_trans frame count {trans.shape[0]} does not match pose frames {frame_count}")
    return np.ascontiguousarray(trans, dtype=np.float32)
