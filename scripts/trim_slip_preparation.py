"""Remove the generated crouch that precedes the curated slip preset.

The source motion has a useful walking lead-in and floor landing, but frames
45-51 contain a deliberate squat.  Replacing that interval with a short
rotation-space blend produces an immediate loss of balance while preserving
the generated landing and recovery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _blend_axis_angles(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, count, dtype=np.float64)
    blended = np.empty((count, *start.shape), dtype=np.float32)
    for joint in range(start.shape[0]):
        rotations = Rotation.from_rotvec(np.stack((start[joint], end[joint])))
        blended[:, joint] = Slerp([0.0, 1.0], rotations)(fractions).as_rotvec()
    return blended


def trim_preparation(source: Path, destination: Path, stand: int, falling: int) -> None:
    with np.load(source) as archive:
        payload = {key: archive[key] for key in archive.files}

    pose = np.asarray(payload["smpl_pose"], dtype=np.float32).copy()
    root = np.asarray(payload["root"], dtype=np.float32).copy()
    if not (0 <= stand < falling < pose.shape[0]):
        raise ValueError("expected 0 <= stand < falling < frame count")

    count = falling - stand + 1
    pose[stand : falling + 1] = _blend_axis_angles(pose[stand], pose[falling], count)
    fractions = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    root[stand : falling + 1] = root[stand] + fractions * (root[falling] - root[stand])

    payload["smpl_pose"] = pose
    payload["root"] = root
    payload["prompt"] = np.asarray(
        "a standing person unexpectedly loses balance, slips, and falls",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--stand-frame", type=int, default=44)
    parser.add_argument("--falling-frame", type=int, default=52)
    args = parser.parse_args()
    trim_preparation(args.source, args.destination, args.stand_frame, args.falling_frame)


if __name__ == "__main__":
    main()
