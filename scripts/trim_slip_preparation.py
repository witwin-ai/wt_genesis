"""Remove the generated crouch that precedes the curated slip preset.

The source motion has a useful walking lead-in and floor landing, but frames
45-51 contain a deliberate squat.  Replacing that interval with a direct
rotation-space fall preserves the later generated bracing pose: the shoulders,
elbows, wrists, spine and legs react instead of the body rotating as one plank.
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


def trim_preparation(
    source: Path,
    destination: Path,
    stand: int,
    falling: int,
    reaction: int,
    landing_tilt_radians: float,
    landing_root_y: float,
) -> None:
    with np.load(source) as archive:
        payload = {key: archive[key] for key in archive.files}

    pose = np.asarray(payload["smpl_pose"], dtype=np.float32).copy()
    root = np.asarray(payload["root"], dtype=np.float32).copy()
    if not (0 <= stand < falling <= reaction < pose.shape[0]):
        raise ValueError("expected stand < falling <= reaction within the clip")

    count = falling - stand + 1
    source_fall_pose = pose[reaction].copy()
    landing_pose = pose[stand].copy()
    landing_pose[0] = (
        Rotation.from_rotvec([landing_tilt_radians, 0.0, 0.0])
        * Rotation.from_rotvec(pose[stand, 0])
    ).as_rotvec()
    # Keep the later generated bracing articulation while skipping the source
    # motion's squat/burpee preparation. The root is authored separately below;
    # all other joints retain asymmetric arm support, spinal flexion and a
    # non-rigid leg reaction at floor contact.
    landing_pose[1:] = source_fall_pose[1:]
    pose[stand : falling + 1] = _blend_axis_angles(pose[stand], landing_pose, count)
    pose[falling:] = landing_pose

    landing_root = root[falling].copy()
    landing_root[1] = landing_root_y
    fractions = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    root[stand : falling + 1] = root[stand] + fractions * (landing_root - root[stand])
    root[falling:] = landing_root

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
    parser.add_argument("--falling-frame", type=int, default=58)
    parser.add_argument("--reaction-frame", type=int, default=64)
    parser.add_argument("--landing-tilt-radians", type=float, default=1.35)
    parser.add_argument("--landing-root-y", type=float, default=0.65)
    args = parser.parse_args()
    trim_preparation(
        args.source,
        args.destination,
        args.stand_frame,
        args.falling_frame,
        args.reaction_frame,
        args.landing_tilt_radians,
        args.landing_root_y,
    )


if __name__ == "__main__":
    main()
