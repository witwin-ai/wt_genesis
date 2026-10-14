"""Replace the generated crouch with an involuntary, articulated slip.

The source motion has a useful walking lead-in and floor landing, but frames
45-51 contain a deliberate squat.  The replacement deliberately staggers the
protective arm response ahead of the torso and legs, then moves the arms toward
the generated landing pose.  This keeps the fall accidental instead of making
the body rotate as one rigid plank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ARM_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21, 22, 23)
TORSO_JOINTS = (0, 3, 6, 9, 12, 15)
LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)


def _ease(count: int) -> np.ndarray:
    linear = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return linear * linear * (3.0 - 2.0 * linear)


def _blend_joints(
    pose: np.ndarray,
    first: int,
    last: int,
    start: np.ndarray,
    end: np.ndarray,
    joints: tuple[int, ...],
) -> None:
    fractions = _ease(last - first + 1)
    for joint in joints:
        rotations = Rotation.from_rotvec(np.stack((start[joint], end[joint])))
        pose[first : last + 1, joint] = Slerp([0.0, 1.0], rotations)(
            fractions,
        ).as_rotvec()


def trim_preparation(
    source: Path,
    destination: Path,
    stand: int,
    falling: int,
    reaction: int,
    protective: int,
    landing_tilt_radians: float,
    landing_root_y: float,
) -> None:
    with np.load(source) as archive:
        payload = {key: archive[key] for key in archive.files}

    pose = np.asarray(payload["smpl_pose"], dtype=np.float32).copy()
    root = np.asarray(payload["root"], dtype=np.float32).copy()
    if not (0 <= protective < stand < falling <= reaction < pose.shape[0]):
        raise ValueError(
            "expected protective < stand < falling <= reaction within the clip",
        )

    source_fall_pose = pose[reaction].copy()
    protective_pose = pose[protective].copy()
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
    standing_pose = pose[stand].copy()
    pose[stand : falling + 1] = standing_pose
    arm_peak = stand + max(2, (falling - stand) // 2)
    _blend_joints(
        pose,
        stand,
        arm_peak,
        standing_pose,
        protective_pose,
        ARM_JOINTS,
    )
    _blend_joints(
        pose,
        arm_peak,
        falling,
        protective_pose,
        landing_pose,
        ARM_JOINTS,
    )
    _blend_joints(
        pose,
        stand + 3,
        falling,
        standing_pose,
        landing_pose,
        TORSO_JOINTS,
    )
    _blend_joints(
        pose,
        stand + 5,
        falling,
        standing_pose,
        landing_pose,
        LEG_JOINTS,
    )
    pose[falling:] = landing_pose

    landing_root = root[falling].copy()
    landing_root[1] = landing_root_y
    root_start = stand + 3
    fractions = _ease(falling - root_start + 1).astype(np.float32)[:, None]
    root[stand:root_start] = root[stand]
    root[root_start : falling + 1] = (
        root[stand] + fractions * (landing_root - root[stand])
    )
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
    parser.add_argument("--protective-frame", type=int, default=5)
    parser.add_argument("--landing-tilt-radians", type=float, default=1.35)
    parser.add_argument("--landing-root-y", type=float, default=0.65)
    args = parser.parse_args()
    trim_preparation(
        args.source,
        args.destination,
        args.stand_frame,
        args.falling_frame,
        args.reaction_frame,
        args.protective_frame,
        args.landing_tilt_radians,
        args.landing_root_y,
    )


if __name__ == "__main__":
    main()
