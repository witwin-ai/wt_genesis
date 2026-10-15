"""Replace the generated crouch with an involuntary, articulated slip.

The source motion has a useful walking lead-in and floor landing, but frames
45-51 contain a deliberate squat.  The replacement deliberately moves the arms
through an asymmetric balance-loss pose before the protective landing.  The
staggered arms, torso and legs keep the fall accidental instead of making the
body rotate as one rigid plank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ARM_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21, 22, 23)
TORSO_JOINTS = (0, 3, 6, 9, 12, 15)
LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
LEFT_ARM_JOINTS = (13, 16, 18, 20, 22)
RIGHT_ARM_JOINTS = (14, 17, 19, 21, 23)
LEFT_LEG_JOINTS = (1, 4, 7, 10)
RIGHT_LEG_JOINTS = (2, 5, 8, 11)


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


def _offset_pose(
    source: np.ndarray,
    offsets: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    """Return a pose with local, composable reflex offsets applied."""

    result = source.copy()
    for joint, rotvec in offsets.items():
        result[joint] = (
            Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(source[joint])
        ).as_rotvec()
    return result


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
    standing_pose = pose[stand].copy()
    pose[stand : falling + 1] = standing_pose

    # First reaction: the body twists off-axis and throws the arms apart to
    # recover balance.  It must happen before meaningful height loss so the
    # motion reads as a surprise rather than a deliberate squat.
    balance_frame = stand + max(3, (falling - stand) // 3)
    brace_frame = stand + max(6, (falling - stand) * 2 // 3)
    balance_pose = _offset_pose(
        standing_pose,
        {
            0: (0.10, -0.04, 0.16),
            3: (-0.05, 0.08, -0.11),
            6: (-0.08, 0.10, -0.13),
            9: (-0.10, 0.10, -0.16),
            16: (0.10, -0.25, -0.85),
            18: (-0.20, 0.05, -0.55),
            17: (-0.12, 0.28, 1.00),
            19: (0.28, -0.05, 0.70),
            2: (-0.12, 0.06, -0.18),
            5: (0.16, 0.0, 0.10),
        },
    )
    _blend_joints(pose, stand, balance_frame, standing_pose, balance_pose, TORSO_JOINTS)
    _blend_joints(pose, stand, balance_frame, standing_pose, balance_pose, LEFT_ARM_JOINTS)
    _blend_joints(pose, stand + 1, balance_frame, standing_pose, balance_pose, RIGHT_ARM_JOINTS)

    # The nearer hand commits to the floor first while the opposite arm keeps
    # counterbalancing. This avoids a symmetric, push-up-like preparation.
    _blend_joints(
        pose,
        balance_frame,
        brace_frame,
        balance_pose,
        landing_pose,
        LEFT_ARM_JOINTS,
    )
    _blend_joints(
        pose,
        balance_frame + 1,
        falling,
        balance_pose,
        landing_pose,
        TORSO_JOINTS,
    )
    _blend_joints(
        pose,
        brace_frame,
        falling,
        balance_pose,
        landing_pose,
        RIGHT_ARM_JOINTS,
    )
    _blend_joints(
        pose,
        balance_frame,
        falling,
        balance_pose,
        landing_pose,
        RIGHT_LEG_JOINTS,
    )
    _blend_joints(
        pose,
        brace_frame,
        falling,
        balance_pose,
        landing_pose,
        LEFT_LEG_JOINTS,
    )
    pose[falling:] = landing_pose

    # A short post-impact give in the elbows and spine prevents the body from
    # becoming a frozen board on the exact contact frame.
    settle_frame = min(falling + 6, pose.shape[0] - 1)
    settle_pose = _offset_pose(
        landing_pose,
        {
            3: (0.04, 0.0, -0.04),
            6: (0.05, 0.0, -0.03),
            18: (0.12, 0.0, -0.08),
            19: (0.10, 0.0, 0.06),
        },
    )
    _blend_joints(pose, falling, settle_frame, landing_pose, settle_pose, TORSO_JOINTS)
    _blend_joints(pose, falling, settle_frame, landing_pose, settle_pose, ARM_JOINTS)
    pose[settle_frame:] = settle_pose

    landing_root = root[falling].copy()
    landing_root[1] = landing_root_y
    root_start = stand + 2
    balance_root = root[stand].copy()
    balance_root += np.asarray([0.10, -0.025, -0.08], dtype=np.float32)
    brace_root = root[stand].copy()
    brace_root += np.asarray([0.20, -0.12, -0.18], dtype=np.float32)
    root[stand:root_start] = root[stand]
    for first, last, start_root, end_root in (
        (root_start, balance_frame, root[stand], balance_root),
        (balance_frame, brace_frame, balance_root, brace_root),
        (brace_frame, falling, brace_root, landing_root),
    ):
        fractions = _ease(last - first + 1).astype(np.float32)[:, None]
        root[first : last + 1] = start_root + fractions * (end_root - start_root)
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
    parser.add_argument("--falling-frame", type=int, default=62)
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
