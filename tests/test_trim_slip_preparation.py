from pathlib import Path

import numpy as np

from scripts.trim_slip_preparation import trim_preparation


def test_slip_staggers_balance_reaction_and_floor_bracing(tmp_path: Path) -> None:
    frames = 90
    pose = np.zeros((frames, 24, 3), dtype=np.float32)
    pose[70, 16:24, 2] = np.linspace(-0.3, 0.4, 8)
    pose[70, 1:12, 0] = 0.25
    root = np.zeros((frames, 3), dtype=np.float32)
    root[:, 1] = 1.1
    source = tmp_path / "source.npz"
    destination = tmp_path / "slip.npz"
    np.savez_compressed(
        source,
        smpl_pose=pose,
        root=root,
        shape=np.zeros(10, dtype=np.float32),
        fps=np.asarray(20, dtype=np.float32),
        prompt=np.asarray("source"),
    )

    trim_preparation(source, destination, 40, 62, 70, 1.35, 0.65)

    result = np.load(destination)
    output_pose = result["smpl_pose"]
    output_root = result["root"]
    balance_frame = 47
    brace_frame = 54

    assert output_root[balance_frame, 1] > output_root[brace_frame, 1]
    assert output_root[balance_frame, 0] != output_root[40, 0]
    assert not np.allclose(output_pose[balance_frame, 16], output_pose[balance_frame, 17])
    assert not np.allclose(output_pose[brace_frame, 16], output_pose[brace_frame, 17])
    assert not np.allclose(output_pose[balance_frame, 2], output_pose[balance_frame, 1])
    assert not np.allclose(output_pose[62], output_pose[68])
    assert np.allclose(output_pose[68], output_pose[-1])
    assert str(result["prompt"]) == "a standing person unexpectedly loses balance, slips, and falls"
