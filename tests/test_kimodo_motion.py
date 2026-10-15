from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def test_soma_motion_maps_major_joints_and_resamples_to_genesis_fps(tmp_path):
    from wt_genesis.adapter.kimodo_motion import load_kimodo_motion

    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None], 4 * 77, axis=0).reshape(4, 77, 3, 3)
    rotations[:, 67] = Rotation.from_rotvec([[0.0, 0.0, angle] for angle in (0.0, 0.1, 0.2, 0.3)]).as_matrix()
    roots = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]], dtype=np.float32)
    path = tmp_path / "soma.npz"
    np.savez(path, local_rot_mats=rotations, root_positions=roots, fps=np.array(30.0))

    result = load_kimodo_motion(path, target_fps=20.0)

    assert result["smpl_pose"].shape == (1, 3, 24, 3)
    assert result["smpl_betas"].shape == (1, 3, 10)
    assert result["smpl_trans"].shape == (1, 3, 3)
    np.testing.assert_allclose(result["smpl_trans"][0, -1], [0, 1, 3])
    np.testing.assert_allclose(result["smpl_pose"][0, -1, 1], [0, 0, 0.3], atol=1e-5)


def test_amass_motion_pads_smpl_hands_and_converts_z_up_to_y_up(tmp_path):
    from wt_genesis.adapter.kimodo_motion import load_kimodo_motion

    root = np.zeros((2, 3), dtype=np.float32)
    body = np.zeros((2, 63), dtype=np.float32)
    body[:, :3] = [0.1, 0.2, 0.3]
    path = tmp_path / "amass.npz"
    np.savez(
        path,
        root_orient=root,
        pose_body=body,
        trans=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        mocap_frame_rate=np.array(20.0),
    )

    result = load_kimodo_motion(path)

    np.testing.assert_allclose(result["smpl_pose"][0, 0, 1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result["smpl_pose"][0, :, 22:], 0)
    np.testing.assert_allclose(result["smpl_trans"][0, 0], [-1, 3, 2])
