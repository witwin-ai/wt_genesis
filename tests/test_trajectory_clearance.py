from __future__ import annotations

import pytest

from wt_genesis.adapter.trajectory_clearance import (
    HumanoidTrajectoryBlocked,
    validate_humanoid_trajectory,
)


def _box(object_id, low, high, *, structure=False):
    return {
        "object_id": object_id,
        "name": object_id,
        "structure": structure,
        "points": [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
    }


def test_blocks_a_humanoid_root_segment_that_crosses_furniture():
    with pytest.raises(HumanoidTrajectoryBlocked) as raised:
        validate_humanoid_trajectory(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[-0.2, 0.0, -0.1], [0.2, 1.8, 0.1]],
            [_box("bed", [0.9, 0.0, -0.2], [1.1, 0.6, 0.2])],
        )

    blocker = raised.value.blockers[0]
    assert blocker["object_id"] == "bed"
    assert blocker["segment_index"] == 0
    assert blocker["body_radius_m"] == pytest.approx(0.3036068)


def test_floor_support_is_not_treated_as_a_blocking_obstacle():
    result = validate_humanoid_trajectory(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[-0.2, 0.0, -0.1], [0.2, 1.8, 0.1]],
        [_box("floor", [-5.0, -0.1, -5.0], [5.0, 0.0, 5.0], structure=True)],
    )

    assert result["status"] == "clear_at_root_sweep"
    assert result["obstacles_checked"] == 0
    assert result["continuous_collision_certified"] is False


def test_reports_clear_only_after_every_segment_and_obstacle_is_checked():
    result = validate_humanoid_trajectory(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.5]],
        [[-0.15, 0.0, -0.1], [0.15, 1.8, 0.1]],
        [_box("wardrobe", [2.0, 0.0, 2.0], [2.5, 2.0, 2.5])],
    )

    assert result["segments"] == 2
    assert result["obstacles_checked"] == 1
