"""Public Python API of wt_genesis for other Plugins (D16 layer C).

Other Plugins import only this module, and only after listing
``witwin.genesis`` in their ``pluginDependencies``. Every other module here
is private and may change without notice.
"""
from __future__ import annotations

from .adapter.foot_lock import (
    ground_upright_floats,
    lift_dipping_feet,
    limit_bone_turn_rate,
    lock_planted_feet,
    set_down_hovering_feet,
    settle_first_frame,
)
from .adapter.kimodo_backend import generate_smpl_motion, smpl_rest_skeleton, smpl_skin
from .adapter.kimodo_path import create_humanoid_path_animation, plan_root_path
from .adapter.kimodo_poses import (
    BACK_REGION_JOINTS,
    SEAT_REGION_JOINTS,
    EndPose,
    ground_end_pose,
    posed_vertices,
    supine_pelvis,
    upright_pelvis,
)
from .adapter.kimodo_poses import _world_joints as world_joints
from .adapter.rest_idle import rest_idle_poses
from .adapter.scene_animation import (
    bone_rotation_keys,
    create_humanoid_animation,
    root_motion_keys,
)
from .adapter.scene_animation import _continuous_euler_values as continuous_euler_values
from .adapter.scene_animation import _pivot_root_on_pelvis as pivot_root_on_pelvis
from .adapter.trajectory_clearance import HumanoidTrajectoryBlocked

__all__ = [
    "BACK_REGION_JOINTS",
    "SEAT_REGION_JOINTS",
    "EndPose",
    "HumanoidTrajectoryBlocked",
    "bone_rotation_keys",
    "continuous_euler_values",
    "create_humanoid_animation",
    "create_humanoid_path_animation",
    "generate_smpl_motion",
    "ground_end_pose",
    "ground_upright_floats",
    "lift_dipping_feet",
    "limit_bone_turn_rate",
    "lock_planted_feet",
    "pivot_root_on_pelvis",
    "plan_root_path",
    "posed_vertices",
    "rest_idle_poses",
    "root_motion_keys",
    "set_down_hovering_feet",
    "settle_first_frame",
    "smpl_rest_skeleton",
    "smpl_skin",
    "supine_pelvis",
    "upright_pelvis",
    "world_joints",
]
