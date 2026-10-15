# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UFactory xArm6 with the UFactory xArm Gripper, as an OmniGibson robot.

Lives in simfoundry/ rather than OmniGibson's robots/ because deps/ must not be edited
(AGENTS.md). Defining the class registers it with OmniGibson (Registerable.__init_subclass__),
so any code path that calls ``simfoundry.import_og_dependencies()`` can use
``robot_config.type: XArm6``.

The USD is produced by ``scripts/assets/import_custom_robot.py`` from
``assets/robots/xarm6/import_xarm6.yaml``; see ``assets/robots/xarm6/`` for how the URDF was
derived from the official xarm_description package.
"""

import math
import os
from functools import cached_property

import torch as th

from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.transform_utils import euler2quat

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XARM6_ASSET_DIR = os.environ.get(
    "SIMFOUNDRY_XARM6_ASSET_DIR",
    os.path.join(_REPO_ROOT, "assets", "og_custom", "objects", "robot", "xarm6"),
)

# drive_joint range from the xArm Gripper URDF: 0 is fully open, 0.85 rad fully closed. The
# UFactory SDK reports the same gripper with the opposite sign and units (850 = open, 0 =
# closed), and LeRobot's UFACTORY datasets normalise it to [0, 1] with 1 = closed.
XARM_GRIPPER_CLOSED_RAD = 0.85

# The two arm configurations below sit inside the ranges of the ys903/xarm6_pick_place_50_episodes
# real-robot dataset (gripper pointing down over the table in front of the base), so a policy
# trained on it starts from a familiar state.
XARM6_HOME_ARM_JOINTS = [0.0, -0.26, -0.63, 0.0, 0.9, 0.0]


class XArm6(ManipulationRobot):
    """The UFactory xArm6 6-DOF arm equipped with the UFactory xArm Gripper."""

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        fixed_base=True,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        finger_static_friction=None,
        finger_dynamic_friction=None,
        **kwargs,
    ):
        """See omnigibson.robots.ur5e.UR5e for argument documentation; the signature is identical."""
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            # Upper joint limit (0.85 rad) is the closed gripper.
            grasping_direction="upper",
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            **kwargs,
        )

    @property
    def usd_path(self):
        return os.path.join(XARM6_ASSET_DIR, "usd", "xarm6.usda")

    @property
    def urdf_path(self):
        return os.path.join(XARM6_ASSET_DIR, "urdf", "xarm6.urdf")

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("XArm6 does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        # 6 arm joints, then the gripper's 6 joints (drive + 5 mimics), all open.
        return th.tensor(XARM6_HOME_ARM_JOINTS + [0.0] * 6)

    @cached_property
    def arm_link_names(self):
        return {self.default_arm: ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]}

    @cached_property
    def arm_joint_names(self):
        return {self.default_arm: [f"joint{i}" for i in range(1, 7)]}

    @cached_property
    def eef_link_names(self):
        return {self.default_arm: "eef_link"}

    @cached_property
    def finger_link_names(self):
        return {self.default_arm: ["left_finger", "right_finger"]}

    @cached_property
    def finger_joint_names(self):
        # The two independently driven outer knuckles; every other gripper joint is a PhysX
        # mimic of one of these (see clean_urdf.py), so commanding them directly would fight
        # the mimic constraint.
        return {self.default_arm: ["drive_joint", "right_outer_knuckle_joint"]}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: euler2quat([-math.pi, 0, 0])}

    # Assisted-grasp rays run between the inner pad faces, read off the finger meshes: in each
    # finger's link frame the pad spans x in [-0.016, 0.016], z in [0, 0.061] (z = 0.061 is the
    # fingertip, 1 cm short of eef_link) and faces the other finger at y = -0.026 (left) /
    # +0.026 (right). Rays sit in the lower half of the pad, where a small object is held.
    _PAD_POINTS = [(x, z) for x in (-0.01, 0.01) for z in (0.035, 0.05)]

    @property
    def _assisted_grasp_start_points(self):
        return {
            self.default_arm: [GraspingPoint(link_name="left_finger", position=th.tensor([x, -0.026, z]))
                               for x, z in self._PAD_POINTS]
        }

    @property
    def _assisted_grasp_end_points(self):
        return {
            self.default_arm: [GraspingPoint(link_name="right_finger", position=th.tensor([x, 0.026, z]))
                               for x, z in self._PAD_POINTS]
        }

    @property
    def disabled_collision_pairs(self):
        # The four-bar linkage's links overlap by construction; PhysX can't close the loop, so
        # letting them collide only makes the gripper jitter.
        links = [
            "xarm_gripper_base_link",
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        ]
        return [[a, b] for i, a in enumerate(links) for b in links[i + 1 :]]
