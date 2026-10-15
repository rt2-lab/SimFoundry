# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OmniGibson task env for an xArm6 in a reconstructed scene, in the UFACTORY LeRobot convention.

Shared by policy eval (scripts/xarm6/eval_pi05_xarm6.py) and scripted demo generation
(scripts/xarm6/generate_scripted_demos.py), so that both see the same controllers, cameras and
state/action layout:

    state / action = [J1..J6 (rad, absolute), gripper (0 = open, 1 = closed)]
    images         = {"front": (H, W, 3) uint8, "wrist": (H, W, 3) uint8}

Must be imported after OmniGibson is up (it imports omnigibson at module scope).
"""

import json
import os

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import parse_config

from simfoundry import CFG_DIR, import_og_dependencies
from simfoundry.robots.xarm6 import XARM_GRIPPER_CLOSED_RAD
from simfoundry.utils.python_utils import resolve_task_config_path
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

CAMERA_NAMES = ("front", "wrist")


def make_xarm6_env(
    scene_json,
    task_name,
    out_dir,
    sensors_cfg="sim_xarm6_front_wrist",
    action_freq=30,
    max_steps=900,
    grasping_mode="physical",
    robot_pose=None,
):
    """Build the env and install absolute joint control on the arm and a 0..1 gripper command.

    Args:
        scene_json (str): Stage 14 scene JSON with an XArm6 named robot0.
        task_name (str): Task YAML under scripts/cfg/task.
        out_dir (str): Where the modified scene JSON is written.
        sensors_cfg (str): External sensors YAML under scripts/cfg/external_sensors; its
            cameras must be named "<name>_cam" for each name in CAMERA_NAMES.
        action_freq (int): Control rate. The ys903 xArm6 dataset is 30 Hz.
        max_steps (int): Task termination horizon.
        grasping_mode (str): "physical" or "assisted".
        robot_pose (None or (pos, quat_xyzw)): Overrides robot0's base pose from the scene JSON.
    """
    import_og_dependencies()
    os.makedirs(out_dir, exist_ok=True)

    scene = load_json_with_absolute_usd_paths(scene_json)
    robot_info = scene["objects_info"]["init_info"]["robot0"]
    assert robot_info["class_name"] == "XArm6", f"{scene_json} has a {robot_info['class_name']}, not an XArm6"
    robot_info["args"]["grasping_mode"] = grasping_mode
    if robot_pose is not None:
        root = scene["state"]["registry"]["object_registry"]["robot0"]["root_link"]
        root["pos"], root["ori"] = list(robot_pose[0]), list(robot_pose[1])
    modified = os.path.join(out_dir, "xarm6_scene.json")
    with open(modified, "w") as f:
        json.dump(scene, f, indent=2)

    task_cfg = parse_config(resolve_task_config_path(CFG_DIR, task_name))["og_task_config"]
    task_cfg["termination_config"]["max_steps"] = max_steps
    sensors = parse_config(f"{CFG_DIR}/external_sensors/{sensors_cfg}.yaml")["external_sensors"]

    env = og.Environment(configs={
        "env": {"external_sensors": sensors, "action_frequency": action_freq,
                "rendering_frequency": action_freq, "physics_frequency": 120},
        "scene": {"type": "Scene", "scene_file": modified, "use_floor_plane": True,
                  "floor_plane_visible": True, "use_skybox": True, "include_robots": True},
        "task": task_cfg,
    })
    env.reset()
    robot = env.robots[0]
    _apply_finger_friction(robot)
    robot.reload_controllers({
        f"arm_{robot.default_arm}": {"name": "JointController", "motor_type": "position",
                                     "command_input_limits": None, "use_delta_commands": False},
        # MultiFingerGripperController maps 0..1 onto the finger joint range; `inverted` makes
        # 1 = closed regardless of which joint limit is the closed one.
        f"gripper_{robot.default_arm}": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0),
                                         "mode": "smooth", "inverted": robot._grasping_direction == "lower"},
    })
    return env


def _apply_finger_friction(robot, friction=2.0):
    """High-friction pads on the gripper's finger and knuckle links, as the Franka eval stage does
    (1_eval_policy_og_scene.py); the default material lets a squeezed cube slide out on lift."""
    import omnigibson.lazy as lazy

    mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
        prim_path=f"{robot.prim_path}/Looks/finger_mat", name="finger_mat",
        static_friction=friction, dynamic_friction=friction, restitution=None)
    for name, link in robot.links.items():
        if "finger" in name or "knuckle" in name:
            for mesh in link.collision_meshes.values():
                mesh.apply_physics_material(mat)


def reset_xarm6_env(env, home=None, settle_steps=30):
    """env.reset(), then drive the arm to @home (default: the robot's default pose) with the
    gripper open and let objects settle. Returns the action that holds this pose."""
    robot = env.robots[0]
    if robot._ag_obj_in_hand[robot.default_arm] is not None:  # an episode may end mid-grasp
        robot.release_grasp_immediately(arm=robot.default_arm)
    env.reset()
    home = robot._default_joint_pos[:6] if home is None else th.as_tensor(home, dtype=th.float32)
    robot.set_joint_positions(th.cat([home, th.zeros(robot.n_dof - 6)]))
    robot.keep_still()
    hold = th.cat([home, th.tensor([0.0])])
    for _ in range(settle_steps):
        env.step(hold)
    return hold


def xarm6_state(robot):
    """(7,) [J1..J6, gripper] with gripper = drive_joint / closed angle (0 = open, 1 = closed)."""
    q = robot.get_joint_positions()
    arm = q[robot.arm_control_idx[robot.default_arm]]
    grip = q[list(robot.joints).index("drive_joint")] / XARM_GRIPPER_CLOSED_RAD  # joints are in DOF order
    return np.concatenate([arm.cpu().numpy(), [float(grip)]]).astype(np.float32)


def xarm6_images(env):
    """{"front", "wrist"} -> (H, W, 3) uint8."""
    out = {}
    for name in CAMERA_NAMES:
        rgb = env.external_sensors[f"{name}_cam"].get_obs()[0]["rgb"][..., :3]
        out[name] = rgb.cpu().numpy().astype(np.uint8)
    return out


def articulated_joint(env, category, joint_name):
    """The named joint of the scene object of @category. Fails loudly: the task's
    group_init_joint_positions silently skips objects without joints, which once hid a box
    whose lid had been imported as a fixed joint."""
    objs = env.scene.object_registry("category", category)
    assert objs, f"no object of category {category!r} in the scene"
    obj = next(iter(objs))
    assert joint_name in obj.joints, (
        f"{obj.name} ({category}) has joints {list(obj.joints)}, not {joint_name!r}; "
        "was the articulation propagated through stages 11-14?")
    return obj.joints[joint_name]


class GraspLatch:
    """Optional grasp constraint for the xArm6 driven by geometry instead of contact reports.

    Not needed with the force-drive asset (scripts/assets/fix_xarm6_usd.py), which grasps
    physically. Kept for debugging. OmniGibson's own assisted grasp can't be used instead: it
    needs contact reports from both finger links (none are reported for this imported gripper)
    and treats the lower joint limit as closed, so it releases an xArm grasp on the next step.
    The robot runs in "physical" mode and this class attaches / releases with OmniGibson's grasp
    constraint. Call update() after every env.step():

    - commanded closed, gripper closed past @min_closure on something, and a candidate's centre
      between the pads -> attach that candidate;
    - commanded open while holding -> release.

    Caveat: once attached, OmniGibson freezes the gripper at its current opening, so attaching
    before the pads touch leaves the object floating between open fingers. Set @min_closure
    at or above the contact closure for the object size.
    """

    # Pad region in the eef_link frame: x across the pad width, y along the closing direction,
    # z from the fingertips (1 cm behind eef_link) back to the pad root.
    PAD_BOX_LO = np.array([-0.025, -0.045, -0.075])
    PAD_BOX_HI = np.array([0.025, 0.045, 0.0])

    def __init__(self, robot, candidates, min_closure=0.3):
        assert robot.grasping_mode == "physical", "GraspLatch replaces OmniGibson's assisted grasp; use physical"
        self.robot, self.candidates, self.min_closure = robot, list(candidates), min_closure

    def holding(self):
        return self.robot._ag_obj_in_hand[self.robot.default_arm]

    def update(self, gripper_cmd):
        robot, arm = self.robot, self.robot.default_arm
        if self.holding() is not None:
            if gripper_cmd < 0.5:
                robot.release_grasp_immediately(arm=arm)
            return self.holding()
        if gripper_cmd < 0.5 or xarm6_state(robot)[6] < self.min_closure:
            return None
        eef_pos, eef_quat = robot.get_eef_position(), robot.get_eef_orientation()
        rot = T.quat2mat(eef_quat)
        for obj in self.candidates:
            center = obj.get_position_orientation()[0]
            local = (rot.T @ (center - eef_pos)).cpu().numpy()
            if np.all(local > self.PAD_BOX_LO) and np.all(local < self.PAD_BOX_HI):
                robot._establish_grasp(arm=arm, ag_data=(obj, obj.root_link), contact_pos=center)
                return obj
        return None
