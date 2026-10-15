# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the imported xArm6 in an empty scene and check what pipelines depend on.

Checks: DOF layout, mimic joints following their drivers, eef frame convention, absolute joint
control (what policy eval uses), the 0..1 gripper command direction under the eval's controller
config, and that a wrist camera parented to link6 renders. Writes images to --out-dir.

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/assets/smoke_test_xarm6.py --out-dir /tmp/xarm6_smoke
"""

import argparse
import json
import os

import numpy as np

WRIST_CAM = {
    "sensor_type": "VisionSensor",
    "name": "wrist_cam",
    "relative_prim_path": "/controllable__xarm6__robot0/link6/wrist_cam",
    "modalities": ["rgb"],
    "sensor_kwargs": {"image_height": 480, "image_width": 640, "focal_length": 1.93, "horizontal_aperture": 3.6},
    # UFactory RealSense D435i wrist mount, color optical frame in link6, USD camera convention.
    "position": [0.06746, -0.0325, 0.0237],
    "orientation": [0.70711, 0.70711, 0.0, 0.0],
    "pose_frame": "parent",
}
def look_at_quat(eye, target):
    """XYZW quaternion for a USD camera (looks along -z, +y up) at @eye facing @target."""
    from scipy.spatial.transform import Rotation

    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat().tolist()


VIEW_EYE, VIEW_TARGET = [1.0, 0.7, 0.8], [0.15, 0.0, 0.25]
VIEW_CAM = {
    "sensor_type": "VisionSensor",
    "name": "view_cam",
    "relative_prim_path": "/view_cam",
    "modalities": ["rgb"],
    "sensor_kwargs": {"image_height": 480, "image_width": 640},
    "position": VIEW_EYE,
    "orientation": look_at_quat(VIEW_EYE, VIEW_TARGET),
    "pose_frame": "world",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import omnigibson as og
    import torch as th
    from omnigibson.utils.ui_utils import KeyboardEventHandler  # noqa: F401  (kit must be up)
    from PIL import Image

    from simfoundry import import_og_dependencies
    import_og_dependencies()

    cfg = {
        "env": {"external_sensors": [WRIST_CAM, VIEW_CAM], "action_frequency": 15, "rendering_frequency": 15,
                "physics_frequency": 120},
        "scene": {"type": "Scene", "use_floor_plane": True},
        "robots": [{
            "type": "XArm6", "name": "robot0", "obs_modalities": ["proprio"], "action_normalize": False,
            "grasping_mode": "physical", "self_collisions": False,
        }],
    }
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    report = {}

    report["dof_names"] = list(robot.joints.keys())
    report["n_dof"] = robot.n_dof
    report["arm_control_idx"] = robot.arm_control_idx[robot.default_arm].tolist()
    report["gripper_control_idx"] = robot.gripper_control_idx[robot.default_arm].tolist()
    report["joint_limits"] = {n: [float(j.lower_limit), float(j.upper_limit)] for n, j in robot.joints.items()}

    env.reset()
    # Same controllers the eval stage installs (1_eval_policy_og_scene.py), except the gripper's
    # `inverted`: the eval hardcodes True, which gives 1 = closed only for grippers whose lower
    # limit is closed (Franka). The xArm Gripper closes at its upper limit.
    robot.reload_controllers({
        f"arm_{robot.default_arm}": {"name": "JointController", "motor_type": "position",
                                     "command_input_limits": None, "use_delta_commands": False},
        f"gripper_{robot.default_arm}": {"name": "MultiFingerGripperController", "command_input_limits": (0.0, 1.0),
                                         "mode": "smooth", "inverted": robot._grasping_direction == "lower"},
    })

    def settle(action, n=45):
        for _ in range(n):
            env.step(action)

    home = th.tensor(robot._default_joint_pos[:6], dtype=th.float32)
    settle(th.cat([home, th.tensor([0.0])]))
    q = robot.get_joint_positions()
    report["home_arm_q"] = q[robot.arm_control_idx[robot.default_arm]].tolist()
    pos, quat = robot.get_eef_position(), robot.get_eef_orientation()
    from omnigibson.utils.transform_utils import quat2mat
    R = quat2mat(quat)
    report["eef_pos"] = pos.tolist()
    report["eef_z_axis_world"] = R[:, 2].tolist()   # should point roughly down (-z) at home
    report["eef_y_axis_world"] = R[:, 1].tolist()

    def gripper_state():
        q = robot.get_joint_positions()
        return {n: round(float(q[i]), 4) for i, n in enumerate(robot.joints) if "knuckle" in n or "finger" in n or n == "drive_joint"}

    report["gripper_after_cmd_0"] = gripper_state()
    Image.fromarray(env.external_sensors["wrist_cam"].get_obs()[0]["rgb"][..., :3].cpu().numpy()).save(f"{args.out_dir}/wrist_open.png")
    Image.fromarray(env.external_sensors["view_cam"].get_obs()[0]["rgb"][..., :3].cpu().numpy()).save(f"{args.out_dir}/view_open.png")

    settle(th.cat([home, th.tensor([1.0])]))
    report["gripper_after_cmd_1"] = gripper_state()
    Image.fromarray(env.external_sensors["view_cam"].get_obs()[0]["rgb"][..., :3].cpu().numpy()).save(f"{args.out_dir}/view_cmd1.png")

    # Absolute joint target tracking.
    target = home.clone()
    target[0] += 0.4
    target[4] -= 0.3
    settle(th.cat([target, th.tensor([0.0])]), n=60)
    q = robot.get_joint_positions()[robot.arm_control_idx[robot.default_arm]]
    report["joint_track_target"] = target.tolist()
    report["joint_track_reached"] = q.tolist()
    report["joint_track_max_err"] = float((q - target).abs().max())

    with open(f"{args.out_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    og.shutdown()


if __name__ == "__main__":
    main()
