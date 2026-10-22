# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move the real xArm6 to the mean start pose of the autoalloc demos and open the gripper.

Position mode, 0.25 rad/s joint-space move. If the straight joint-space path would take the flange
below 0.2 m (e.g. right after a grasp), it first lifts 6 cm in Cartesian space so the move clears
the box. Run before every deploy_pi05_xarm.py rollout: the policy never saw other start poses.

    .cache/radio-prep-venv/bin/python scripts/xarm6/real/go_to_start_pose.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from xarm.wrapper import XArmAPI  # noqa: E402

from simfoundry.robots.xarm6_kinematics import XArm6Kinematics  # noqa: E402

START = np.array([-0.052, -0.332, -0.517, 0.026, 0.957, 0.0])  # mean of the 50 demo start poses


def main(ip="192.168.1.111"):
    arm = XArmAPI(ip, is_radian=True)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    q = np.asarray(arm.get_servo_angle(is_radian=True)[1][:6])
    kin = XArm6Kinematics(tip_link="link6")
    fz = [kin.fk(q + s * (START - q))[2, 3] for s in np.linspace(0, 1, 11)]
    print("path flange z min", round(min(fz), 3))
    if min(fz) < 0.2:
        arm.set_position(z=60, relative=True, speed=50, wait=True)
    arm.set_gripper_position(840, wait=False)
    code = arm.set_servo_angle(angle=START.tolist(), speed=0.25, mvacc=1.0, is_radian=True, wait=True)
    print("move code", code, "now", np.round(arm.get_servo_angle(is_radian=True)[1][:6], 3), "err", arm.error_code)
    arm.disconnect()


if __name__ == "__main__":
    main(*sys.argv[1:])
