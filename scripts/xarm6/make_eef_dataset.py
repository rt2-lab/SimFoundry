# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn a joint-space xArm6 LeRobot dataset (convert_real_h5_to_lerobot.py) into a task-space one.
Videos and frames are reused as they are; only observation.state and action change:

    action            (10) = TCP target  [x, y, z, r11, r21, r31, r12, r22, r32, gripper]
    observation.state (16) = TCP current [x, y, z, r11..r32, gripper] + [J1..J6]

TCP = eef_link of the xArm6 URDF (link6 + 0.172 m, the gripper's fingertip centre), in the robot
base frame, by forward kinematics of the dataset's own joint columns: the target from the desired
joints (xarm_stack's IK output; matches the recorded robot_desired_pose to 0.13 mm median), the
current pose from the measured joints. Rotation is the 6D representation (first two columns of R):
continuous, no quaternion sign flips (the recorded quaternions do flip).

The first 10 state dims line up with the action, so LeRobot's stock relative actions
(action[:10] - state[:10], gripper excluded) give, per 50-step chunk, the TCP translation in the
base frame and the change of the rotation columns relative to the pose at the chunk's first frame.
Decode at run time: target = relative + current, then Gram-Schmidt the two columns.

    .cache/pi05-serve-venv/bin/python scripts/xarm6/make_eef_dataset.py \\
        --src Data/cube2_scene/lerobot/xarm6_real_sim --root Data/cube2_scene/lerobot/xarm6_real_sim_eef
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

POS = ["tcp.x", "tcp.y", "tcp.z"]
ROT = ["tcp.r11", "tcp.r21", "tcp.r31", "tcp.r12", "tcp.r22", "tcp.r32"]
ACTION_NAMES = POS + ROT + ["gripper.pos"]
STATE_NAMES = ACTION_NAMES + ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos"]


def tcp_pose9(kin, q):
    """(N, 6) joints -> (N, 9) [xyz, first two columns of R] of eef_link in the base frame."""
    out = np.empty((len(q), 9), np.float32)
    for i, qi in enumerate(q):
        T = kin.fk(qi)
        out[i, :3] = T[:3, 3]
        out[i, 3:] = T[:3, :2].T.reshape(-1)  # r11 r21 r31 r12 r22 r32
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="joint-space dataset (7-D state/action)")
    ap.add_argument("--root", required=True, help="output dataset directory (must not exist)")
    ap.add_argument("--chunk", type=int, default=50, help="policy chunk_size, for relative action stats")
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import modify_features, recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from simfoundry.robots.xarm6_kinematics import XArm6Kinematics

    src_id = "local/" + os.path.basename(args.src.rstrip("/"))
    out_id = "local/" + os.path.basename(args.root.rstrip("/"))
    ds = LeRobotDataset(src_id, root=args.src)
    assert list(ds.meta.features["action"]["names"])[:6] == ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos"]
    cols = ds.hf_dataset.with_format("numpy")
    q_cur = np.asarray(cols["observation.state"], np.float64)
    q_des = np.asarray(cols["action"], np.float64)
    order = np.asarray(cols["index"])
    assert (order == np.arange(len(order))).all(), "expected frames stored in index order"

    kin = XArm6Kinematics(tip_link="eef_link")
    state = np.concatenate([tcp_pose9(kin, q_cur[:, :6]), q_cur[:, 6:7], q_cur[:, :6]], axis=1).astype(np.float32)
    action = np.concatenate([tcp_pose9(kin, q_des[:, :6]), q_des[:, 6:7]], axis=1).astype(np.float32)
    print(f"[eef] {len(state)} frames; TCP z range {state[:, 2].min():.3f}..{state[:, 2].max():.3f} m", flush=True)

    # modify_features will not add a name that already exists, so swap through temporary columns.
    tmp_root = args.root.rstrip("/") + "_tmp"
    info_a = {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES}
    info_s = {"dtype": "float32", "shape": (16,), "names": STATE_NAMES}
    tmp = modify_features(ds, add_features={"tmp_action": (action, info_a), "tmp_state": (state, info_s)},
                          remove_features=["action", "observation.state"], output_dir=tmp_root, repo_id=out_id + "_tmp")
    modify_features(tmp, add_features={"action": (action, info_a), "observation.state": (state, info_s)},
                    remove_features=["tmp_action", "tmp_state"], output_dir=args.root, repo_id=out_id)
    shutil.rmtree(tmp_root)
    shutil.copy(os.path.join(args.src, "meta", "sources.jsonl"), os.path.join(args.root, "meta", "sources.jsonl"))

    out = LeRobotDataset(out_id, root=args.root)
    recompute_stats(out, relative_action=True, relative_exclude_joints=["gripper"], chunk_size=args.chunk)
    print(f"[eef] wrote {args.root}: action {ACTION_NAMES}, state {len(STATE_NAMES)}-D, relative stats (chunk {args.chunk})")


if __name__ == "__main__":
    main()
