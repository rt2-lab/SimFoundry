# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack scripted xArm6 demos (scripts/xarm6/generate_scripted_demos.py) into a LeRobot v3 dataset
with the same schema as the real-robot UFACTORY data (e.g. ys903/xarm6_pick_place_50_episodes):
observation.images.{front,wrist}, observation.state and action = [J1..J6.pos, gripper.pos].

Runs in the LeRobot venv, not the simfoundry env:

    .cache/pi05-serve-venv/bin/python scripts/xarm6/convert_demos_to_lerobot.py \
        --demo-dir Data/cube2_scene/xarm6_demos --root Data/cube2_scene/lerobot/xarm6_cube_in_box_sim \
        --repo-id local/xarm6_cube_in_box_sim --relative-action-chunk 15

--relative-action-chunk N recomputes the action stats in relative space (joint action minus the
current state over N-step chunks, gripper kept absolute), which is what the normalizer sees when
training with policy.use_relative_actions=true. N must equal the policy's chunk_size.
"""

import argparse
import json
import os

import av
import numpy as np

NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "gripper.pos"]
# DROID layout: 7 arm joints + gripper. The xArm6 fills J1-J6, J7 is a constant 0, and the gripper
# lands in index 7, the slot a DROID-pretrained pi0.5 (e.g. pi05_droid_jointpos) already uses for
# its gripper in both the state tokens and the action head.
DROID_NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "J7.pad", "gripper.pos"]


def to_droid_layout(x):
    """(T, 7) [J1..J6, gripper] -> (T, 8) [J1..J6, 0, gripper]."""
    return np.concatenate([x[:, :6], np.zeros((len(x), 1), x.dtype), x[:, 6:7]], axis=1)


def read_video(path):
    with av.open(path) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", required=True, nargs="+",
                    help="one or more generate_scripted_demos.py output dirs (e.g. parallel generators)")
    ap.add_argument("--max-episodes", type=int, default=None, help="stop after this many episodes in total")
    ap.add_argument("--root", required=True, help="output dataset directory (must not exist)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--relative-action-chunk", type=int, default=None)
    ap.add_argument("--droid-layout", action="store_true",
                    help="8-dim state/action [J1..J6, 0, gripper] matching a DROID-pretrained base")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for demo_dir in args.demo_dir:
        with open(os.path.join(demo_dir, "episodes.jsonl")) as f:
            episodes += [dict(json.loads(line), dir=demo_dir) for line in f if line.strip()]
    episodes = episodes[:args.max_episodes]
    first = read_video(os.path.join(episodes[0]["dir"], f"{episodes[0]['episode']}_front.mp4"))
    h, w = first.shape[1:3]
    image_feature = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    names = DROID_NAMES if args.droid_layout else NAMES
    features = {
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.images.front": image_feature,
        "observation.images.wrist": dict(image_feature),
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features, root=args.root,
                               robot_type="UFACTORY Robot", use_videos=True)
    for ep in episodes:
        name = ep["episode"]
        demo_dir = ep["dir"]
        lowdim = dict(np.load(os.path.join(demo_dir, f"{name}.npz")))
        if args.droid_layout:
            lowdim = {k: to_droid_layout(v) for k, v in lowdim.items()}
        front = read_video(os.path.join(demo_dir, f"{name}_front.mp4"))
        wrist = read_video(os.path.join(demo_dir, f"{name}_wrist.mp4"))
        n = len(lowdim["action"])
        assert len(front) == len(wrist) == n == len(lowdim["state"]), (name, len(front), len(wrist), n)
        for t in range(n):
            ds.add_frame({
                "action": lowdim["action"][t].astype(np.float32),
                "observation.state": lowdim["state"][t].astype(np.float32),
                "observation.images.front": front[t],
                "observation.images.wrist": wrist[t],
                "task": ep["prompt"],
            })
        ds.save_episode()
        print(f"[convert] {name}: {n} frames, task={ep['prompt']!r}", flush=True)
    ds.finalize()
    print(f"[convert] {len(episodes)} episodes -> {args.root}")
    if args.relative_action_chunk:
        from lerobot.datasets.dataset_tools import recompute_stats

        ds = LeRobotDataset(args.repo_id, root=args.root)
        recompute_stats(ds, relative_action=True, relative_exclude_joints=["gripper"],
                        chunk_size=args.relative_action_chunk)
        print(f"[convert] action stats recomputed in relative space (chunk {args.relative_action_chunk})")


if __name__ == "__main__":
    main()
