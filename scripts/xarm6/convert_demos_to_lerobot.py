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


def read_video(path):
    with av.open(path) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--root", required=True, help="output dataset directory (must not exist)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--relative-action-chunk", type=int, default=None)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with open(os.path.join(args.demo_dir, "episodes.jsonl")) as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    first = read_video(os.path.join(args.demo_dir, f"{episodes[0]['episode']}_front.mp4"))
    h, w = first.shape[1:3]
    image_feature = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": NAMES},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": NAMES},
        "observation.images.front": image_feature,
        "observation.images.wrist": dict(image_feature),
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features, root=args.root,
                               robot_type="UFACTORY Robot", use_videos=True)
    for ep in episodes:
        name = ep["episode"]
        lowdim = np.load(os.path.join(args.demo_dir, f"{name}.npz"))
        front = read_video(os.path.join(args.demo_dir, f"{name}_front.mp4"))
        wrist = read_video(os.path.join(args.demo_dir, f"{name}_wrist.mp4"))
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
