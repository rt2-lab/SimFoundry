# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack xArm6 episodes in the real recorder's format (<dir>/epXXXXXX/episode.h5) into a LeRobot v3
dataset. One converter for both sources, so real and sim data share every convention:

- real: the 2026-10 autoalloc recordings (no prompt stored; the target colour comes from
  --real-colors, the prompt from the first template);
- sim: generate_scripted_demos.py --real-layout (prompt and colour in the file's attrs).

Mapping (same feature names as convert_demos_to_lerobot.py):

    observation.images.front  <- mount_image_rgb_compressed   (fixed D435i)
    observation.images.wrist  <- gripper_image_rgb_compressed (wrist D435i)
    observation.state         <- [robot_current_joint_angles, 1 - gripper_actual_position / 850]
    action                    <- [robot_desired_joint_angles, 1 - gripper_commanded_position / 850]

The gripper becomes 0 = open .. 1 = closed, as in all earlier SimFoundry xArm6 datasets.
Runs in the LeRobot venv:

    .cache/pi05-serve-venv/bin/python scripts/xarm6/convert_real_h5_to_lerobot.py \\
        --src /home/tingting/Documents/data/autoalloc_real --exclude 8,10,45 \\
        --root Data/cube2_scene/lerobot/xarm6_real --repo-id local/xarm6_real --relative-action-chunk 50
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import h5py
import numpy as np

NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "gripper.pos"]
DROID_NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "J7.pad", "gripper.pos"]
PROMPT = "Pick up the {color} cube and place it into the box."
GRIPPER_RAW_OPEN = 850.0


def parse_ranges(spec):
    """'0-27:blue,28-50:yellow' -> {0: 'blue', ..., 50: 'yellow'}."""
    out = {}
    for part in spec.split(","):
        rng, color = part.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        out.update({i: color for i in range(lo, hi + 1)})
    return out


def decode(buf):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="directories holding ep*/episode.h5")
    ap.add_argument("--exclude", default="", help="real episode indices to skip, e.g. 8,10,45 (sim files are kept)")
    ap.add_argument("--real-colors", default="0-27:blue,28-50:yellow",
                    help="target colour by episode index for files without a prompt attribute")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--root", required=True, help="output dataset directory (must not exist)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--relative-action-chunk", type=int, default=None)
    ap.add_argument("--droid-layout", action="store_true")
    ap.add_argument("--max-lid-change", type=float, default=2.0,
                    help="deg; skip sim episodes whose arm moved the lid more than this (attrs['disturbance'])")
    ap.add_argument("--max-box-shift", type=float, default=0.01,
                    help="m; skip sim episodes whose arm pushed the box further than this")
    ap.add_argument("--workers", type=int, default=1,
                    help="convert in this many parallel shards, then merge (video encoding is the bottleneck)")
    ap.add_argument("--speedup", action="store_true",
                    help="keep only the frames demo_speedup.speedup_indices selects (drops pauses and the tail, "
                         "plays free-space motion 2x); the dataset stays 20 fps, so the demos become ~2x faster")
    ap.add_argument("--files-from", default=None, help=argparse.SUPPRESS)  # internal: one shard's file list
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    exclude = {int(x) for x in args.exclude.split(",") if x}
    colors = parse_ranges(args.real_colors)
    if args.files_from:
        files = [line.strip() for line in open(args.files_from) if line.strip()]
    else:
        files, skipped = [], []
        for d in args.src:
            for p in sorted(glob.glob(os.path.join(d, "ep*", "episode.h5"))):
                with h5py.File(p, "r") as f:
                    is_real = "source" not in f.attrs
                    dist = json.loads(f.attrs.get("disturbance", "{}"))
                if is_real and int(os.path.basename(os.path.dirname(p))[2:]) in exclude:
                    continue
                if dist.get("lid_max_change_deg", 0) > args.max_lid_change or dist.get("box_max_shift_m", 0) > args.max_box_shift:
                    skipped.append((p, dist))  # the scripted arm bumped the lid / box
                    continue
                files.append(p)
        for p, dist in skipped:
            print(f"[convert] skip {p}: {dist}", flush=True)
        files = files[:args.max_episodes]
    assert files, f"no episodes under {args.src}"
    if args.workers > 1 and not args.files_from:
        return convert_sharded(args, files)

    names = DROID_NAMES if args.droid_layout else NAMES
    with h5py.File(files[0], "r") as f:
        h, w = decode(f["mount_image_rgb_compressed"][0]).shape[:2]
    image_feature = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.images.front": image_feature,
        "observation.images.wrist": dict(image_feature),
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features, root=args.root,
                               robot_type="UFACTORY Robot", use_videos=True)
    index = []
    for p in files:
        ep = int(os.path.basename(os.path.dirname(p))[2:])
        with h5py.File(p, "r") as f:
            prompt = f.attrs.get("prompt")
            if prompt is None:
                prompt = PROMPT.format(color=colors[ep])
            grip = 1.0 - f["gripper_actual_position"][:] / GRIPPER_RAW_OPEN
            grip_cmd = 1.0 - f["gripper_commanded_position"][:] / GRIPPER_RAW_OPEN
            state = np.concatenate([f["robot_current_joint_angles"][:], grip[:, None]], axis=1)
            action = np.concatenate([f["robot_desired_joint_angles"][:], grip_cmd[:, None]], axis=1)
            state, action = np.clip(state, -10, 10).astype(np.float32), action.astype(np.float32)
            n_orig = len(state)
            idx = np.arange(n_orig)
            if args.speedup:
                from demo_speedup import speedup_indices

                idx, _ = speedup_indices(f["robot_desired_pose"][:, :3], f["robot_desired_joint_angles"][:],
                                         f["gripper_commanded_position"][:], f["gripper_actual_position"][:],
                                         fps=args.fps)
                state, action = state[idx], action[idx]
            state[:, 6], action[:, 6] = np.clip(state[:, 6], 0, 1), np.clip(action[:, 6], 0, 1)
            if args.droid_layout:
                pad = np.zeros((len(state), 1), np.float32)
                state = np.concatenate([state[:, :6], pad, state[:, 6:]], axis=1)
                action = np.concatenate([action[:, :6], pad, action[:, 6:]], axis=1)
            front, wrist = f["mount_image_rgb_compressed"], f["gripper_image_rgb_compressed"]
            for t, src_t in enumerate(idx):
                ds.add_frame({"action": action[t], "observation.state": state[t],
                              "observation.images.front": decode(front[src_t]),
                              "observation.images.wrist": decode(wrist[src_t]), "task": prompt})
            source = f.attrs.get("source", "real")
        ds.save_episode()
        index.append({"file": p, "source": source, "prompt": prompt, "length": len(state), "orig_length": n_orig})
        print(f"[convert] {p}: {len(state)} frames, {source}, task={prompt!r}", flush=True)
    ds.finalize()
    with open(os.path.join(args.root, "meta", "sources.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in index)
    print(f"[convert] {len(files)} episodes -> {args.root}")
    if args.relative_action_chunk and not args.files_from:
        from lerobot.datasets.dataset_tools import recompute_stats

        ds = LeRobotDataset(args.repo_id, root=args.root)
        recompute_stats(ds, relative_action=True, relative_exclude_joints=["gripper"],
                        chunk_size=args.relative_action_chunk)
        print(f"[convert] action stats recomputed in relative space (chunk {args.relative_action_chunk})")


def convert_sharded(args, files):
    """Run this script on contiguous slices of @files in parallel, merge the shards in order, then
    recompute stats once on the merged dataset."""
    from lerobot.datasets.dataset_tools import merge_datasets, recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shard_dir = args.root.rstrip("/") + "_shards"
    os.makedirs(shard_dir, exist_ok=False)
    chunks = [list(c) for c in np.array_split(np.array(files), args.workers) if len(c)]
    procs, roots = [], []
    for i, chunk in enumerate(chunks):
        lst, root = os.path.join(shard_dir, f"shard{i}.txt"), os.path.join(shard_dir, f"shard{i}")
        with open(lst, "w") as f:
            f.writelines(p + "\n" for p in chunk)
        cmd = [sys.executable, __file__, "--src", *args.src, "--files-from", lst, "--root", root,
               "--repo-id", f"{args.repo_id}_shard{i}", "--fps", str(args.fps), "--real-colors", args.real_colors]
        if args.droid_layout:
            cmd.append("--droid-layout")
        if args.speedup:
            cmd.append("--speedup")
        procs.append(subprocess.Popen(cmd, stdout=open(root + ".log", "w"), stderr=subprocess.STDOUT))
        roots.append(root)
    codes = [p.wait() for p in procs]
    assert not any(codes), f"shard exit codes {codes}; logs in {shard_dir}"
    shards = [LeRobotDataset(f"{args.repo_id}_shard{i}", root=r) for i, r in enumerate(roots)]
    merge_datasets(shards, output_repo_id=args.repo_id, output_dir=args.root)
    with open(os.path.join(args.root, "meta", "sources.jsonl"), "w") as out:
        for r in roots:
            out.write(open(os.path.join(r, "meta", "sources.jsonl")).read())
    print(f"[convert] merged {len(roots)} shards, {len(files)} episodes -> {args.root}", flush=True)
    if args.relative_action_chunk:
        ds = LeRobotDataset(args.repo_id, root=args.root)
        recompute_stats(ds, relative_action=True, relative_exclude_joints=["gripper"],
                        chunk_size=args.relative_action_chunk)
        print(f"[convert] action stats recomputed in relative space (chunk {args.relative_action_chunk})")
    shutil.rmtree(shard_dir)


if __name__ == "__main__":
    main()
