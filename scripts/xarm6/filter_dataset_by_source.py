# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep only the real or only the sim episodes of a mixed xArm6 LeRobot dataset.

The source dataset's meta/sources.jsonl (one line per episode, in episode order; written by
convert_real_h5_to_lerobot.py and carried over by make_eef_dataset.py) says which episodes are
real. The output gets the matching sources.jsonl and action stats recomputed in relative space
(LeRobot's episode deletion aggregates absolute stats, which relative-action training must not use).

    .cache/pi05-serve-venv/bin/python scripts/xarm6/filter_dataset_by_source.py \\
        --src Data/cube2_scene/lerobot/xarm6_real_sim_eef --root Data/cube2_scene/lerobot/xarm6_sim_eef --keep sim
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", required=True, help="output dataset directory (must not exist)")
    ap.add_argument("--keep", required=True, choices=["real", "sim"])
    ap.add_argument("--chunk", type=int, default=50, help="policy chunk_size, for relative action stats")
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import delete_episodes, recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with open(os.path.join(args.src, "meta", "sources.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    src = LeRobotDataset("local/" + os.path.basename(args.src.rstrip("/")), root=args.src)
    assert len(rows) == src.meta.total_episodes, f"sources.jsonl has {len(rows)} rows, dataset {src.meta.total_episodes} episodes"
    drop = [i for i, r in enumerate(rows) if r.get("source", "real") != args.keep]
    out_id = "local/" + os.path.basename(args.root.rstrip("/"))
    delete_episodes(src, drop, output_dir=args.root, repo_id=out_id)
    with open(os.path.join(args.root, "meta", "sources.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for i, r in enumerate(rows) if i not in set(drop))

    out = LeRobotDataset(out_id, root=args.root)
    recompute_stats(out, relative_action=True, relative_exclude_joints=["gripper"], chunk_size=args.chunk)
    print(f"[filter] kept {out.meta.total_episodes} {args.keep} episodes / {out.meta.total_frames} frames -> {args.root}")


if __name__ == "__main__":
    main()
