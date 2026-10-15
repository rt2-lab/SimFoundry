# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop eval of an xArm6 pi05 checkpoint (UFACTORY LeRobot layout) in a reconstructed scene.

Talks to an already-running server (start it first, e.g. in tmux):

    .cache/pi05-serve-venv/bin/python simfoundry/policies/pi05_serve/server.py \
        --checkpoint checkpoints/ys903_pi05_xarm6 --device cuda --port 8420

then:

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
        --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json \
        --task cube2_pick_place_cube_in_box --out-dir Data/cube2_scene/s15_eval/xarm6_ys903 \
        --prompt "Pick up the blue block and drop it into the box." --episodes 5

Writes per-episode videos (front | wrist) and results.json.
"""

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--action-freq", type=int, default=30)
    ap.add_argument("--execute-horizon", type=int, default=10, help="checkpoint's n_action_steps")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--sensors", default="sim_xarm6_front_wrist")
    ap.add_argument("--camera-keys", default="ufactory", choices=["ufactory", "droid"],
                    help="names the checkpoint expects for the front / wrist views")
    ap.add_argument("--grasp", default="physical", choices=["physical", "latch"],
                    help="physical: friction grasp (needs the force-drive asset from fix_xarm6_usd.py); "
                         "latch: geometric grasp constraint (GraspLatch)")
    ap.add_argument("--box-category", default="toolbox")
    ap.add_argument("--lid-joint", default="toolbox_base_link_to_lid_link")
    ap.add_argument("--min-lid-open", type=float, default=1.0, help="rad; below this the box counts as closed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import imageio
    import torch as th

    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_CCD = True

    from simfoundry.policies.pi05_lerobot import PI05XArm6Client
    from simfoundry.robots.xarm6_env import (
        GraspLatch, articulated_joint, make_xarm6_env, reset_xarm6_env, xarm6_images, xarm6_state,
    )

    n_steps = int(args.timeout_s * args.action_freq)
    env = make_xarm6_env(args.scene_json, args.task, args.out_dir, sensors_cfg=args.sensors,
                         action_freq=args.action_freq, max_steps=n_steps + 100,
                         grasping_mode="physical")
    robot = env.robots[0]
    lid = articulated_joint(env, args.box_category, args.lid_joint)
    # Same grasp mechanics the scripted demos were recorded with (see GraspLatch).
    latch = GraspLatch(robot, [o for o in env.scene.objects if o.category.endswith("_cube")])
    policy = PI05XArm6Client(checkpoint_path=None, host=args.host, port=args.port, external_server=True,
                             camera_keys=args.camera_keys)

    results = []
    for ep in range(args.episodes):
        reset_xarm6_env(env)
        lid_pos = float(lid.get_state()[0])
        assert lid_pos > args.min_lid_open, f"lid at {lid_pos:.3f} rad after reset; the box must start open"
        policy.reset()
        frames, success, t0 = [], False, time.time()
        milestones, step = {}, 0
        imgs = xarm6_images(env)
        terminated = truncated = False
        while step < n_steps and not success:
            out = policy.infer({**imgs, "state": xarm6_state(robot)}, args.prompt)
            for a in out["action"][: args.execute_horizon]:
                action = th.from_numpy(a.astype(np.float32))
                action[6] = 1.0 if action[6] > 0.5 else 0.0  # training targets are binary
                _, _, terminated, truncated, info = env.step(action)
                if args.grasp == "latch":
                    latch.update(float(action[6]))
                step += 1
                milestones = info.get("milestones", milestones)
                success = bool(env.task.success)
                imgs = xarm6_images(env)
                frames.append(np.concatenate([imgs["front"], imgs["wrist"]], axis=1))
                if success or terminated or truncated or step >= n_steps:
                    break
            if terminated or truncated:
                break
        final_lid = float(lid.get_state()[0])
        imageio.mimsave(f"{args.out_dir}/episode_{ep:02d}.mp4", frames, fps=args.action_freq)
        result = {"episode": ep, "success": success, "steps": step, "lid_start": lid_pos,
                  "lid_end": final_lid, "milestones": milestones, "wall_s": round(time.time() - t0, 1)}
        results.append(result)
        print(f"[eval] {json.dumps(result)}", flush=True)

    summary = {"prompt": args.prompt, "scene_json": args.scene_json, "task": args.task,
               "success_rate": float(np.mean([r["success"] for r in results])), "episodes": results}
    with open(f"{args.out_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] success {sum(r['success'] for r in results)}/{len(results)} -> {args.out_dir}/results.json")
    og.shutdown()


if __name__ == "__main__":
    main()
