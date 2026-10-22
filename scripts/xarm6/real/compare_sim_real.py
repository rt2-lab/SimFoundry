# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the real-aligned sim at recorded real states and put the renders next to the real frames.

For each requested episode and step: the sim arm is set to the recorded joint angles, the box and
cubes to that episode's layout (scripts/cfg/real/xarm6_real_layout.json), and both cameras
(sim_xarm6_real) are rendered. Output: one PNG per (episode, step) with real | sim | blend rows for
the mount and wrist cameras.

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/real/compare_sim_real.py \
        --data /home/tingting/Documents/data/autoalloc_real --episodes 0,20,40 --out-dir /tmp/sim_real
"""

import argparse
import json
import os

import cv2
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", default="0,20,40")
    ap.add_argument("--steps", default="0,grasp", help="comma list of step indices or 'grasp'")
    ap.add_argument("--layout", default="scripts/cfg/real/xarm6_real_layout.json")
    ap.add_argument("--scene-json", default="Data/cube2_scene/s14_og/reconstructed_og_scene.json")
    ap.add_argument("--task", default="cube2_pick_place_cube_in_box")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--box-scale", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import omnigibson as og
    import torch as th

    from simfoundry.robots.xarm6_env import make_xarm6_env, reset_xarm6_env, xarm6_images
    from simfoundry.robots.xarm6_randomize import DomainRandomizer
    from simfoundry.robots.xarm6_real import BOX_SCALE, RealScene, make_real_scene_json

    scene_json = make_real_scene_json(args.scene_json, os.path.join(args.out_dir, "real_scene.json"),
                                      box_scale=args.box_scale or BOX_SCALE)
    env = make_xarm6_env(scene_json, args.task, args.out_dir, sensors_cfg="sim_xarm6_real", action_freq=20)
    robot = env.robots[0]
    box = next(iter(env.scene.object_registry("category", "toolbox")))
    cubes = {c: next(iter(env.scene.object_registry("category", f"{c}_cube"))) for c in ("blue", "yellow")}
    reset_xarm6_env(env)
    real = RealScene(env, box, cubes.values())
    from simfoundry.robots.xarm6_real import LIGHT_WOODS

    rand = DomainRandomizer(env, np.random.default_rng(0), p_wood=1.0, cam_jitter=False, light_jitter=False,
                            wood_species=LIGHT_WOODS)
    rand.apply()
    real.add_surroundings()
    print("[compare] sim box footprint", np.round(real.box_size, 3), "height", round(2 * real.box_half[2], 3), flush=True)
    layout = {e["ep"]: e for e in json.load(open(args.layout))["episodes"]}

    for ep in [int(x) for x in args.episodes.split(",")]:
        e = layout[ep]
        f = h5py.File(os.path.join(args.data, f"ep{ep:06d}", "episode.h5"), "r")
        J = f["robot_current_joint_angles"][:]
        ga = f["gripper_actual_position"][:]
        for s in args.steps.split(","):
            t = e["grasp_step"] - 15 if s == "grasp" else int(s)
            hold = reset_xarm6_env(env, home=J[t])
            real.place_box(e["box"]["center"], e["box"]["yaw"])
            other = "yellow" if e["target_color"] == "blue" else "blue"
            real.place_cube(cubes[e["target_color"]], e["target_cube"][:2], 0.0)
            if e.get("other_cube"):
                real.place_cube(cubes[other], e["other_cube"], 0.0)
            else:
                real.place_cube(cubes[other], [0.0, 0.9], 0.0)  # out of view
            g = float(np.clip(1.0 - ga[t] / 850.0, 0, 1))
            hold = th.cat([th.as_tensor(J[t], dtype=th.float32), th.tensor([g])])
            for _ in range(15):
                env.step(hold)
            sim = xarm6_images(env)
            rows = []
            for cam_real, cam_sim in (("mount", "front"), ("gripper", "wrist")):
                r = cv2.imdecode(np.frombuffer(bytes(f[f"{cam_real}_image_rgb_compressed"][t]), np.uint8), cv2.IMREAD_COLOR)
                si = cv2.cvtColor(sim[cam_sim], cv2.COLOR_RGB2BGR)
                rows.append(np.hstack([r, si, cv2.addWeighted(r, 0.5, si, 0.5, 0)]))
            out = os.path.join(args.out_dir, f"ep{ep:02d}_t{t:03d}.png")
            cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
            print("[compare] wrote", out, flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
