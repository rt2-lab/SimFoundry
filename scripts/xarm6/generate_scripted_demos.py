# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted xArm6 demos of "pick up a cube and place it into the (already open) box".

Each attempt randomizes the cubes and the box, checks the box lid is open (the "human opened the
box" precondition from the task graph), then follows TCP waypoints with per-step IK:

    above cube -> down to grasp -> close -> lift -> above box -> lower -> open -> retreat

Only attempts whose cube ends up inside the box are kept. Recorded at the control rate in the
UFACTORY LeRobot layout (see simfoundry/robots/xarm6_env.py): front/wrist video, state and
absolute joint + gripper actions. Output, one set per kept episode:

    <out-dir>/episode_XXXX_front.mp4, episode_XXXX_wrist.mp4, episode_XXXX.npz (state, action),
    <out-dir>/episodes.jsonl (prompt, cube, seed, ...)

Convert with scripts/xarm6/convert_demos_to_lerobot.py.

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/generate_scripted_demos.py \
        --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json \
        --task cube2_pick_place_cube_in_box --out-dir Data/cube2_scene/xarm6_demos --n-demos 50
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation

CUBE_COLORS = {"blue_cube": "blue", "yellow_cube": "yellow"}
PROMPTS = (
    "Pick up the {color} cube and place it into the box.",
    "Put the {color} cube in the box.",
    "Place the {color} block into the box.",
)
# eef_link orientation at home in the base frame: z down, y (finger closing axis) along +y.
R_DOWN = Rotation.from_matrix(np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]))


def min_jerk(n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def grasp_yaw(cube_rot_base):
    """Yaw (about base z) of the gripper so its fingers close across two faces of the cube,
    folded into [-pi/4, pi/4] so the wrist turns as little as possible."""
    m = cube_rot_base.as_matrix()
    vertical = np.argmax(np.abs(m[2]))
    horizontal = m[:, (vertical + 1) % 3]
    # Fingers close along eef y, which is base +y rotated by the gripper yaw.
    yaw = math.atan2(horizontal[1], horizontal[0]) - math.pi / 2
    return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


class World2Base:
    def __init__(self, robot):
        pos, quat = robot.get_position_orientation()
        self.t = pos.cpu().numpy()
        self.r = Rotation.from_quat(quat.cpu().numpy())

    def pos(self, p):
        return self.r.inv().apply(np.asarray(p) - self.t)

    def rot(self, quat):
        return self.r.inv() * Rotation.from_quat(np.asarray(quat))


def randomize_layout(env, rng, cubes, box, xy_range, box_xy_range, min_sep):
    """Rejection-sample cube / box xy offsets around their reconstructed poses, plus a random cube
    yaw. Poses are relative to @env's scene-JSON poses, cached on first call."""
    import torch as th

    if not hasattr(randomize_layout, "nominal"):
        randomize_layout.nominal = {o.name: [x.clone() for x in o.get_position_orientation()] for o in cubes + [box]}
    nominal = randomize_layout.nominal
    for _ in range(100):
        box_off = rng.uniform(-box_xy_range, box_xy_range, 2)
        offs = {c.name: rng.uniform(-xy_range, xy_range, 2) for c in cubes}
        box_xy = nominal[box.name][0][:2].numpy() + box_off
        box_half = 0.5 * (box.aabb[1] - box.aabb[0])[:2].cpu().numpy()
        xys = [nominal[c.name][0][:2].numpy() + offs[c.name] for c in cubes]
        clear_of_box = all(np.any(np.abs(xy - box_xy) > box_half + min_sep / 2) for xy in xys)
        apart = all(np.linalg.norm(a - b) > min_sep for i, a in enumerate(xys) for b in xys[i + 1:])
        if clear_of_box and apart:
            break
    pos, ori = nominal[box.name]
    box.set_position_orientation(position=pos + th.tensor([*box_off, 0.0]), orientation=ori)
    for c in cubes:
        pos, ori = nominal[c.name]
        yaw = Rotation.from_euler("z", rng.uniform(-math.pi, math.pi))
        new_ori = th.tensor((yaw * Rotation.from_quat(ori.numpy())).as_quat(), dtype=th.float32)
        c.set_position_orientation(position=pos + th.tensor([*offs[c.name], 0.01]), orientation=new_ori)
    for o in cubes + [box]:
        o.keep_still()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-demos", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=None, help="default 3 x n-demos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-freq", type=int, default=30)
    ap.add_argument("--sensors", default="sim_xarm6_front_wrist")
    ap.add_argument("--grasp", default="physical", choices=["physical", "latch"],
                    help="physical: friction grasp (needs the force-drive asset from fix_xarm6_usd.py); "
                         "latch: geometric grasp constraint (GraspLatch)")
    ap.add_argument("--cubes", default="blue_cube,yellow_cube")
    ap.add_argument("--box-category", default="toolbox")
    ap.add_argument("--box-link", default="toolbox_base_link", help="box body (its AABB excludes the lid)")
    ap.add_argument("--lid-joint", default="toolbox_base_link_to_lid_link")
    ap.add_argument("--min-lid-open", type=float, default=1.0)
    ap.add_argument("--cube-xy-range", type=float, default=0.08)
    ap.add_argument("--box-xy-range", type=float, default=0.04)
    ap.add_argument("--speed", type=float, default=0.15, help="nominal TCP speed, m/s")
    ap.add_argument("--settle-tol", type=float, default=0.01,
                    help="rad; at each waypoint, hold until every arm joint is this close to its target")
    ap.add_argument("--save-failures", action="store_true", help="also keep failed attempts' videos for debugging")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    max_attempts = args.max_attempts or 3 * args.n_demos

    import imageio
    import torch as th

    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_CCD = True

    from simfoundry.robots.xarm6_env import (
        GraspLatch, articulated_joint, make_xarm6_env, reset_xarm6_env, xarm6_images, xarm6_state,
    )
    from simfoundry.robots.xarm6_kinematics import XArm6Kinematics

    rng = np.random.default_rng(args.seed)
    kin = XArm6Kinematics()
    env = make_xarm6_env(args.scene_json, args.task, args.out_dir, sensors_cfg=args.sensors,
                         action_freq=args.action_freq, max_steps=100000, grasping_mode="physical")
    robot = env.robots[0]
    lid = articulated_joint(env, args.box_category, args.lid_joint)
    cubes = [next(iter(env.scene.object_registry("category", c))) for c in args.cubes.split(",")]
    box = next(iter(env.scene.object_registry("category", args.box_category)))
    latch = GraspLatch(robot, cubes)
    home = robot._default_joint_pos[:6].cpu().numpy()

    meta_path = os.path.join(args.out_dir, "episodes.jsonl")
    n_kept = sum(1 for _ in open(meta_path)) if os.path.exists(meta_path) else 0
    attempt = 0
    while n_kept < args.n_demos and attempt < max_attempts:
        attempt += 1
        hold = reset_xarm6_env(env, home=home + rng.uniform(-0.05, 0.05, 6))
        randomize_layout(env, rng, cubes, box, args.cube_xy_range, args.box_xy_range, min_sep=0.09)
        for _ in range(20):
            env.step(hold)
        lid_pos = float(lid.get_state()[0])
        if lid_pos < args.min_lid_open:
            print(f"[demo] attempt {attempt}: lid at {lid_pos:.2f} rad, box not open -- skipped", flush=True)
            continue

        cube = cubes[attempt % len(cubes)]
        w2b = World2Base(robot)
        cube_pos, cube_quat = (x.cpu().numpy() for x in cube.get_position_orientation())
        c = w2b.pos(cube_pos)
        cube_half = 0.5 * float((cube.aabb[1] - cube.aabb[0])[2])
        box_lo, box_hi = (x.cpu().numpy() for x in box.links[args.box_link].aabb)
        b = w2b.pos(0.5 * (box_lo + box_hi))
        rim = w2b.pos(box_hi)[2]

        grip_rot = Rotation.from_euler("z", grasp_yaw(w2b.rot(cube_quat))) * R_DOWN
        place_rot = Rotation.from_euler("z", rng.uniform(-0.3, 0.3)) * R_DOWN
        jitter = lambda s: rng.uniform(-s, s, 3) * [1, 1, 0.5]  # noqa: E731
        # The open fingertips hang ~1 cm above eef_link and the linkage swings them down as it
        # closes, so the TCP goes 6 mm above the cube's bottom face: the closed pads cover most of
        # the cube without touching the floor (physical lift 12/12 in the grasp benchmark).
        grasp = c + [0.0, 0.0, -cube_half + 0.006] + jitter(0.003) * [1, 1, 0]
        drop = np.array([b[0], b[1], rim + rng.uniform(0.05, 0.08)]) + jitter(0.02) * [1, 1, 0]
        # (TCP target, orientation, gripper command, hold steps after reaching it)
        waypoints = [
            (c + [0, 0, rng.uniform(0.08, 0.12)] + jitter(0.01), grip_rot, 0.0, 0),
            (grasp, grip_rot, 0.0, 3),
            (grasp, grip_rot, 1.0, 25),
            (c + [0, 0, rng.uniform(0.12, 0.18)], grip_rot, 1.0, 0),
            (drop + [0, 0, rng.uniform(0.04, 0.08)], place_rot, 1.0, 0),
            (drop, place_rot, 1.0, 3),
            (drop, place_rot, 0.0, 12),
            (drop + [0, 0, 0.10], place_rot, 0.0, 10),
        ]

        q = xarm6_state(robot)[:6].astype(float)
        pos, quat = kin.fk_pose(q)
        cur = (pos, Rotation.from_quat(quat))
        grip, ok = 0.0, True
        states, actions, front, wrist = [], [], [], []
        speed = args.speed * rng.uniform(0.8, 1.2)

        def step(q_cmd, g):
            imgs = xarm6_images(env)
            front.append(imgs["front"])
            wrist.append(imgs["wrist"])
            states.append(xarm6_state(robot))
            action = np.concatenate([q_cmd, [g]]).astype(np.float32)
            actions.append(action)
            env.step(th.from_numpy(action))
            if args.grasp == "latch":
                latch.update(g)

        for target, rot, g, hold_steps in waypoints:
            dist = np.linalg.norm(target - cur[0])
            ang = (rot * cur[1].inv()).magnitude()
            n = max(int(args.action_freq * max(dist / speed, ang / 1.5, 0.3)), 1) if g == grip else 1
            for s in min_jerk(n):
                p = cur[0] + s * (target - cur[0])
                q, ok_ik = kin.ik(p, _slerp(cur[1].as_quat(), rot.as_quat(), s), q)
                if not ok_ik:
                    ok = False
                    break
                step(q, grip)
            if not ok:
                break
            # The PD drive lags the min-jerk profile; let it catch up before the gripper acts.
            for _ in range(30):
                if np.max(np.abs(xarm6_state(robot)[:6] - q)) < args.settle_tol:
                    break
                step(q, grip)
            if g != grip and g == 1.0:
                tcp = kin.fk_pose(xarm6_state(robot)[:6])[0]
                cube_b = w2b.pos(cube.get_position_orientation()[0].cpu().numpy())
                print(f"[demo]   close at tcp={tcp.round(3)} cube={cube_b.round(3)} "
                      f"err={np.round(tcp - cube_b, 3)}", flush=True)
            grip = g
            for _ in range(hold_steps):
                step(q, grip)
            cur = (target, rot)

        if ok:
            for _ in range(20):
                env.step(th.from_numpy(np.concatenate([q, [0.0]]).astype(np.float32)))
                latch.update(0.0)
        cube_now = w2b.pos(cube.get_position_orientation()[0].cpu().numpy())
        lo, hi = w2b.pos(box_lo), w2b.pos(box_hi)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        inside = bool(np.all(cube_now[:2] > lo[:2]) and np.all(cube_now[:2] < hi[:2]) and cube_now[2] < hi[2])
        success = ok and inside
        tag = "OK " if success else ("IK " if not ok else "MISS")
        print(f"[demo] attempt {attempt} {tag} cube={cube.category} steps={len(actions)} lid={lid_pos:.2f} "
              f"kept={n_kept + success}/{args.n_demos}", flush=True)
        if not (success or args.save_failures):
            continue

        name = f"episode_{n_kept:04d}" if success else f"failed_{attempt:04d}"
        for cam, frames in (("front", front), ("wrist", wrist)):
            imageio.mimsave(os.path.join(args.out_dir, f"{name}_{cam}.mp4"), frames, fps=args.action_freq,
                            quality=9, macro_block_size=1)
        np.savez(os.path.join(args.out_dir, f"{name}.npz"), state=np.stack(states), action=np.stack(actions))
        if success:
            color = CUBE_COLORS.get(cube.category, cube.category.replace("_", " "))
            with open(meta_path, "a") as f:
                f.write(json.dumps({"episode": name, "cube": cube.category, "attempt": attempt, "seed": args.seed,
                                    "prompt": PROMPTS[rng.integers(len(PROMPTS))].format(color=color),
                                    "length": len(actions), "lid_rad": lid_pos}) + "\n")
            n_kept += 1

    print(f"[demo] kept {n_kept} demos from {attempt} attempts -> {args.out_dir}")
    og.shutdown()


def _slerp(q0, q1, s):
    q0, q1 = np.asarray(q0), np.asarray(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - s) * theta) * q0 + math.sin(s * theta) * q1) / math.sin(theta)


if __name__ == "__main__":
    main()
