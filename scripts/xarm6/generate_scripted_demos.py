# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted xArm6 demos of "pick up a cube and place it into the (already open) box".

Each attempt randomizes the cubes and the box, checks the box lid is open (the "human opened the
box" precondition from the task graph), then follows TCP waypoints with per-step IK:

    above cube -> down to grasp -> close -> lift -> above box -> lower -> open -> retreat

Layout, tabletop texture, lighting and camera poses are randomized per attempt (--no-domain-rand
to turn the visual part off). Only attempts whose cube ends up inside the box are kept. Recorded at the control rate in the
UFACTORY LeRobot layout (see simfoundry/robots/xarm6_env.py): front/wrist video, state and
absolute joint + gripper actions. Output, one set per kept episode:

    <out-dir>/episode_XXXX_front.mp4, episode_XXXX_wrist.mp4, episode_XXXX.npz (state, action),
    <out-dir>/episodes.jsonl (prompt, cube, seed, ...)

Convert with scripts/xarm6/convert_demos_to_lerobot.py.

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/generate_scripted_demos.py \
        --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json \
        --task cube2_pick_place_cube_in_box --out-dir Data/cube2_scene/xarm6_demos --n-demos 50

--real-layout switches to the real-rig setup (simfoundry/robots/xarm6_real.py): box and cubes at
real size, layouts and start/end poses drawn around the recorded real episodes, calibrated
cameras, and episodes written in the real recorder's format, <out-dir>/epXXXXXX/episode.h5
(convert with scripts/xarm6/convert_real_h5_to_lerobot.py, the same converter as the real data):

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/generate_scripted_demos.py \
        --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
        --out-dir Data/cube2_scene/xarm6_demos_real --n-demos 50 \
        --real-layout scripts/cfg/real/xarm6_real_layout.json --action-freq 20 --speed 0.08 --gripper-rate 0.059
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
    ap.add_argument("--no-domain-rand", action="store_true",
                    help="keep the nominal white floor / lighting / cameras (default: randomize tabletop wood "
                         "texture, lighting and camera pose per episode; see xarm6_randomize.py)")
    ap.add_argument("--p-wood", type=float, default=0.8, help="probability the tabletop is wood (else plain colour)")
    ap.add_argument("--box-xy-range", type=float, default=0.04)
    ap.add_argument("--speed", type=float, default=0.15, help="nominal TCP speed, m/s")
    ap.add_argument("--settle-tol", type=float, default=0.01,
                    help="rad; at each waypoint, hold until every arm joint is this close to its target")
    ap.add_argument("--save-failures", action="store_true", help="also keep failed attempts' videos for debugging")
    ap.add_argument("--real-layout", default=None,
                    help="scripts/cfg/real/xarm6_real_layout.json: real-rig mode (see module docstring)")
    ap.add_argument("--gripper-rate", type=float, default=0.0,
                    help="max gripper command change per step (0..1 units); 0 = jump. The real teleop "
                         "ramps 50 raw/step = 0.059")
    args = ap.parse_args()
    real_mode = args.real_layout is not None
    if real_mode and args.sensors == "sim_xarm6_front_wrist":
        args.sensors = "sim_xarm6_real"
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
    scene_json = args.scene_json
    if real_mode:
        from simfoundry.robots.xarm6_real import (
            RealEpisodeRecorder, RealLayoutSampler, RealScene, command_to_raw, gripper_raw, make_real_scene_json,
        )

        scene_json = make_real_scene_json(args.scene_json, os.path.join(args.out_dir, "real_scene.json"),
                                          box_category=args.box_category)
        kin6 = XArm6Kinematics(tip_link="link6")
    env = make_xarm6_env(scene_json, args.task, args.out_dir, sensors_cfg=args.sensors,
                         action_freq=args.action_freq, max_steps=100000, grasping_mode="physical")
    robot = env.robots[0]
    lid = articulated_joint(env, args.box_category, args.lid_joint)
    cubes = [next(iter(env.scene.object_registry("category", c))) for c in args.cubes.split(",")]
    box = next(iter(env.scene.object_registry("category", args.box_category)))
    latch = GraspLatch(robot, cubes)
    randomizer = None
    if not args.no_domain_rand:
        from simfoundry.robots.xarm6_randomize import DomainRandomizer

        # Calibrated cameras in real mode: jitter only by the calibration error.
        jit = {}
        if real_mode:
            from simfoundry.robots.xarm6_real import LIGHT_WOODS

            jit = dict(front_jitter=(0.01, 1.0), wrist_jitter=(0.003, 1.0), focal_jitter=0.02, wood_species=LIGHT_WOODS)
        randomizer = DomainRandomizer(env, rng, p_wood=args.p_wood, **jit)
    home = robot._default_joint_pos[:6].cpu().numpy()
    hz = args.action_freq / 30.0  # hold counts below were tuned at 30 Hz
    if real_mode:
        reset_xarm6_env(env)
        real = RealScene(env, box, cubes, box_link=args.box_link)
        sampler = RealLayoutSampler(args.real_layout, rng, box_size=real.box_size)
        if randomizer is None:
            print("[demo] real mode without domain randomization: white floor", flush=True)

    meta_path = os.path.join(args.out_dir, "episodes.jsonl")
    n_kept = sum(1 for _ in open(meta_path)) if os.path.exists(meta_path) else 0
    attempt = 0
    while n_kept < args.n_demos and attempt < max_attempts:
        attempt += 1
        cube = cubes[attempt % len(cubes)]
        if real_mode:
            color = CUBE_COLORS.get(cube.category, cube.category)
            layout = sampler.sample(color)
            hold = reset_xarm6_env(env, home=np.asarray(layout["start_q"]))
            real.place_box(layout["box_center"], layout["box_yaw"], layout["lid_deg"])
            real.place_cube(cube, layout["target_xy"], rng.uniform(-math.pi, math.pi))
            for other in cubes:
                if other is not cube:
                    real.place_cube(other, layout["other_xy"], rng.uniform(-math.pi, math.pi))
        else:
            hold = reset_xarm6_env(env, home=home + rng.uniform(-0.05, 0.05, 6))
            randomize_layout(env, rng, cubes, box, args.cube_xy_range, args.box_xy_range, min_sep=0.09)
        rand_info = randomizer.apply() if randomizer else None
        if real_mode:
            rand_info = dict(rand_info or {}, surroundings=real.add_surroundings(rng))
            if randomizer is not None:
                real.recolor_cubes(rng)
        for _ in range(20):
            env.step(hold)
        lid_pos = float(lid.get_state()[0])
        if lid_pos < args.min_lid_open:
            print(f"[demo] attempt {attempt}: lid at {lid_pos:.2f} rad, box not open -- skipped", flush=True)
            continue

        w2b = World2Base(robot)
        cube_pos, cube_quat = (x.cpu().numpy() for x in cube.get_position_orientation())
        c = w2b.pos(cube_pos)
        if real_mode and np.linalg.norm(c[:2] - layout["target_xy"]) > 0.03:
            print(f"[demo] attempt {attempt}: cube at {c[:2].round(3)}, layout {layout['target_xy']} -- skipped",
                  flush=True)
            continue
        cube_half = 0.5 * float((cube.aabb[1] - cube.aabb[0])[2])
        box_lo, box_hi = (x.cpu().numpy() for x in box.links[args.box_link].aabb)
        b = w2b.pos(0.5 * (box_lo + box_hi))
        rim = w2b.pos(box_hi)[2]
        if real_mode:
            # Drop anywhere over the box floor, like the real operator (release points spread over
            # the box, cube released 2-15 cm above the rim).
            bc, byaw = real.box_pose_b
            half_long, half_short = 0.5 * np.asarray(real.box_size)
            off = Rotation.from_euler("z", byaw).apply([rng.uniform(-1, 1) * (half_long - 0.07),
                                                       rng.uniform(-1, 1) * (half_short - 0.06), 0.0])
            b = np.array([bc[0] + off[0], bc[1] + off[1], b[2]])

        grip_rot = Rotation.from_euler("z", grasp_yaw(w2b.rot(cube_quat))) * R_DOWN
        place_rot = Rotation.from_euler("z", rng.uniform(-0.3, 0.3)) * R_DOWN
        jitter = lambda s: rng.uniform(-s, s, 3) * [1, 1, 0.5]  # noqa: E731
        # The open fingertips hang ~1 cm above eef_link and the linkage swings them down as it
        # closes, so the TCP goes 6 mm above the cube's bottom face: the closed pads cover most of
        # the cube without touching the floor (physical lift 12/12 in the grasp benchmark).
        grasp = c + [0.0, 0.0, -cube_half + 0.006] + jitter(0.003) * [1, 1, 0]
        drop_dz = rng.uniform(0.02, 0.10) if real_mode else rng.uniform(0.05, 0.08)
        drop = np.array([b[0], b[1], rim + drop_dz]) + jitter(0.0 if real_mode else 0.02) * [1, 1, 0]
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
        g_cmd = [0.0]  # ramped gripper command actually sent
        recorder = RealEpisodeRecorder(kin6, 1.0 / args.action_freq) if real_mode else None

        def step(q_cmd, g):
            if args.gripper_rate > 0:
                g_cmd[0] += float(np.clip(g - g_cmd[0], -args.gripper_rate, args.gripper_rate))
            else:
                g_cmd[0] = g
            imgs = xarm6_images(env)
            state = xarm6_state(robot)
            action = np.concatenate([q_cmd, [g_cmd[0]]]).astype(np.float32)
            if recorder is not None:
                dq = robot.get_joint_velocities()[robot.arm_control_idx[robot.default_arm]].cpu().numpy()
                lid_deg, box_shift, box_dyaw = real.disturbance()
                recorder.add(state[:6], dq, q_cmd, gripper_raw(robot), command_to_raw(g_cmd[0]),
                             imgs["front"], imgs["wrist"],
                             extras={"lid_angle_deg": lid_deg, "box_shift_m": box_shift, "box_dyaw_deg": box_dyaw})
            else:
                front.append(imgs["front"])
                wrist.append(imgs["wrist"])
            states.append(state)
            actions.append(action)
            env.step(th.from_numpy(action))
            if args.grasp == "latch":
                latch.update(g_cmd[0])

        if real_mode:  # the operator usually waits a moment before moving
            for _ in range(int(rng.integers(5, 30))):
                step(q, 0.0)

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
            ramp = int(math.ceil(1.0 / args.gripper_rate)) if (args.gripper_rate > 0 and g != grip) else 0
            grip = g
            for _ in range(int(round(hold_steps * hz)) + ramp):
                step(q, grip)
            cur = (target, rot)

        if ok and real_mode:
            # Real episodes end wherever the operator parks the arm (the next episode's start pose).
            end_q = np.asarray(layout["end_q"])
            n = max(int(args.action_freq * np.max(np.abs(end_q - q)) / (0.5 * speed / 0.08)), 1)
            q0 = q.copy()
            for s in min_jerk(n):
                q = q0 + s * (end_q - q0)
                step(q, 0.0)
            for _ in range(int(rng.integers(10, 40))):
                step(q, 0.0)
        if ok:
            for _ in range(20):
                env.step(th.from_numpy(np.concatenate([q, [0.0]]).astype(np.float32)))
                latch.update(0.0)
        cube_now = w2b.pos(cube.get_position_orientation()[0].cpu().numpy())
        lo, hi = w2b.pos(box_lo), w2b.pos(box_hi)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        if real_mode:  # the box may be turned: test in its own frame
            inside = real.inside_box(cube_now)
        else:
            inside = bool(np.all(cube_now[:2] > lo[:2]) and np.all(cube_now[:2] < hi[:2]) and cube_now[2] < hi[2])
        success = ok and inside
        tag = "OK " if success else ("IK " if not ok else "MISS")
        print(f"[demo] attempt {attempt} {tag} cube={cube.category} steps={len(actions)} lid={lid_pos:.2f} "
              f"kept={n_kept + success}/{args.n_demos}", flush=True)
        if not (success or args.save_failures):
            continue
        if real_mode:
            ex = recorder.extras
            contact = {"lid_max_change_deg": round(float(np.max(np.abs(np.asarray(ex["lid_angle_deg"]) - layout["lid_deg"]))), 2),
                       "box_max_shift_m": round(float(np.max(ex["box_shift_m"])), 4),
                       "box_max_dyaw_deg": round(float(np.max(np.abs(ex["box_dyaw_deg"]))), 2)}
            print(f"[demo]   lid/box disturbance {contact}", flush=True)
            name = f"ep{n_kept:06d}" if success else f"failed_{attempt:06d}"
            color = CUBE_COLORS.get(cube.category, cube.category.replace("_", " "))
            prompt = PROMPTS[rng.integers(len(PROMPTS))].format(color=color)
            os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
            recorder.write(os.path.join(args.out_dir, name, "episode.h5"),
                           {"prompt": prompt, "target_color": color, "source": "sim", "seed": args.seed,
                            "attempt": attempt, "layout": layout, "domain_rand": rand_info, "disturbance": contact})
            if success:
                with open(meta_path, "a") as f:
                    f.write(json.dumps({"episode": name, "cube": cube.category, "attempt": attempt, "seed": args.seed,
                                        "prompt": prompt, "length": len(recorder), "layout": layout,
                                        "disturbance": contact,
                                        "domain_rand": rand_info}) + "\n")
                n_kept += 1
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
                                    "length": len(actions), "lid_rad": lid_pos, "domain_rand": rand_info}) + "\n")
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
