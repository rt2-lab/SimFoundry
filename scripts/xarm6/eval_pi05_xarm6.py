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

--real-layout evaluates in the real-rig setup the --real-layout demos were generated in
(simfoundry/robots/xarm6_real.py): real-size box / cubes, layouts and start poses drawn around the
recorded real episodes, calibrated cameras, and the gripper state in the real convention
(1 - opening / 85 mm). The target cube alternates blue / yellow and "{color}" in --prompt is filled in:

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/eval_pi05_xarm6.py \
        --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json --task cube2_pick_place_cube_in_box \
        --out-dir Data/cube2_scene/s15_eval/<name> --prompt "Pick up the {color} cube and place it into the box." \
        --real-layout scripts/cfg/real/xarm6_real_layout.json --action-freq 20 --randomize --episodes 10
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
    ap.add_argument("--grip-mode", default="hysteresis", choices=["hysteresis", "threshold", "raw"],
                    help="hysteresis: GripperHysteresis(--grip-close-at, --grip-open-at, --grip-steps); "
                         "threshold: close whenever the raw prediction >= --grip-threshold; "
                         "raw: send the prediction as is (training targets are ramps, e.g. real-format data)")
    ap.add_argument("--grip-threshold", type=float, default=0.5)
    ap.add_argument("--grip-close-at", type=float, default=0.8)
    ap.add_argument("--grip-open-at", type=float, default=0.3)
    ap.add_argument("--grip-steps", type=int, default=3, help="consecutive steps past a threshold to switch")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--sensors", default="sim_xarm6_front_wrist")
    ap.add_argument("--randomize", action="store_true",
                    help="per-episode tabletop / lighting / camera randomization as in the training data "
                         "(simfoundry/robots/xarm6_randomize.py); default: the nominal white floor")
    ap.add_argument("--wood-only", action="store_true", help="with --randomize: always a wood tabletop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-space", default="joint", choices=["joint", "eef"],
                    help="eef: checkpoint trained on make_eef_dataset.py data (10-D TCP action, 16-D state); "
                         "the predicted TCP target is turned into joints by IK seeded from the previous target")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="play the policy's actions this many times faster than the data (20 Hz): the env steps at "
                         "action-freq x speed and each env step consumes one action; the episode keeps --timeout-s x "
                         "action-freq steps, i.e. a timeout of --timeout-s / speed sim seconds")
    ap.add_argument("--track", default="position", choices=["position", "vel"],
                    help="position: action = joint position target (earlier evals). vel: emulate the real arm's "
                         "xarm_stack joint_vel_controller, target = q + clip(kp (q_des - q), +-vmax) dt, so joint "
                         "speed is capped like on the real robot")
    ap.add_argument("--kp", type=float, default=10.0)
    ap.add_argument("--vmax", type=float, default=3.14, help="rad/s; xArm6 hardware limit is pi")
    ap.add_argument("--droid-layout", action="store_true",
                    help="checkpoint trained on 8-dim [J1..J6, 0, gripper] state/action")
    ap.add_argument("--camera-keys", default="ufactory", choices=["ufactory", "droid"],
                    help="names the checkpoint expects for the front / wrist views")
    ap.add_argument("--grasp", default="physical", choices=["physical", "latch"],
                    help="physical: friction grasp (needs the force-drive asset from fix_xarm6_usd.py); "
                         "latch: geometric grasp constraint (GraspLatch)")
    ap.add_argument("--box-category", default="toolbox")
    ap.add_argument("--lid-joint", default="toolbox_base_link_to_lid_link")
    ap.add_argument("--box-link", default="toolbox_base_link", help="box body; its AABB excludes the lid")
    ap.add_argument("--target-cube", default="blue_cube", help="category the prompt asks for")
    ap.add_argument("--min-lid-open", type=float, default=1.0, help="rad; below this the box counts as closed")
    ap.add_argument("--real-layout", default=None, help="scripts/cfg/real/xarm6_real_layout.json: real-rig mode")
    args = ap.parse_args()
    real_mode = args.real_layout is not None
    if real_mode and args.sensors == "sim_xarm6_front_wrist":
        args.sensors = "sim_xarm6_real"
    os.makedirs(args.out_dir, exist_ok=True)

    import imageio
    import torch as th

    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_CCD = True

    from scipy.spatial.transform import Rotation

    from make_eef_dataset import tcp_pose9
    from simfoundry.policies.gripper_filter import GripperHysteresis
    from simfoundry.policies.pi05_lerobot import PI05XArm6Client
    from simfoundry.robots.xarm6_kinematics import XArm6Kinematics
    from simfoundry.robots.xarm6_env import (
        GraspLatch, articulated_joint, make_xarm6_env, reset_xarm6_env, xarm6_images, xarm6_state,
    )

    n_steps = int(args.timeout_s * args.action_freq)
    env_hz = int(round(args.action_freq * args.speed))
    physics_hz = env_hz * int(np.ceil(120 / env_hz))  # OmniGibson: a multiple of the action rate, >= the usual 120 Hz
    scene_json = args.scene_json
    if real_mode:
        from simfoundry.robots.xarm6_real import (
            LIGHT_WOODS, RealLayoutSampler, RealScene, gripper_raw, GRIPPER_RAW_OPEN, make_real_scene_json,
        )

        scene_json = make_real_scene_json(args.scene_json, os.path.join(args.out_dir, "real_scene.json"),
                                          box_category=args.box_category)
    env = make_xarm6_env(scene_json, args.task, args.out_dir, sensors_cfg=args.sensors,
                         action_freq=env_hz, max_steps=n_steps + int(100 * args.speed),
                         grasping_mode="physical", physics_freq=physics_hz)
    robot = env.robots[0]
    lid = articulated_joint(env, args.box_category, args.lid_joint)
    # Same grasp mechanics the scripted demos were recorded with (see GraspLatch).
    latch = GraspLatch(robot, [o for o in env.scene.objects if o.category.endswith("_cube")])
    box = next(iter(env.scene.object_registry("category", args.box_category)))
    target = next(iter(env.scene.object_registry("category", args.target_cube)))

    cubes = {c.category: c for c in env.scene.objects if c.category.endswith("_cube")}
    if real_mode:
        reset_xarm6_env(env)
        real = RealScene(env, box, cubes.values(), box_link=args.box_link)
        sampler = RealLayoutSampler(args.real_layout, np.random.default_rng(args.seed + 1), box_size=real.box_size)

    eef = args.action_space == "eef"
    kin_tcp = XArm6Kinematics(tip_link="eef_link") if eef else None
    ik_seed = {"fail": 0, "log": []}

    def state():
        s = xarm6_state(robot)
        if real_mode:  # the real-format datasets store 1 - opening / 85 mm
            s[6] = 1.0 - gripper_raw(robot) / GRIPPER_RAW_OPEN
        if eef:  # [TCP xyz, R cols 1-2, gripper, J1..J6], as in make_eef_dataset.py
            s = np.concatenate([tcp_pose9(kin_tcp, np.asarray(s[:6], dtype=float)[None])[0], s[6:7], s[:6]]).astype(np.float32)
        return s

    def joint_action(a):
        """Policy action -> the env's [J1..J6, gripper]. TCP targets go through IK seeded from the
        measured joints and the best-effort solution is used even when it did not fully converge,
        like xarm_stack's simple_ik (a weighted least-squares solve that always returns a joint
        target); non-converged solves are counted and the first few logged for diagnosis."""
        if not eef:
            return a
        q_meas = np.asarray(state()[10:16], dtype=float)
        b1 = a[3:6] / np.linalg.norm(a[3:6])
        b2 = a[6:9] - b1 @ a[6:9] * b1
        b2 /= np.linalg.norm(b2)
        r_tgt = np.stack([b1, b2, np.cross(b1, b2)], axis=1)
        q, ok, _ = kin_tcp.ik_nearest(np.asarray(a[:3], dtype=float), Rotation.from_matrix(r_tgt).as_quat(), q_meas)
        if not ok:
            ik_seed["fail"] += 1
            if len(ik_seed["log"]) < 3:
                T = kin_tcp.fk(q)
                T0 = kin_tcp.fk(q_meas)
                ik_seed["log"].append({
                    "target_xyz": np.round(a[:3], 3).tolist(), "tcp_now_xyz": np.round(T0[:3, 3], 3).tolist(),
                    "residual_mm": round(float(np.linalg.norm(T[:3, 3] - a[:3]) * 1000), 1),
                    "residual_deg": round(float(np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3].T @ r_tgt).as_rotvec()))), 1),
                    "rot_cols_norm": [round(float(np.linalg.norm(a[3:6])), 3), round(float(np.linalg.norm(a[6:9])), 3)],
                    "max_joint_step_rad": round(float(np.max(np.abs(q - q_meas))), 3)})
        q = np.clip(q, q_meas - 0.35, q_meas + 0.35)  # same per-step bound as the real-arm safety check
        return np.concatenate([q, a[9:10]]).astype(np.float32)

    def cube_in_box():
        """Strict success. The task's InsideAABB uses the whole box AABB, lid included, so pushing
        the box over a cube (or the lid over it) counted as a success in the first eval."""
        if real_mode:
            return real.inside_box(real.to_base(target.get_position_orientation()[0].cpu().numpy()))
        lo, hi = (x.cpu().numpy() for x in box.links[args.box_link].aabb)
        p = target.get_position_orientation()[0].cpu().numpy()
        return bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] < hi[2])

    randomizer = None
    if args.randomize:
        from simfoundry.robots.xarm6_randomize import DomainRandomizer

        jit = dict(front_jitter=(0.01, 1.0), wrist_jitter=(0.003, 1.0), focal_jitter=0.02,
                   wood_species=LIGHT_WOODS) if real_mode else {}
        randomizer = DomainRandomizer(env, np.random.default_rng(args.seed), p_wood=1.0 if args.wood_only else 0.8, **jit)

    policy = PI05XArm6Client(checkpoint_path=None, host=args.host, port=args.port, external_server=True,
                             camera_keys=args.camera_keys, droid_layout=args.droid_layout,
                             **({"action_dim": 10} if eef else {}))

    results = []
    for ep in range(args.episodes):
        prompt, layout = args.prompt, None
        if real_mode:
            names = sorted(cubes)  # blue_cube, yellow_cube
            target = cubes[names[ep % len(names)]]
            color = target.category.replace("_cube", "")
            layout = sampler.sample(color)
            reset_xarm6_env(env, home=np.asarray(layout["start_q"]))
            real.place_box(layout["box_center"], layout["box_yaw"], layout["lid_deg"])
            real.place_cube(target, layout["target_xy"], float(np.random.default_rng(args.seed + ep).uniform(-3.14, 3.14)))
            for other in cubes.values():
                if other is not target:
                    real.place_cube(other, layout["other_xy"], 0.3)
            prompt = args.prompt.format(color=color)
        else:
            reset_xarm6_env(env)
        rand_info = randomizer.apply() if randomizer else None
        if real_mode:
            rand_info = dict(rand_info or {}, surroundings=real.add_surroundings(np.random.default_rng(args.seed + ep)))
            if randomizer is not None:
                real.recolor_cubes(np.random.default_rng(args.seed + 100 + ep))
            hold = th.cat([th.as_tensor(layout["start_q"], dtype=th.float32), th.tensor([0.0])])
            for _ in range(int(round(20 * args.speed))):  # 1 s of settling at any env rate
                env.step(hold)
        lid_pos = float(lid.get_state()[0])
        assert lid_pos > args.min_lid_open, f"lid at {lid_pos:.3f} rad after reset; the box must start open"
        policy.reset()
        ik_seed.update(fail=0, log=[])
        frames, success, t0 = [], False, time.time()
        dist = {"lid_max_change_deg": 0.0, "box_max_shift_m": 0.0, "box_max_dyaw_deg": 0.0}
        milestones, step = {}, 0
        raw_grip, closes = [], []
        grip = GripperHysteresis(args.grip_close_at, args.grip_open_at, args.grip_steps)
        task_success = False
        imgs = xarm6_images(env)
        terminated = truncated = False
        while step < n_steps and not success:
            out = policy.infer({**imgs, "state": state()}, prompt)
            for a in out["action"][: args.execute_horizon]:
                action = th.from_numpy(joint_action(a).astype(np.float32))
                if args.track == "vel":  # the real arm's P velocity loop, one tick per env step
                    q_now = th.as_tensor(xarm6_state(robot)[:6], dtype=th.float32)
                    v = th.clamp(args.kp * (action[:6] - q_now), -args.vmax, args.vmax)
                    action[:6] = q_now + v / env_hz
                raw_grip.append(float(action[6]))
                if args.grip_mode == "hysteresis":
                    action[6] = grip(float(action[6]))
                elif args.grip_mode == "raw":
                    action[6] = float(np.clip(float(action[6]), 0.0, 1.0))
                else:  # training targets are binary
                    action[6] = 1.0 if action[6] >= args.grip_threshold else 0.0
                closes.append(float(action[6]))
                _, _, terminated, truncated, info = env.step(action)
                if real_mode:
                    # The task YAML only knows blue_cube and its loose InsideAABB (box AABB incl. the
                    # raised lid) fires while the cube is still held above the box, which ended blue
                    # episodes before the release. Only the strict check and the timeout end an episode.
                    terminated = truncated = False
                if args.grasp == "latch":
                    latch.update(float(action[6]))
                step += 1
                milestones = info.get("milestones", milestones)
                success = cube_in_box()
                if real_mode:  # did the policy bump the lid or push the box?
                    lid_deg, shift, dyaw = real.disturbance()
                    dist["lid_max_change_deg"] = max(dist["lid_max_change_deg"], round(abs(lid_deg - real.lid_deg), 2))
                    dist["box_max_shift_m"] = max(dist["box_max_shift_m"], round(shift, 4))
                    dist["box_max_dyaw_deg"] = max(dist["box_max_dyaw_deg"], round(abs(dyaw), 2))
                task_success = task_success or bool(env.task.success)
                imgs = xarm6_images(env)
                if int(step * args.action_freq / env_hz) != int((step - 1) * args.action_freq / env_hz):  # video stays real time at 20 fps
                    frames.append(np.concatenate([imgs["front"], imgs["wrist"]], axis=1))
                if success or terminated or truncated or step >= n_steps:
                    break
            if terminated or truncated:
                break
        final_lid = float(lid.get_state()[0])
        # Name = episode, target colour (when the layout has one) and outcome, so a folder listing
        # already shows what failed: episode_03_yellow_fail.mp4
        tag = "".join(f"_{x}" for x in ([color] if real_mode else [])) + ("_success" if success else "_fail")
        video = f"episode_{ep:02d}{tag}.mp4"
        imageio.mimsave(f"{args.out_dir}/{video}", frames, fps=args.action_freq)
        result = {"episode": ep, "video": video, "speed": args.speed, "env_hz": env_hz, "track": args.track,
                  "sim_time_s": round(step / env_hz, 2), "ik_failures": ik_seed["fail"] if eef else None, "ik_fail_examples": ik_seed["log"] if eef else None, "prompt": prompt, "layout": layout, "success": success,
                  "disturbance": dist if real_mode else None, "task_success_loose": task_success, "domain_rand": rand_info, "steps": step, "lid_start": lid_pos,
                  "lid_end": final_lid, "milestones": milestones,
                  "first_close_step": next((i for i, g in enumerate(closes) if g > 0.5), None),
                  "raw_grip_first60": [round(g, 2) for g in raw_grip[:60:6]], "wall_s": round(time.time() - t0, 1)}
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
