# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a pi0.5 xArm6 checkpoint on the real arm, without ROS.

Same execution path the autoalloc demos were recorded with (xarm_stack's joint_vel_controller):
absolute joint targets are tracked by a P loop in joint-velocity mode (mode 4,
v = Kp (q_des - q), clipped, 80 Hz, vc_set_joint_velocity with a 0.2 s watchdog); the gripper gets
set_gripper_position(850 * (1 - g)). Observations are the two D435i colour streams (mount ->
front, gripper -> wrist, 640x480) and the state the checkpoint was trained on:

  --action-space joint  state [J1..J6, g],                       action [J1..J6, g]
  --action-space eef    state [TCP xyz, R cols 1-2, g, J1..J6],  action [TCP xyz, R cols 1-2, g]
                        (make_eef_dataset.py; TCP = eef_link; targets go through damped least
                        squares IK seeded from the previous target, like xarm_stack's simple_ik)

Chunks are executed asynchronously: the next chunk is requested --replan-every steps into the
current one and computed while the arm keeps moving; when it arrives, the steps that elapsed since
its observation are skipped and it is cross-faded in over --blend steps. (--sync reproduces the
first deployment: run --execute-horizon steps, then stop for inference, ~0.25 s every 0.75 s.)

Start the policy server first (simfoundry/policies/pi05_serve/server.py --port 8420), put the arm
at the start pose (go_to_start_pose.py), then:

    .cache/radio-prep-venv/bin/python scripts/xarm6/real/deploy_pi05_xarm.py --prompt "Pick up the yellow cube and place it into the box." --dry-run
    .cache/radio-prep-venv/bin/python scripts/xarm6/real/deploy_pi05_xarm.py --prompt "..." --action-space eef --max-seconds 60

--speed S plays the actions S times faster than the data (see docs/XARM6_REAL_DEPLOY_GUIDE.md §6b).

--dry-run never enables motion: it reads cameras and joints, queries the policy, and logs the
predicted targets. Safety: a joint target more than --max-jump rad from the measured joints, a
flange below --min-flange-z stops the run (IK targets past a limit get the nearest reachable pose); Ctrl-C stops; on any exit the arm gets
zero velocity and position mode. Keep a hand on the emergency stop. Every run writes
<out-dir>/<time>/cameras.mp4 (H.264, front | wrist) and log.jsonl.
"""

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MOUNT_SN, WRIST_SN = "243222070233", "243122074665"
GRIPPER_OPEN = 850.0


def open_video(path, fps):
    """H.264 (plays in VS Code / browsers, unlike OpenCV's mp4v) in a fragmented MP4: the index is
    written as the video goes, so a run that is killed (e-stop, SIGKILL, crash) still leaves a
    playable file up to the last second."""
    import imageio

    return imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                              ffmpeg_params=["-crf", "23", "-preset", "veryfast",
                                             "-movflags", "frag_keyframe+empty_moov+default_base_moof"])


class VideoThread(threading.Thread):
    """Encodes frames off the control loop (a pipe write to ffmpeg can block for tens of ms)."""

    def __init__(self, path, fps):
        super().__init__(daemon=True)
        self.writer, self.q = open_video(path, fps), queue.Queue(maxsize=40)
        self.start()

    def put(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:  # never stall control for the video
            pass

    def run(self):
        while (frame := self.q.get()) is not None:
            self.writer.append_data(frame)
        self.writer.close()

    def close(self):
        self.q.put(None)
        self.join(timeout=10.0)


class Cameras(threading.Thread):
    """Keeps the latest frame of both D435i colour streams (30 fps)."""

    def __init__(self):
        super().__init__(daemon=True)
        import pyrealsense2 as rs

        self.pipes = {}
        for name, sn in (("front", MOUNT_SN), ("wrist", WRIST_SN)):
            cfg = rs.config()
            cfg.enable_device(sn)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipe = rs.pipeline()
            pipe.start(cfg)
            self.pipes[name] = pipe
        for _ in range(30):  # let auto exposure settle
            self._grab()
        self.lock, self.running = threading.Lock(), True
        self.latest = self._grab()
        self.start()

    def _grab(self):
        out = {}
        for name, pipe in self.pipes.items():
            frames = pipe.wait_for_frames(timeout_ms=2000)
            out[name] = np.asanyarray(frames.get_color_frame().get_data()).copy()
        return out

    def run(self):
        while self.running:
            frame = self._grab()
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            return dict(self.latest)

    def stop(self):
        self.running = False
        self.join(timeout=2.0)
        for p in self.pipes.values():
            p.stop()


class JointVelTracker(threading.Thread):
    """xarm_stack joint_vel_controller: v = Kp (q_des - q), deadband, clip; 80 Hz."""

    def __init__(self, arm, kp, vmax, rate=80.0):
        super().__init__(daemon=True)
        self.arm, self.kp, self.vmax, self.dt = arm, kp, vmax, 1.0 / rate
        self.q_des, self.lock, self.running, self.error = None, threading.Lock(), True, None

    def set_target(self, q):
        with self.lock:
            self.q_des = np.asarray(q, dtype=float)

    def run(self):
        while self.running:
            t0 = time.time()
            with self.lock:
                q_des = None if self.q_des is None else self.q_des.copy()
            if q_des is not None:
                code, ang = self.arm.get_servo_angle(is_radian=True)
                if code != 0:
                    self.error = f"get_servo_angle code {code}"
                    break
                v = self.kp * (q_des - np.asarray(ang[:6]))
                v[np.abs(v) < 0.001] = 0.0
                v = np.clip(v, -self.vmax, self.vmax)
                code = self.arm.vc_set_joint_velocity(v.tolist(), is_radian=True, is_sync=True, duration=0.2)
                if code != 0:
                    self.error = f"vc_set_joint_velocity code {code}"
                    break
            time.sleep(max(0.0, self.dt - (time.time() - t0)))
        self.arm.vc_set_joint_velocity([0.0] * 6, is_radian=True, is_sync=True, duration=0.2)


def rot_from_cols(r6):
    """[r11 r21 r31 r12 r22 r32] -> rotation matrix (Gram-Schmidt)."""
    b1 = r6[:3] / np.linalg.norm(r6[:3])
    b2 = r6[3:6] - b1 @ r6[3:6] * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


class ChunkPlanner(threading.Thread):
    """Requests chunks from the policy server off the control loop. Each result is stamped with
    the time its observation was taken, so the executor can skip the steps that already elapsed."""

    def __init__(self, policy, cams, get_state, prompt, action_dim):
        super().__init__(daemon=True)
        self.policy, self.cams, self.get_state, self.prompt, self.action_dim = policy, cams, get_state, prompt, action_dim
        self.request, self.lock, self.running = threading.Event(), threading.Lock(), True
        self.result, self.error, self.latency = None, None, []

    def run(self):
        while self.running:
            if not self.request.wait(timeout=0.1):
                continue
            self.request.clear()
            try:
                t_obs = time.time()
                imgs, state = self.cams.read(), self.get_state()
                chunk = np.asarray(self.policy.infer({**imgs, "state": state}, self.prompt)["action"])[:, :self.action_dim]
                self.latency.append(time.time() - t_obs)
                with self.lock:
                    self.result = (t_obs, chunk)
            except Exception as e:  # surfaced by the control loop, which stops the arm
                self.error = repr(e)

    def take(self):
        with self.lock:
            res, self.result = self.result, None
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--action-space", choices=("joint", "eef"), default="joint")
    ap.add_argument("--robot-ip", default="192.168.1.111")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=40.0)
    ap.add_argument("--hz", type=float, default=20.0, help="action rate of the data (dataset fps)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="play the policy's actions this many times faster than the data (actions/s = hz x speed). "
                         "Targets are interpolated between actions, so any --loop-hz works")
    ap.add_argument("--loop-hz", type=float, default=None, help="control loop rate; default min(200, hz x speed)")
    ap.add_argument("--tracker-hz", type=float, default=80.0, help="velocity loop rate (xarm_stack: 80); up to ~200")
    ap.add_argument("--sync", action="store_true", help="old behaviour: stop for inference every --execute-horizon steps")
    ap.add_argument("--execute-horizon", type=int, default=15, help="--sync only")
    ap.add_argument("--replan-every", type=int, default=None,
                    help="actions into a chunk at which the next is requested; default round(10 / speed), min 1")
    ap.add_argument("--blend", type=int, default=5, help="steps to cross-fade from the old chunk to the new one")
    ap.add_argument("--kp", type=float, default=10.0, help="xarm_stack config.yaml Kp")
    ap.add_argument("--vmax", type=float, default=None,
                    help="rad/s clip; default min(pi, 0.6 x speed) (xarm_stack: 1.0; demos p99 0.45; hardware pi)")
    ap.add_argument("--max-jump", type=float, default=None,
                    help="rad; stop if a target is this far from the arm; default min(0.8, 0.35 + 0.05 (speed - 1)): "
                         "tracking error grows with speed")
    ap.add_argument("--min-flange-z", type=float, default=0.15, help="m; stop below (grasps put the flange ~0.17)")
    ap.add_argument("--out-dir", default="Data/cube2_scene/real_deploy")
    args = ap.parse_args()

    from make_eef_dataset import tcp_pose9
    from xarm.wrapper import XArmAPI

    from simfoundry.policies.pi05_lerobot import PI05XArm6Client
    from simfoundry.robots.xarm6_kinematics import XArm6Kinematics

    s_ = max(args.speed, 1e-3)
    args.loop_hz = args.loop_hz or min(200.0, args.hz * s_)
    args.replan_every = args.replan_every if args.replan_every is not None else max(1, int(round(10 / s_)))
    args.vmax = args.vmax if args.vmax is not None else min(np.pi, 0.6 * s_)
    args.max_jump = args.max_jump if args.max_jump is not None else min(0.8, 0.35 + 0.05 * (s_ - 1))
    act_rate = args.hz * s_  # actions per second
    print(f"[deploy] speed x{s_:g}: {act_rate:g} actions/s (a 50-action chunk lasts {50 / act_rate:.2f} s), loop {args.loop_hz:g} Hz, "
          f"velocity loop {args.tracker_hz:g} Hz, vmax {args.vmax:.2f} rad/s, max-jump {args.max_jump:.2f} rad, "
          f"replan every {args.replan_every} actions", flush=True)
    eef = args.action_space == "eef"
    action_dim = 10 if eef else 7
    kin_flange, kin_tcp = XArm6Kinematics(tip_link="link6"), XArm6Kinematics(tip_link="eef_link")
    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "log.jsonl"), "w")
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    policy = PI05XArm6Client(checkpoint_path=None, port=args.port, external_server=True, camera_keys="droid",
                             action_dim=action_dim)
    cams = Cameras()
    arm = XArmAPI(args.robot_ip, is_radian=True)
    arm.clean_warn()
    arm.clean_error()
    tracker, planner, writer = None, None, None
    stop = {"flag": False, "why": ""}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True, why="ctrl-c"))

    grip_cache = {"g": GRIPPER_OPEN, "t": -1.0}

    def measured(grip_max_age=0.0):
        """Measured joints and gripper. The gripper read costs ~4 ms (joints ~1 ms), so the fast
        control loop reuses a reading up to @grip_max_age s old; the policy state always reads it."""
        _, ang = arm.get_servo_angle(is_radian=True)
        if time.time() - grip_cache["t"] >= grip_max_age:
            _, g = arm.get_gripper_position()
            grip_cache.update(g=GRIPPER_OPEN if g is None else float(g), t=time.time())
        return np.asarray(ang[:6], dtype=float), grip_cache["g"]

    def policy_state():
        q, g = measured()
        grip = np.float32(1.0 - g / GRIPPER_OPEN)
        if eef:
            return np.concatenate([tcp_pose9(kin_tcp, q[None])[0], [grip], q]).astype(np.float32)
        return np.r_[q, grip].astype(np.float32)

    q_seed = {"q": None}

    def to_joints(a, q_meas):
        """Policy action -> (joint target, ok, TCP xyz). Joint actions pass through."""
        if not eef:
            return np.asarray(a[:6], dtype=float), True, None
        R = rot_from_cols(np.asarray(a[3:9], dtype=float))
        # Seeded from the measured joints like simple_ik; a target past a limit gets the nearest
        # reachable pose (see XArm6Kinematics.ik_nearest), bounded by the --max-jump check below.
        q, ok, _ = kin_tcp.ik_nearest(np.asarray(a[:3], dtype=float), Rotation.from_matrix(R).as_quat(), q_meas)
        return q, True, np.round(a[:3], 4).tolist()

    def interp(chunk, kf):
        """Action at fractional index kf: linear between neighbours (rotation columns are
        re-orthonormalised in to_joints); the gripper takes the nearer one."""
        i = min(int(kf), len(chunk) - 2)
        f = float(np.clip(kf - i, 0.0, 1.0))
        out = (1 - f) * chunk[i] + f * chunk[i + 1]
        out[-1] = chunk[i + 1][-1] if f >= 0.5 else chunk[i][-1]
        return out

    def blend(old, new, w):
        out = (1 - w) * old + w * new
        out[-1] = new[-1] if w >= 0.5 else old[-1]  # gripper: switch, do not average
        return out

    try:
        if not args.dry_run:
            arm.motion_enable(True)
            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(int(min(5000, 1500 * s_)))  # xarm_stack uses 1500; the gripper allows 5000
            arm.set_mode(4)
            arm.set_state(0)
            time.sleep(1.0)  # xarm_stack: the arm may jerk if commanded before it is set up
            tracker = JointVelTracker(arm, args.kp, args.vmax, rate=args.tracker_hz)
            tracker.start()
        policy.reset()
        q_seed["q"] = measured()[0]
        planner = ChunkPlanner(policy, cams, policy_state, args.prompt, action_dim)
        planner.start()
        writer = VideoThread(os.path.join(run_dir, "cameras.mp4"), 20)
        next_frame = 0.0  # video and log flush at 20 Hz of wall time, whatever the loop rate

        planner.request.set()
        while planner.result is None and planner.error is None:  # first chunk: the arm waits
            time.sleep(0.01)
        if planner.error:
            raise RuntimeError(f"policy: {planner.error}")
        # The arm stood still while this chunk was computed, so it starts at step 0 (later chunks
        # skip the steps that elapsed during their inference).
        t_start = chunk_t0 = time.time()
        chunk = planner.take()[1]
        prev = None  # (t0, chunk) being faded out
        fade_left, n_steps, last_grip, requested = 0, 0, None, False
        starved_since, n_starved = None, 0
        while not stop["flag"] and time.time() - t_start < args.max_seconds:
            t0 = time.time()
            if planner.error:
                stop.update(flag=True, why=f"policy: {planner.error}")
                break
            kf = (t0 - chunk_t0) * act_rate  # (fractional) action of the current chunk due now
            k = int(kf)
            if args.sync and k >= args.execute_horizon:  # (actions)
                planner.request.set()  # old behaviour: hold the last target while inferring
                while planner.result is None and planner.error is None:
                    time.sleep(0.005)
                if planner.error:
                    continue
                chunk, chunk_t0 = planner.take()[1], time.time()  # the arm held still: start at step 0
                prev, fade_left = None, 0
                continue
            if not args.sync:
                if not requested and k >= args.replan_every:
                    planner.request.set()
                    requested = True
                res = planner.take()
                if res is not None:
                    prev, (chunk_t0, chunk) = (chunk_t0, chunk), res
                    fade_left, requested = args.blend, False
                    kf = (t0 - chunk_t0) * act_rate
                    k = int(kf)
            if k >= len(chunk) - 1:
                # Inference is behind (at high --speed a chunk lasts less than two inference
                # latencies): hold the chunk's last target until the next one arrives, so the arm
                # slows down instead of the run ending; give up after 1 s without a new chunk.
                starved_since = starved_since or t0
                n_starved += 1
                if t0 - starved_since > 1.0:
                    stop.update(flag=True, why="no new chunk for 1 s (inference too slow)")
                    break
                kf = len(chunk) - 1
            else:
                starved_since = None
            a = interp(chunk, kf)
            if prev is not None and fade_left > 0:
                a = blend(interp(prev[1], min((t0 - prev[0]) * act_rate, len(prev[1]) - 1)), a, 1.0 - fade_left / (args.blend + 1))
                fade_left -= 1
            q, g_raw = measured(grip_max_age=0.05)
            q_tgt, ok, tcp = to_joints(a, q)
            jump = float(np.max(np.abs(q_tgt - q)))
            fz = float(kin_flange.fk(q_tgt)[2, 3])
            rec = {"t": round(t0 - t_start, 3), "k": k, "q": np.round(q, 4).tolist(), "grip_raw": g_raw,
                   "q_des": np.round(q_tgt, 4).tolist(), "g_des": round(float(a[-1]), 3), "tcp_des": tcp,
                   "ik_ok": ok, "jump": round(jump, 3), "flange_z_des": round(fz, 3), "fading": prev is not None and fade_left > 0,
                   "starved": starved_since is not None}
            log.write(json.dumps(rec) + "\n")
            if not ok or jump > args.max_jump or fz < args.min_flange_z:
                stop.update(flag=True, why=f"safety: ik_ok {ok}, jump {jump:.3f} rad, flange z {fz:.3f} m")
                break
            if args.dry_run:
                if n_steps % max(1, int(args.loop_hz / 4)) == 0:
                    print(f"[dry] t={rec['t']:.2f} k={k} jump={jump:.3f} flange_z={fz:.3f} g={a[-1]:.2f} tcp={tcp}", flush=True)
            else:
                tracker.set_target(q_tgt)
                g_cmd = float(np.clip(GRIPPER_OPEN * (1.0 - a[-1]), 0, GRIPPER_OPEN))
                if last_grip is None or abs(g_cmd - last_grip) >= 5:
                    arm.set_gripper_position(g_cmd, wait=False)
                    last_grip = g_cmd
                if tracker.error:
                    stop.update(flag=True, why=tracker.error)
                    break
            if t0 - t_start >= next_frame:
                frame = cams.read()
                writer.put(np.hstack([frame["front"], frame["wrist"]]))
                log.flush()
                next_frame += 0.05
            n_steps += 1
            if args.dry_run and t0 - t_start >= 3.0:
                stop.update(flag=True, why="dry run done")
            time.sleep(max(0.0, 1.0 / args.loop_hz - (time.time() - t0)))
        why = stop["why"] or "time limit"
        lat = np.asarray(planner.latency)
        print(f"[deploy] stopped: {why}; {n_steps} loop ticks ({n_steps / max(time.time() - t_start, 1e-3):.0f} Hz), {n_starved / max(n_steps, 1):.0%} of them waiting for a chunk; inference latency median {np.median(lat):.3f}s "
              f"max {lat.max():.3f}s over {len(lat)} chunks; log + video in {run_dir}", flush=True)
    finally:
        if tracker is not None:
            tracker.running = False
            tracker.join(timeout=2.0)
            arm.vc_set_joint_velocity([0.0] * 6, is_radian=True, is_sync=True, duration=0.2)
            arm.set_mode(0)
            arm.set_state(0)
        if planner is not None:
            planner.running = False
        if writer is not None:
            writer.close()
        cams.stop()
        arm.disconnect()
        log.close()


if __name__ == "__main__":
    main()
