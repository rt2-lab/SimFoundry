# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose which frames of a 20 Hz xArm6 episode to keep so the demonstration plays faster.

A policy imitates the pace of its demonstrations, and the autoalloc teleop demos are slow: median
22.7 s, 28 % of it standing still, and the last 7 s (31 %) after the cube is released. Every kept
frame is still labelled as one 20 Hz step, so dropping frames compresses time:

    gripper moving   (command or reading changing, +-3 frames)  keep all   - physical speed limit
    precision        (the --precision-s before a close / open)    keep all, pauses capped at --pause-keep
    idle             (TCP < 5 mm/s and joints < 0.05 rad/s)       keep 1 frame per pause
    free space       (everything else)                            keep every --free-stride-th frame
    tail             (more than --tail-s after the release ends)  drop

Wherever frames are skipped, a frame is still kept as soon as the joints moved more than
--max-joint-step rad or the TCP more than --max-tcp-step m since the last kept one, so slow wrist
rotations inside a "pause" or drift are not turned into jumps. The kept indices are increasing, so
actions stay the recorded targets of later kept frames: the trajectory is resampled in time, not
altered. Used by convert_real_h5_to_lerobot.py --speedup.

    /home/tingting/miniforge3/envs/simfoundry/bin/python scripts/xarm6/demo_speedup.py <ep dirs...>   # prints the effect
"""

import numpy as np

KEEP, PRECISION, IDLE, FREE, TAIL = "gripper", "precision", "idle", "free", "tail"


def speedup_indices(tcp_xyz, q, grip_cmd, grip_act, fps=20.0, free_stride=2, precision_s=1.5, pause_keep=4,
                    tail_s=1.0, idle_speed=0.005, idle_joint_speed=0.05, max_joint_step=0.02, max_tcp_step=0.003):
    """Return (kept frame indices, per-frame label) for one episode.

    tcp_xyz: (T, 3) desired TCP / flange positions [m]; q: (T, 6) desired joints [rad];
    grip_cmd, grip_act: (T,) gripper command and reading in raw units (0 closed .. 850 open).
    """
    n = len(tcp_xyz)
    speed = np.r_[0.0, np.linalg.norm(np.diff(tcp_xyz, axis=0), axis=1) * fps]
    jspeed = np.r_[0.0, np.abs(np.diff(q, axis=0)).max(axis=1) * fps]
    moving_grip = np.r_[False, (np.abs(np.diff(grip_cmd)) > 0.5) | (np.abs(np.diff(grip_act)) > 2.0)]
    moving_grip = np.convolve(moving_grip.astype(float), np.ones(7), mode="same") > 0  # +-3 frames

    # Gripper events: starts of closing / opening command ramps.
    dcmd = np.diff(grip_cmd)
    starts = [i for i in range(1, len(dcmd)) if dcmd[i] != 0 and dcmd[i - 1] == 0] + ([0] if dcmd[0] != 0 else [])
    precision = np.zeros(n, bool)
    for s in starts:
        precision[max(0, s - int(precision_s * fps)):s + 1] = True

    # Tail: after the last opening ramp ends.
    opening = np.where(dcmd > 0.5)[0]
    end = n
    if len(opening):
        end = min(n, opening[-1] + 2 + int(tail_s * fps))

    label = np.full(n, FREE, dtype=object)
    label[(speed < idle_speed) & (jspeed < idle_joint_speed)] = IDLE
    label[precision] = PRECISION
    label[moving_grip] = KEEP
    label[end:] = TAIL

    keep, run_idle, free_count = [0], 0, 0
    for i in range(1, end):
        lab, still = label[i], speed[i] < idle_speed and jspeed[i] < idle_joint_speed
        run_idle = run_idle + 1 if still else 0
        if lab == KEEP:
            take = True
        elif lab == PRECISION:
            take = run_idle <= pause_keep
        elif lab == IDLE:
            take = run_idle == 1
        else:  # free space
            take = free_count % free_stride == 0
            free_count += 1
        j = keep[-1]
        moved = np.abs(q[i] - q[j]).max() > max_joint_step or np.linalg.norm(tcp_xyz[i] - tcp_xyz[j]) > max_tcp_step
        if take or (moved and i - j > 1 and lab != FREE) or (lab == FREE and i - j >= free_stride):
            keep.append(i)
    return np.asarray(keep), label


def _summary(paths):
    import h5py

    rows = []
    for p in paths:
        with h5py.File(p, "r") as f:
            idx, lab = speedup_indices(f["robot_desired_pose"][:, :3], f["robot_desired_joint_angles"][:],
                                       f["gripper_commanded_position"][:], f["gripper_actual_position"][:])
            src = f.attrs.get("source", "real")
        counts = {k: int(np.sum(lab == k)) for k in (KEEP, PRECISION, IDLE, FREE, TAIL)}
        rows.append((src, len(lab), len(idx), counts))
    for src in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == src]
        a, b = np.array([r[1] for r in rr]) / 20, np.array([r[2] for r in rr]) / 20
        share = {k: np.mean([r[3][k] / r[1] for r in rr]) for k in rr[0][3]}
        print(f"{src}: {len(rr)} eps, {np.median(a):.1f} s -> {np.median(b):.1f} s median ({np.median(a / b):.2f}x), "
              f"frames kept {b.sum() / a.sum():.0%}; original frame labels: "
              + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))


if __name__ == "__main__":
    import glob
    import sys

    files = [p for d in sys.argv[1:] for p in sorted(glob.glob(d.rstrip("/") + "/ep*/episode.h5"))]
    _summary(files)
