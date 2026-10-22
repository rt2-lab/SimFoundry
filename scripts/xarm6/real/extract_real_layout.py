# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode layout of the real xArm6 cube-into-box recordings, in the robot base frame.

For every episode: start joints, target cube position (forward kinematics at the grasp times the
cube offset in the flange from estimate_mount_extrinsics.py), the other cube (colour blob in the
first mount frame, back-projected onto its top face), the box pose (footprint of the mount depth
cloud), and the timing of grasp and release. scripts/xarm6/generate_scripted_demos.py
--real-layout samples sim layouts from this file.

    mamba run -n simfoundry python scripts/xarm6/real/extract_real_layout.py \
        --data /home/tingting/Documents/data/autoalloc_real \
        --mount scripts/cfg/real/xarm6_mount_extrinsics.json --out scripts/cfg/real/xarm6_real_layout.json

The depth stream is not aligned to colour: it is decoded with the depth intrinsics and moved into
the colour optical frame with the factory depth->colour extrinsics (read off the cameras with
pyrealsense2). With that, the table comes out at z = 0 +- 5 mm, which checks the mount extrinsics.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from simfoundry.robots.xarm6_kinematics import XArm6Kinematics  # noqa: E402

# Mount D435i (SN 243222070233) depth stream, 640x480, and depth->colour extrinsics.
K_DEPTH = np.array([[386.635, 0.0, 322.408], [0.0, 386.635, 244.182], [0.0, 0.0, 1.0]])
R_DEPTH_TO_COLOR = np.array([[0.9999, 0.0046, 0.0114], [-0.0046, 1.0, 0.0037], [-0.0114, -0.0038, 0.9999]])
T_DEPTH_TO_COLOR = np.array([0.0146, 0.0, 0.0003])
# Outer footprint of the organiser box (transparent lid hinged on one long side): the depth
# footprint of the fully visible episodes (0.33-0.36 x 0.24-0.25, inflated by the lid hinge) and
# sim renders at the recorded states agree with the reconstructed toolbox at 0.70x.
BOX_SIZE = (0.321, 0.242, 0.083)  # long side, short side, height (m)
CUBE_SIZE = 0.025
HSV = {"blue": ((95, 120, 40), (130, 255, 255)), "yellow": ((18, 110, 110), (35, 255, 255))}


def decode_rgb(buf):
    return cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)


def depth_cloud(buf, T_base_cam):
    """Mount depth frame -> (N, 3) points in the robot base frame."""
    d = cv2.imdecode(np.frombuffer(bytes(buf)[12:], np.uint8), cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
    v, u = np.nonzero(d)
    z = d[v, u]
    X = np.stack([(u - K_DEPTH[0, 2]) / K_DEPTH[0, 0] * z, (v - K_DEPTH[1, 2]) / K_DEPTH[1, 1] * z, z], -1)
    Xc = X @ R_DEPTH_TO_COLOR.T + T_DEPTH_TO_COLOR
    return Xc @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def fit_box(P, res=0.004):
    """Box centre (x, y) and yaw of its long side from a top-down occupancy map of points at wall
    height. The lid stands above the rim and is transparent, so it is mostly cut by the height band."""
    sel = (P[:, 2] > 0.035) & (P[:, 2] < 0.085) & (P[:, 0] > -0.1) & (P[:, 0] < 0.7) & (P[:, 1] < 0.05) & (P[:, 1] > -0.75)
    Q = P[sel]
    if len(Q) < 200:
        return None
    x0, y0 = -0.1, -0.75
    occ = np.zeros((int(0.8 / res), int(0.8 / res)), np.uint8)
    occ[((Q[:, 0] - x0) / res).astype(int), ((Q[:, 1] - y0) / res).astype(int)] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.argwhere(lab == i).astype(np.float32)  # (row = x, col = y)
    (cr, cc), (w, h), ang = cv2.minAreaRect(pts[:, ::-1].copy())  # points as (y, x)
    cx, cy = x0 + cc * res, y0 + cr * res
    # minAreaRect works in (y, x) image coordinates; turn its angle into the yaw of the long side.
    a = np.radians(ang)
    side = np.array([np.cos(a), np.sin(a)])  # direction of side "w" in (y, x)
    if h > w:
        side = np.array([-side[1], side[0]])
    yaw = np.arctan2(side[0], side[1])  # long side direction in base (x, y): (side[1], side[0])
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # the rectangle is symmetric; lid side is fixed below
    return float(cx), float(cy), float(yaw), (max(w, h) * res, min(w, h) * res)


def detect_on_table(img, color, T_base_cam, K):
    """Largest blob of @color whose back-projection lands on the table workspace (blue chairs and
    lights behind the table pass the colour test too). Returns the cube centre xy, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = HSV[color]
    m = cv2.morphologyEx(cv2.inRange(hsv, np.array(lo), np.array(hi)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    best, best_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        xy = backproject(cents[i], T_base_cam, K, CUBE_SIZE)[:2]
        if 40 <= area <= 3000 and area > best_area and 0.05 < xy[0] < 0.6 and -0.25 < xy[1] < 0.35:
            best, best_area = xy, area
    return best


def backproject(uv, T_base_cam, K, z):
    d = T_base_cam[:3, :3] @ np.linalg.solve(K, [uv[0], uv[1], 1.0])
    o = T_base_cam[:3, 3]
    return o + (z - o[2]) / d[2] * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mount", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = json.load(open(args.mount))["all"]
    T_bc, K, off = np.array(m["T_base_cam_opencv"]), np.array(m["K"]), np.array(m["cube_offset_in_flange"])
    k6 = XArm6Kinematics(tip_link="link6")
    episodes, prev_yaw = [], None
    for p in sorted(glob.glob(os.path.join(args.data, "ep*/episode.h5"))):
        ep = int(p.split("/")[-2][2:])
        f = h5py.File(p, "r")
        J = f["robot_current_joint_angles"][:].astype(float)
        ga = f["gripper_actual_position"][:]
        # Holding = the fingers at the episode's narrowest width (the cube, ~200-230 raw). A fixed
        # threshold misses ep9, whose release only reopens to 264.
        closed = (ga < np.percentile(ga, 5) + 25) & (ga < 400)
        if ga.max() - ga.min() < 50:
            print(f"ep{ep}: gripper never closes, skipped")
            continue
        # The carry is the longest closed run; a few episodes have a missed grasp before it.
        edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
        runs = edges.reshape(-1, 2)
        g0, r0 = runs[np.argmax(runs[:, 1] - runs[:, 0])]
        g0, r0 = int(g0), int(min(r0, len(ga) - 1))
        Tg, Tr = k6.fk(J[g0]), k6.fk(J[r0])
        target = Tg[:3, :3] @ off + Tg[:3, 3]
        release = Tr[:3, :3] @ off + Tr[:3, 3]
        color = "blue" if ep <= 27 else "yellow"
        other = "yellow" if color == "blue" else "blue"
        img0 = decode_rgb(f["mount_image_rgb_compressed"][0])
        other_xy = detect_on_table(img0, other, T_bc, K)
        box = fit_box(depth_cloud(f["mount_image_depth_compressed"][0], T_bc))
        rec = {"ep": ep, "n_steps": len(ga), "grasp_step": g0, "release_step": r0, "target_color": color,
               "start_q": J[0].round(4).tolist(), "target_cube": target.round(4).tolist(),
               "release_cube": release.round(4).tolist(),
               "other_cube": None if other_xy is None else other_xy.round(4).tolist(),
               "gripper_open_max": float(ga[:g0].max()) if g0 else float(ga[0])}
        if target[2] > 0.06:  # ep10: the cube was picked off something, not the table
            print(f"ep{ep}: grasp at z={target[2]:.3f}, skipped")
            continue
        if box is not None:
            cx, cy, yaw, dims = box
            if dims[0] - dims[1] < 0.03 and prev_yaw is not None:  # near-square footprint: long side ambiguous
                yaw = min((yaw, yaw + np.pi / 2, yaw - np.pi / 2), key=lambda a: abs(np.sin(a - prev_yaw)))
            prev_yaw = yaw
            # The box is often only partly in view and the footprint then slides away from the robot.
            # The cube was dropped into it, so pull the box along its short axis until the release
            # point is at least 3 cm inside the rim.
            c, s_ = np.cos(yaw), np.sin(yaw)
            ly = -s_ * (release[0] - cx) + c * (release[1] - cy)
            lim = BOX_SIZE[1] / 2 - 0.03
            if abs(ly) > lim:
                shift = ly - np.sign(ly) * lim
                cx, cy = cx - s_ * shift, cy + c * shift
            # Lid side: the box's long side facing away from the robot base (true in every frame
            # checked). Record the yaw so that the box's local -y (lid side) points away.
            n_out = np.array([np.sin(yaw), -np.cos(yaw)])  # local -y in base
            if n_out @ np.array([cx, cy]) < 0:
                yaw = yaw + np.pi if yaw < 0 else yaw - np.pi
            rec["box"] = {"center": [round(cx, 4), round(cy, 4)], "yaw": round(yaw, 4),
                          "measured_dims": [round(d, 3) for d in dims]}
        episodes.append(rec)
        b = rec.get("box", {})
        print(f"ep{ep:02d} {color:6s} target {target.round(3)} other {rec['other_cube']} box {b.get('center')} "
              f"yaw {np.degrees(b.get('yaw', np.nan)):.0f} dims {b.get('measured_dims')}")
    out = {"box_size": BOX_SIZE, "cube_size": CUBE_SIZE, "rate_hz": 20, "episodes": episodes,
           "note": "base frame of the xArm6 (link_base), metres/radians; box yaw = direction of the long side, "
                   "lid hinged on the box's local -y long side"}
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
