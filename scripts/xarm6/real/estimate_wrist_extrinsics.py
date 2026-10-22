# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimate the wrist camera's pose in the xArm6 flange (link6) frame from recorded episodes.

The target cube does not move between the start of an episode and the grasp, and its position at
the grasp is known: flange pose (forward kinematics) times the cube offset in the flange found by
estimate_mount_extrinsics.py. While the arm approaches, the wrist camera sees that fixed point from
many flange poses, which pins down the fixed camera-in-flange transform (least squares on the
reprojection error, initialised from UFactory's D435i wrist mount).

    mamba run -n simfoundry python scripts/xarm6/real/estimate_wrist_extrinsics.py \
        --data /home/tingting/Documents/data/autoalloc_real \
        --mount scripts/cfg/real/xarm6_mount_extrinsics.json --out scripts/cfg/real/xarm6_wrist_extrinsics.json
"""

import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from simfoundry.robots.xarm6_kinematics import XArm6Kinematics  # noqa: E402

# Wrist camera (D435i SN 243122074665), colour 640x480, factory intrinsics (distortion is zero).
K_WRIST = np.array([[600.8926, 0.0, 331.0353], [0.0, 599.8849, 244.8186], [0.0, 0.0, 1.0]])
HSV = {"blue": ((95, 120, 40), (130, 255, 255)), "yellow": ((18, 150, 110), (35, 255, 255))}
# UFactory D435i wrist mount (the value used in sim): colour optical centre in link6, USD camera
# axes (looking down -z, y up). OpenCV axes are USD axes with y and z flipped.
INIT_POS = np.array([0.06746, -0.0325, 0.0237])
INIT_ROT = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0]) * Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]))


# Near-field constraint: with the gripper fully open (gripper_actual ~841-850), the inner corners of
# the two fingertips sit at fixed pixels in every wrist frame (median over 222 open-gripper frames of
# ep14-50: (52, 322) and (616, 320)). In the flange frame (= gripper base frame) those corners are at
# y = +-0.0445 m (inner pad faces, open), z = 0.162 m (fingertips; both checked against the sim
# gripper), and x = +0.016 m: the camera is ~10 cm from the tips, so the 32 mm pad width spans
# ~170 px vertically and the silhouette's top corner is the pad's far edge, not its middle (using
# x = 0 put the sim fingers ~90 px above the real ones). Far-away cube observations alone leave
# rotation and translation trading off against each other; these two close points do not.
FINGERTIP_PX = np.array([[52.0, 322.0], [616.0, 320.0]])
FINGERTIP_FLANGE = np.array([[0.016, 0.0445, 0.162], [0.016, -0.0445, 0.162]])
# 200: the fingertips are matched exactly (the cube median error goes 21 -> 25 px, within the ~1 cm
# uncertainty of the cube rest position); 15 lets far cube points pull the fingers 55-80 px off.
FINGERTIP_WEIGHT = float(os.environ.get("FINGERTIP_WEIGHT", 200.0))


def detect(img_bgr, color, min_area=150):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo, hi = HSV[color]
    m = cv2.morphologyEx(cv2.inRange(hsv, np.array(lo), np.array(hi)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    # Reject blobs cut by the image border (centroid would be biased) or too small to trust.
    if area < min_area or x <= 1 or y <= 1 or x + w >= 639 or y + h >= 479:
        return None
    return cents[i]


def collect(paths, k6, off, stride):
    obs = []  # (episode, base->flange 4x4, cube xyz in base, uv)
    for p in paths:
        ep = int(os.path.basename(os.path.dirname(p))[2:])
        f = h5py.File(p, "r")
        J = f["robot_current_joint_angles"][:].astype(float)
        ga = f["gripper_actual_position"][:]
        closed = ga < 300
        if not closed.any():
            continue
        g0 = int(np.argmax(closed))
        Tg = k6.fk(J[g0])
        X = Tg[:3, :3] @ off + Tg[:3, 3]  # cube rest position (it does not move before the grasp)
        imgs = f["gripper_image_rgb_compressed"]
        for t in range(0, max(g0 - 5, 0), stride):
            T = k6.fk(J[t])
            if T[2, 3] - X[2] < 0.12:  # too close: cube fills / leaves the frame
                continue
            img = cv2.imdecode(np.frombuffer(bytes(imgs[t]), np.uint8), cv2.IMREAD_COLOR)
            uv = detect(img, "blue" if ep <= 27 else "yellow")
            if uv is not None:
                obs.append((ep, T, X, uv))
    return obs


def project(params, obs):
    T_fc = np.eye(4)
    T_fc[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T_fc[:3, 3] = params[3:6]
    uv = []
    for _, T_bf, X, _ in obs:
        T_bc = T_bf @ T_fc
        Xc = T_bc[:3, :3].T @ (X - T_bc[:3, 3])
        uv.append(K_WRIST[:2, :2] @ (Xc[:2] / Xc[2]) + K_WRIST[:2, 2])
    return np.array(uv)


def project_fingertips(params):
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    Xc = (R.T @ (FINGERTIP_FLANGE - params[3:6]).T).T
    return (K_WRIST[:2, :2] @ (Xc[:, :2] / Xc[:, 2:3]).T).T + K_WRIST[:2, 2]


def match_fingertips(params):
    """Order the 3D fingertips to match the pixel ones (left/right in the image)."""
    uv = project_fingertips(params)
    if uv[0, 0] > uv[1, 0]:
        FINGERTIP_FLANGE[:] = FINGERTIP_FLANGE[::-1].copy()


def fit(obs, p0):
    target = np.array([o[3] for o in obs])
    res = least_squares(lambda p: np.r_[(project(p, obs) - target).ravel(),
                                        FINGERTIP_WEIGHT * (project_fingertips(p) - FINGERTIP_PX).ravel()],
                        p0, loss="huber", f_scale=5.0)
    return res.x, np.linalg.norm(project(res.x, obs) - target, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mount", required=True, help="estimate_mount_extrinsics.py output (for the cube offset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gate", type=float, default=40.0)
    args = ap.parse_args()
    off = np.array(json.load(open(args.mount))["all"]["cube_offset_in_flange"])
    k6 = XArm6Kinematics(tip_link="link6")
    paths = [p for p in sorted(glob.glob(os.path.join(args.data, "ep*/episode.h5"))) if not p.endswith("ep000008/episode.h5")]
    obs = collect(paths, k6, off, args.stride)
    p = np.r_[INIT_ROT.as_rotvec(), INIT_POS]
    match_fingertips(p)
    err0 = np.linalg.norm(project(p, obs) - np.array([o[3] for o in obs]), axis=1)
    print(f"{len(obs)} observations; with the UFactory mount guess: median {np.median(err0):.1f}px")
    p, err = fit(obs, p)
    for _ in range(3):
        keep = err < args.gate
        obs = [o for o, k in zip(obs, keep) if k]
        p, err = fit(obs, p)
    R = Rotation.from_rotvec(p[:3])
    eps = np.array([o[0] for o in obs])
    per_ep = {int(e): round(float(np.median(err[eps == e])), 1) for e in np.unique(eps)}
    print(f"fit: {len(obs)} points, median {np.median(err):.1f}px, 90% {np.percentile(err, 90):.1f}px; "
          f"fingertip error {np.linalg.norm(project_fingertips(p) - FINGERTIP_PX, axis=1).round(1)} px")
    print(f"camera in flange: position {p[3:6].round(4)} m (UFactory mount guess {INIT_POS}); rotation differs from the guess by "
          f"{np.degrees((R * INIT_ROT.inv()).magnitude()):.1f} deg; optical axis in flange {R.as_matrix()[:, 2].round(3)}")
    print("per-episode median px:", per_ep)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.as_matrix(), p[3:6]
    json.dump({"T_flange_cam_opencv": T.round(6).tolist(), "position": p[3:6].round(4).tolist(),
               "quat_xyzw_opencv": R.as_quat().round(6).tolist(), "n_points": len(obs),
               "reproj_median_px": round(float(np.median(err)), 2), "per_episode_median_px": per_ep,
               "K": K_WRIST.tolist(), "image_size": [640, 480], "camera_serial": "243122074665"},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
