# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimate the fixed (mount) camera's pose in the xArm6 base frame from recorded episodes alone.

No calibration board is needed. While the robot carries a cube, the cube sits at a fixed (unknown)
offset in the flange frame, and the flange pose is known from the recorded joint angles through
forward kinematics. The cube is easy to segment by colour in the mount image. That gives many
2D-3D correspondences; RANSAC PnP initialises the camera pose, and a robust least-squares fit then
refines the camera pose and the cube offset jointly.

    mamba run -n simfoundry python scripts/xarm6/real/estimate_mount_extrinsics.py \
        --data /home/tingting/Documents/data/autoalloc_real --out scripts/cfg/real/xarm6_mount_extrinsics.json

Detections are gated: after a first fit, only blobs within --gate px of the predicted cube position
are kept and the fit is repeated (the colour detector otherwise latches onto the other cube or a
blue light when the held cube is occluded by the gripper). Per-episode reprojection errors flag a
moved camera; on the 2026-10 dataset one pose fits all 50 episodes.
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

# Mount camera (D435i SN 243222070233), colour 640x480, factory intrinsics (distortion is zero).
K_MOUNT = np.array([[605.7482, 0.0, 324.0126], [0.0, 605.3567, 239.6684], [0.0, 0.0, 1.0]])
HSV = {"blue": ((95, 120, 40), (130, 255, 255)), "yellow": ((18, 110, 110), (35, 255, 255))}


def target_color(ep_idx):
    return "blue" if ep_idx <= 27 else "yellow"


def detect(img_bgr, color):
    """Centroid of the largest blob of @color, or None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo, hi = HSV[color]
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    return cents[i] if 40 <= area <= 8000 else None


def collect(paths, k6, stride):
    obs = []  # (episode, flange 4x4, uv)
    for p in paths:
        ep = int(os.path.basename(os.path.dirname(p))[2:])
        f = h5py.File(p, "r")
        J = f["robot_current_joint_angles"][:].astype(float)
        ga = f["gripper_actual_position"][:]
        imgs = f["mount_image_rgb_compressed"]
        for t in range(0, len(ga), stride):
            if ga[t] >= 300:  # not holding
                continue
            T = k6.fk(J[t])
            if T[2, 3] < 0.15:  # flange still low: cube may be on the table / occluded by the box
                continue
            img = cv2.imdecode(np.frombuffer(bytes(imgs[t]), np.uint8), cv2.IMREAD_COLOR)
            uv = detect(img, target_color(ep))
            if uv is not None:
                obs.append((ep, T, uv))
    return obs


def project(params, Ts):
    rvec, tvec, off = params[:3], params[3:6], params[6:9]
    X = np.stack([T[:3, :3] @ off + T[:3, 3] for T in Ts])
    uv, _ = cv2.projectPoints(X, rvec, tvec, K_MOUNT, None)
    return uv.reshape(-1, 2)


def fit(obs, off0=(0.0, 0.0, 0.16)):
    Ts = [o[1] for o in obs]
    uv = np.array([o[2] for o in obs], dtype=np.float64)
    X0 = np.stack([T[:3, :3] @ np.array(off0) + T[:3, 3] for T in Ts])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X0, uv, K_MOUNT, None, reprojectionError=12.0, iterationsCount=2000,
                                              flags=cv2.SOLVEPNP_EPNP)
    assert ok, "PnP failed"
    inl = inl.ravel()
    p0 = np.r_[rvec.ravel(), tvec.ravel(), off0]
    res = least_squares(lambda p: (project(p, [Ts[i] for i in inl]) - uv[inl]).ravel(), p0, loss="huber", f_scale=4.0)
    err = np.linalg.norm(project(res.x, Ts) - uv, axis=1)
    return res.x, err


def cam_pose(params):
    """camera->base transform (OpenCV optical frame: x right, y down, z forward)."""
    Rcw = Rotation.from_rotvec(params[:3]).as_matrix()
    t = params[3:6]
    T = np.eye(4)
    T[:3, :3] = Rcw.T
    T[:3, 3] = -Rcw.T @ t
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--gate", type=float, default=30.0, help="px; detections farther from the prediction are dropped")
    args = ap.parse_args()
    k6 = XArm6Kinematics(tip_link="link6")
    paths = sorted(glob.glob(os.path.join(args.data, "ep*/episode.h5")))
    groups = {"all": [p for p in paths if int(p.split("/")[-2][2:]) != 8]}  # ep8: gripper never closed
    out = {}
    for name, ps in groups.items():
        obs = collect(ps, k6, args.stride)
        params, err = fit(obs)
        for _ in range(3):  # gate detections around the current prediction and refit
            keep = err < args.gate
            obs = [o for o, k in zip(obs, keep) if k]
            params, err = fit(obs, off0=tuple(params[6:9]))
        T = cam_pose(params)
        eps = np.array([o[0] for o in obs])
        per_ep = {int(e): round(float(np.median(err[eps == e])), 1) for e in np.unique(eps)}
        z = T[:3, 2]
        print(f"[{name}] {len(obs)} points, reprojection median {np.median(err):.1f}px, 90% {np.percentile(err, 90):.1f}px")
        print(f"   camera position in base frame {T[:3, 3].round(3)}, viewing direction {z.round(3)}, cube offset in flange {params[6:9].round(3)}")
        print(f"   per-episode median px: {per_ep}")
        out[name] = {"T_base_cam_opencv": T.round(6).tolist(), "position": T[:3, 3].round(4).tolist(),
                     "quat_xyzw_opencv": Rotation.from_matrix(T[:3, :3]).as_quat().round(6).tolist(),
                     "cube_offset_in_flange": params[6:9].round(4).tolist(), "n_points": len(obs),
                     "reproj_median_px": round(float(np.median(err)), 2), "per_episode_median_px": per_ep,
                     "K": K_MOUNT.tolist(), "image_size": [640, 480], "camera_serial": "243222070233"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
