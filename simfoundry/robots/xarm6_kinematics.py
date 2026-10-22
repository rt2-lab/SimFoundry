# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward / inverse kinematics of the xArm6 arm up to eef_link, read from the imported URDF.

Pure numpy, no simulator: the xArm6 asset has no Lula / cuRobo descriptor, so OmniGibson's IK
controllers can't drive it, and scripted demos only need joint-space targets anyway. All poses
are in the robot base frame (link_base); quaternions are xyzw.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from simfoundry import ASSET_DIR

DEFAULT_URDF = os.path.join(ASSET_DIR, "robots", "xarm6", "xarm6.urdf")


def _origin_tf(joint):
    tf = np.eye(4)
    origin = joint.find("origin")
    if origin is not None:
        tf[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
        tf[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return tf


class XArm6Kinematics:
    def __init__(self, urdf_path=DEFAULT_URDF, base_link="link_base", tip_link="eef_link"):
        root = ET.parse(urdf_path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip_link
        while link != base_link:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        chain.reverse()

        # Per joint: fixed origin transform, then (for revolute joints) a rotation about `axis`.
        self.origins = [_origin_tf(j) for j in chain]
        self.axes = [np.array([float(v) for v in j.find("axis").get("xyz").split()]) if j.get("type") == "revolute"
                     else None for j in chain]
        self.joint_names = [j.get("name") for j in chain if j.get("type") == "revolute"]
        limits = [j.find("limit") for j in chain if j.get("type") == "revolute"]
        self.lower = np.array([float(l.get("lower")) for l in limits])
        self.upper = np.array([float(l.get("upper")) for l in limits])
        assert len(self.joint_names) == 6, self.joint_names

    def fk(self, q):
        """4x4 pose of the tip in the base frame."""
        tf, i = np.eye(4), 0
        for origin, axis in zip(self.origins, self.axes):
            tf = tf @ origin
            if axis is not None:
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
                tf = tf @ rot
                i += 1
        return tf

    def fk_pose(self, q):
        tf = self.fk(q)
        return tf[:3, 3], Rotation.from_matrix(tf[:3, :3]).as_quat()

    def _error(self, q, pos, rot):
        tf = self.fk(q)
        return np.concatenate([pos - tf[:3, 3], (rot * Rotation.from_matrix(tf[:3, :3]).inv()).as_rotvec()])

    def ik(self, pos, quat, q_seed, max_iters=200, tol_pos=1e-4, tol_rot=1e-3, damping=1e-3):
        """Damped least squares from @q_seed, clamped to joint limits.

        Returns (q, ok). Seeding from the previous waypoint keeps consecutive solutions on the
        same branch, so joint-space interpolation between them stays smooth.
        """
        rot = Rotation.from_quat(quat)
        q = np.clip(np.asarray(q_seed, dtype=float).copy(), self.lower, self.upper)
        eps = 1e-6
        for _ in range(max_iters):
            err = self._error(q, pos, rot)
            if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
                return q, True
            jac = np.empty((6, 6))
            for k in range(6):
                dq = np.zeros(6)
                dq[k] = eps
                jac[:, k] = (err - self._error(q + dq, pos, rot)) / eps
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            q = np.clip(q + step, self.lower, self.upper)
        err = self._error(q, pos, rot)
        return q, bool(np.linalg.norm(err[:3]) < 5 * tol_pos and np.linalg.norm(err[3:]) < 5 * tol_rot)

    def ik_nearest(self, pos, quat, q_seed, max_iters=50, pos_weight=50.0, ori_weight=10.0, damping=1e-3):
        """Like ik(), but returns the best iterate instead of the last one.

        A target past a joint limit or the workspace edge cannot be met; clamping every step then
        makes plain damped least squares oscillate and its last iterate can be worse than the seed.
        xarm_stack's simple_ik solves a weighted least squares (position 50, orientation 10, plus a
        joint-limit cost) and so always returns the closest reachable pose; the best iterate by the
        same weighted error is the equivalent here. Returns (q, converged, weighted_error).
        """
        rot = Rotation.from_quat(quat)
        q = np.clip(np.asarray(q_seed, dtype=float).copy(), self.lower, self.upper)
        eps = 1e-6
        w = np.r_[np.full(3, pos_weight), np.full(3, ori_weight)]
        best_q, best_cost, converged = q.copy(), np.inf, False
        for _ in range(max_iters):
            err = self._error(q, pos, rot)
            cost = float(np.linalg.norm(w * err))
            if cost < best_cost:
                best_q, best_cost = q.copy(), cost
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                converged = True
                break
            jac = np.empty((6, 6))
            for k in range(6):
                dq = np.zeros(6)
                dq[k] = eps
                jac[:, k] = (err - self._error(q + dq, pos, rot)) / eps
            q = np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err), self.lower, self.upper)
        else:
            err = self._error(q, pos, rot)
            cost = float(np.linalg.norm(w * err))
            if cost < best_cost:
                best_q, best_cost = q.copy(), cost
        return best_q, converged, best_cost
