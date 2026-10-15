# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check an xArm6 task scene before eval / demo generation: the box lid joint exists and
is open after reset (and stays open), objects are within IK reach, and the policy cameras see
the workspace. Saves front.png, wrist.png, overview.png and report.json to --out-dir.

    OMNIGIBSON_HEADLESS=1 mamba run -n simfoundry python scripts/xarm6/check_scene.py \
        --scene-json Data/cube2_scene/s14_og/reconstructed_og_scene.json \
        --task cube2_pick_place_cube_in_box --out-dir /tmp/xarm6_check
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sensors", default="sim_xarm6_front_wrist")
    ap.add_argument("--box-category", default="toolbox")
    ap.add_argument("--lid-joint", default="toolbox_base_link_to_lid_link")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import omnigibson as og
    import torch as th
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from simfoundry.robots.xarm6_env import articulated_joint, make_xarm6_env, reset_xarm6_env, xarm6_images
    from simfoundry.robots.xarm6_kinematics import XArm6Kinematics

    env = make_xarm6_env(args.scene_json, args.task, args.out_dir, sensors_cfg=args.sensors)
    robot = env.robots[0]
    lid = articulated_joint(env, args.box_category, args.lid_joint)
    hold = reset_xarm6_env(env)
    report = {"lid_after_reset": float(lid.get_state()[0]),
              "lid_limits": [float(lid.lower_limit), float(lid.upper_limit)]}
    for _ in range(90):  # 3 s: does the lid stay open on its own?
        env.step(hold)
    report["lid_after_3s"] = float(lid.get_state()[0])

    base_pos, base_quat = (x.cpu().numpy() for x in robot.get_position_orientation())
    to_base = lambda p: Rotation.from_quat(base_quat).inv().apply(np.asarray(p) - base_pos)  # noqa: E731
    kin = XArm6Kinematics()
    down = Rotation.from_matrix(np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])).as_quat()
    home = robot._default_joint_pos[:6].cpu().numpy()
    report["objects"] = {}
    for obj in env.scene.objects:
        if obj is robot or obj.category in ("floors", "agent"):
            continue
        lo, hi = (x.cpu().numpy() for x in obj.aabb)
        p = to_base(0.5 * (lo + hi))
        reach = {}
        for dz in (0.0, 0.15):
            _, ok = kin.ik(p + [0, 0, dz + 0.5 * (hi - lo)[2]], down, home)
            reach[f"tcp_down_at_top+{dz:.2f}"] = ok
        report["objects"][obj.name] = {"category": obj.category, "center_base": p.round(3).tolist(),
                                       "extent": (hi - lo).round(3).tolist(), "ik_reachable": reach}
        if obj.category == args.box_category:
            for name, link in obj.links.items():
                llo, lhi = (x.cpu().numpy() for x in link.aabb)
                report["objects"][obj.name][f"link:{name}"] = {"lo_base": to_base(llo).round(3).tolist(),
                                                               "hi_base": to_base(lhi).round(3).tolist()}

    for name, img in xarm6_images(env).items():
        Image.fromarray(img).save(f"{args.out_dir}/{name}.png")

    from omnigibson.sensors import VisionSensor

    eye = base_pos + Rotation.from_quat(base_quat).apply([0.35, -0.9, 0.9])
    target = base_pos + Rotation.from_quat(base_quat).apply([0.4, 0.0, 0.05])
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    cam_quat = Rotation.from_matrix(np.stack([x, np.cross(z, x), z], axis=1)).as_quat()
    og.sim.viewer_camera.set_position_orientation(th.tensor(eye, dtype=th.float32), th.tensor(cam_quat, dtype=th.float32))
    for _ in range(3):
        og.sim.render()
    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][..., :3].cpu().numpy()
    Image.fromarray(rgb).save(f"{args.out_dir}/overview.png")

    with open(f"{args.out_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    og.shutdown()


if __name__ == "__main__":
    main()
