# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Match the cube2 sim scene to the real xArm6 rig (2026-10 autoalloc recordings).

Measured on the real data (scripts/xarm6/real/*):

- cameras: scripts/cfg/external_sensors/sim_xarm6_real.yaml (D435i intrinsics, estimated extrinsics);
- layout: scripts/cfg/real/xarm6_real_layout.json (per-episode box pose, cube positions, start joints);
- objects: the real box is the reconstructed toolbox at 0.70x (outer 0.32 x 0.24 m; 0.58x from the
  rim corners looked ~20% small next to the real frames, the mesh's latches pad its AABB), the cubes
  are 25 mm. The real (transparent) lid is hinged on a long side and rests leaning back outward;
  the s9 articulation (mobility.urdf) was corrected to that hinge and stages 11-14 rebuilt, so the
  lid is a colliding link. Each episode opens it to 125-165 deg and makes that its stop; it gets a
  translucent material. The box is turned so its inner compartments sit where the real ones do
  (narrow compartment on the side away from the lid);
- tabletop: wood, far edge at x = 0.55 m in the base frame, black breadboard at y > 0.30 m, carpet
  beyond the table;
- gripper: raw units 0 (closed) .. 850 (open) = finger opening in 0.1 mm.

Also writes episodes in the real recorder's episode.h5 schema (write_real_episode), so sim and real
go through one converter (scripts/xarm6/convert_real_h5_to_lerobot.py).

Import after OmniGibson.
"""

import json
import math

import cv2
import h5py
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation

BOX_SCALE = 0.70
CUBE_SIZE = 0.025
# Measured extents of the reconstructed cubes (m), for the scale that makes them CUBE_SIZE.
CUBE_NATIVE_SIZE = {"blue_cube": 0.041, "yellow_cube": 0.0393}
# The sim box's narrow compartment is on its local -y side, the real one on the side away from the
# lid; the layout file's box yaw puts the lid on local -y, so the sim box is turned half a turn.
BOX_YAW_OFFSET = math.pi
GRIPPER_RAW_OPEN = 850.0
TABLE_FAR_EDGE_X = 0.55
BREADBOARD_Y = 0.30
CUBE_REGION = ((0.10, 0.48), (-0.20, 0.22))  # base-frame x, y ranges cubes are drawn from
# Solid cube colours (linear RGB) from the real frames (sRGB ~(10, 110, 190) and ~(225, 190, 45)):
# the reconstructed yellow cube's texture is pale with green blotches.
CUBE_COLORS = {"blue_cube": (0.003, 0.155, 0.51), "yellow_cube": (0.75, 0.51, 0.026)}
FINGER_COLOR = (0.02, 0.02, 0.02)  # the real fingertips are black
LIGHT_WOODS = ("light_oak", "ash", "beech")
# The real lid rests leaning back outward: depth puts its free edge ~11 cm above the rim (~150 deg).
LID_OPEN_DEG = (125.0, 165.0)
# Plastic organiser: the asset gives base and lid 1 kg each, and a 1 kg lid leaning back at 150 deg
# tips the box over (41 deg). Estimates (not weighed): the clear lid is much lighter than the box.
BOX_BASE_MASS = 0.8
BOX_LID_MASS = 0.2
LID_COLOR = (0.85, 0.90, 0.95)  # clear plastic with a slight blue tint
LID_OPACITY = 0.6


def make_real_scene_json(src, dst, box_category="toolbox", box_scale=BOX_SCALE):
    """Copy the stage-14 scene JSON with the box and cubes rescaled to the real objects."""
    with open(src) as f:
        scene = json.load(f)
    for name, info in scene["objects_info"]["init_info"].items():
        cat = info["args"].get("category")
        if cat == box_category:
            info["args"]["scale"] = [box_scale] * 3
        elif cat in CUBE_NATIVE_SIZE:
            info["args"]["scale"] = [CUBE_SIZE / CUBE_NATIVE_SIZE[cat]] * 3
        else:
            continue
        reg = scene["state"]["registry"]["object_registry"][name]
        reg["root_link"]["lin_vel"] = [0.0, 0.0, 0.0]
        reg["root_link"]["ang_vel"] = [0.0, 0.0, 0.0]
    with open(dst, "w") as f:
        json.dump(scene, f, indent=2)
    return dst


class RealScene:
    """Places the box / cubes from base-frame layouts and holds the bits of the real scene the
    reconstruction lacks. Create after the first env reset (calibrates against the loaded poses)."""

    def __init__(self, env, box, cubes, box_link="toolbox_base_link", lid_link="lid_link",
                 lid_joint="toolbox_base_link_to_lid_link"):
        self.env, self.box, self.cubes, self.box_link = env, box, list(cubes), box_link
        robot = env.robots[0]
        pos, quat = (x.cpu().numpy() for x in robot.get_position_orientation())
        self.base_t, self.base_r = pos, Rotation.from_quat(quat)
        self.base_yaw = self.base_r.as_euler("xyz")[2]
        # Box: base-link centre in the root frame and how far below the root the base link reaches.
        root_p, root_q = (x.cpu().numpy() for x in box.get_position_orientation())
        lo, hi = (x.cpu().numpy() for x in box.links[box_link].aabb)
        r_root = Rotation.from_quat(root_q)
        self.box_center_local = r_root.inv().apply(0.5 * (lo + hi) - root_p)
        self.box_root_z = root_p[2] - lo[2] + 0.002
        ext = r_root.inv().apply(hi - lo)
        self.box_half = 0.5 * np.abs(ext)  # local-frame half extents (box is axis-aligned at load)
        self.box_tilt = Rotation.from_euler("z", -r_root.as_euler("xyz")[2]) * r_root  # keep any non-yaw part
        self.cube_nominal = {c.name: c.get_position_orientation()[1].clone() for c in cubes}
        self.cube_half = {c.name: 0.5 * float((c.aabb[1] - c.aabb[0])[2]) for c in cubes}
        self.box_pose_b = None
        self._materials = {}
        self.lid_deg = float(np.mean(LID_OPEN_DEG))
        self.lid = box.joints.get(lid_joint)
        box.links[box_link].mass = BOX_BASE_MASS
        if lid_link in box.links:
            box.links[lid_link].mass = BOX_LID_MASS
        if self.lid is not None:
            # The asset's angular drive would pull the lid toward its target; the real lid is free
            # on its hinge (only gravity and the stop hold it).
            self.lid.stiffness = 0.0
            self.lid.damping = 0.05
            self._paint(box.links[lid_link], "real_lid", LID_COLOR, opacity=LID_OPACITY)
        for link in ("left_finger", "right_finger"):
            self._paint(robot.links[link], "real_finger", FINGER_COLOR)
        self._paint_cubes()

    def _paint(self, link, name, color, opacity=None):
        """Bind a solid-colour material to @link (once) and set its colour.

        Binding edits the rigid body's USD prim, which makes PhysX re-read it and snap the body back
        to its authored pose, so it happens only here, before any object is placed; per-episode
        colour changes go through recolor_cubes(), which only edits the material."""
        import omnigibson as og
        from omnigibson.prims.material_prim import OmniPBRMaterialPrim
        from pxr import UsdShade

        if name not in self._materials:
            self._materials[name] = OmniPBRMaterialPrim.get_material(
                scene=self.env.scene, name=name, prim_path=f"{self.env.scene.prim_path}/Looks/{name}")
        mat = self._materials[name]
        mat.diffuse_texture = ""
        mat.diffuse_color_constant = th.tensor(color, dtype=th.float32)
        if opacity is not None:
            import carb

            mat.set_input("enable_opacity", True)
            mat.set_input("opacity_constant", float(opacity))
            mat.set_input("reflection_roughness_constant", 0.1)
            # RTX real-time renders cutout opacity as on/off unless fractional opacity is enabled.
            carb.settings.get_settings().set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        # On the link, stronger than descendants: the reconstructed meshes carry their own (texture)
        # bindings, which a plain binding on the mesh does not override.
        prim = og.sim.stage.GetPrimAtPath(link.prim_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(og.sim.stage.GetPrimAtPath(mat.prim_path)),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def _paint_cubes(self):
        for c in self.cubes:
            for link in c.links.values():
                self._paint(link, f"real_{c.category}", CUBE_COLORS.get(c.category, (0.5, 0.5, 0.5)))

    def recolor_cubes(self, rng=None):
        """Real cube colours, optionally jittered (brightness / hue) per episode. Material edit only."""
        for c in self.cubes:
            col = np.asarray(CUBE_COLORS.get(c.category, (0.5, 0.5, 0.5)))
            if rng is not None:
                col = np.clip(col * rng.uniform(0.75, 1.25) * rng.uniform(0.9, 1.1, 3), 0, 1)
            self._materials[f"real_{c.category}"].diffuse_color_constant = th.tensor(col, dtype=th.float32)

    def disturbance(self):
        """(lid angle in deg, box base-frame xy shift in m, box yaw change in deg) relative to the
        placed layout: nonzero when the arm or a cube pushed the lid or the box."""
        lid = math.degrees(float(self.lid.get_state()[0][0])) if self.lid is not None else 0.0
        p, q = (x.cpu().numpy() for x in self.box.links[self.box_link].get_position_orientation())
        c = self.to_base(p)[:2]
        yaw = Rotation.from_quat(q).as_euler("xyz")[2] - self.base_yaw - BOX_YAW_OFFSET
        dyaw = math.degrees((yaw - self.box_pose_b[1] + math.pi) % (2 * math.pi) - math.pi)
        return lid, float(np.linalg.norm(c - self.box_pose_b[0])), dyaw

    def set_lid(self, deg):
        """Open the lid to @deg and make that its stop (upper limit): past 90 deg it leans back on the
        stop under gravity, like the real lid, and an arm that bumps it pushes it physically."""
        if self.lid is None:
            return
        self.lid_deg = float(deg)
        rad = math.radians(self.lid_deg)
        self.lid.upper_limit = rad
        self.lid.set_pos(rad)

    @property
    def box_size(self):
        """(long, short) footprint of the loaded box (m)."""
        return tuple(2 * np.sort(self.box_half[:2])[::-1])

    # --- frames -------------------------------------------------------------------------------
    def to_world(self, p_b):
        return self.base_r.apply(np.asarray(p_b, dtype=float)) + self.base_t

    def to_base(self, p_w):
        return self.base_r.inv().apply(np.asarray(p_w, dtype=float) - self.base_t)

    # --- placement ----------------------------------------------------------------------------
    def place_box(self, center_b, yaw_b, lid_deg=None):
        if lid_deg is not None:
            self.lid_deg = float(lid_deg)
        yaw_w = self.base_yaw + yaw_b + BOX_YAW_OFFSET
        r = Rotation.from_euler("z", yaw_w) * self.box_tilt
        c_w = self.to_world([center_b[0], center_b[1], 0.0])
        root = c_w - r.apply(self.box_center_local)
        root[2] = self.box_root_z
        self.box.set_position_orientation(position=th.tensor(root, dtype=th.float32),
                                          orientation=th.tensor(r.as_quat(), dtype=th.float32))
        self.box.keep_still()
        self.box_pose_b = (np.asarray(center_b[:2], dtype=float), float(yaw_b))
        self.set_lid(self.lid_deg)

    def place_cube(self, cube, xy_b, yaw):
        p = self.to_world([xy_b[0], xy_b[1], 0.0])
        p[2] = self.cube_half[cube.name] + 0.003
        ori = Rotation.from_euler("z", yaw) * Rotation.from_quat(self.cube_nominal[cube.name].numpy())
        cube.set_position_orientation(position=th.tensor(p, dtype=th.float32),
                                      orientation=th.tensor(ori.as_quat(), dtype=th.float32))
        cube.keep_still()

    def inside_box(self, p_b, margin=0.0):
        """Is base-frame point @p_b inside the box footprint (shrunk by @margin) and below its rim?"""
        c, yaw = self.box_pose_b
        d = Rotation.from_euler("z", -yaw).apply(np.r_[np.asarray(p_b[:2]) - c, 0.0])
        half = np.sort(self.box_half[:2])[::-1]  # (long, short): layout yaw = long side
        return bool(abs(d[0]) < half[0] - margin and abs(d[1]) < half[1] - margin and p_b[2] < 2 * self.box_half[2])

    # --- surroundings -------------------------------------------------------------------------
    def add_surroundings(self, rng=None, table_mat=None):
        """Shrink the randomizer's table quad to the real table, add the black breadboard on its
        left and a carpet below/behind it. Visual only. Call after DomainRandomizer() (which makes
        the table quad) and again per episode to jitter the edges."""
        import omnigibson as og
        from pxr import Gf, UsdGeom, UsdShade, Sdf

        stage = og.sim.stage
        jit = (lambda s: float(rng.uniform(-s, s))) if rng is not None else (lambda s: 0.0)
        far = TABLE_FAR_EDGE_X + jit(0.03)
        board = BREADBOARD_Y + jit(0.03)

        def quad(path, x0, x1, y0, y1, z, color=None):
            corners = [self.to_world([x, y, 0.0]) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(z)) for p in corners])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
            mesh.SetNormalsInterpolation("vertex")
            mesh.CreateDoubleSidedAttr(True)
            if color is not None:
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in color])])
                mat_path = path + "_mat"
                mat = UsdShade.Material.Define(stage, mat_path)
                shader = UsdShade.Shader.Define(stage, mat_path + "/shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            return mesh

        table = stage.GetPrimAtPath("/World/rand_table")
        if table.IsValid():  # keep its UVs: only the points move
            corners = [self.to_world([x, y, 0.0]) for x, y in ((-0.8, -1.2), (far, -1.2), (far, 1.2), (-0.8, 1.2))]
            UsdGeom.Mesh(table).GetPointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), 3e-4) for p in corners])
        dark = np.clip(np.array([0.05, 0.05, 0.055]) * (rng.uniform(0.8, 1.4) if rng is not None else 1.0), 0, 1)
        quad("/World/real_breadboard", -0.8, far, board, 1.2, 0.006, dark)
        carpet = np.clip(np.array([0.10, 0.10, 0.12]) * (rng.uniform(0.7, 1.3) if rng is not None else 1.0), 0, 1)
        quad("/World/real_carpet", -3.0, 4.0, -4.0, 4.0, -0.72, carpet)
        return {"table_far_edge_x": round(far, 3), "breadboard_y": round(board, 3)}


class RealLayoutSampler:
    """Draws box / cube layouts and start joints around the recorded real episodes."""

    def __init__(self, layout_json, rng, box_size=None):
        """box_size: (long, short) footprint used to keep cubes off the box; default: the file's."""
        with open(layout_json) as f:
            data = json.load(f)
        self.eps = [e for e in data["episodes"] if "box" in e]
        self.rng = rng
        self.box_size = box_size if box_size is not None else data["box_size"][:2]

    def _in_footprint(self, xy, box_c, box_yaw, margin):
        d = Rotation.from_euler("z", -box_yaw).apply(np.r_[np.asarray(xy) - box_c, 0.0])
        return abs(d[0]) < self.box_size[0] / 2 + margin and abs(d[1]) < self.box_size[1] / 2 + margin

    def sample(self, target_color):
        rng = self.rng
        (xr, yr) = CUBE_REGION
        for _ in range(200):
            eb = self.eps[rng.integers(len(self.eps))]
            box_c = np.asarray(eb["box"]["center"]) + rng.normal(0, 0.015, 2)
            box_yaw = eb["box"]["yaw"] + rng.normal(0, math.radians(4))
            et = self.eps[rng.integers(len(self.eps))]
            target = np.asarray(et["target_cube"][:2]) + rng.normal(0, 0.02, 2)
            others = [e["other_cube"] for e in self.eps if e.get("other_cube")]
            if others and rng.random() < 0.5:
                other = np.asarray(others[rng.integers(len(others))]) + rng.normal(0, 0.03, 2)
            else:
                other = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            ok = all(xr[0] <= p[0] <= xr[1] and yr[0] <= p[1] <= yr[1] for p in (target, other))
            ok = ok and not any(self._in_footprint(p, box_c, box_yaw, 0.04) for p in (target, other))
            ok = ok and np.linalg.norm(target - other) > 0.06
            if ok:
                break
        es = self.eps[rng.integers(len(self.eps))]
        start_q = np.asarray(es["start_q"]) + rng.normal(0, 0.03, 6)
        end_q = np.asarray(self.eps[rng.integers(len(self.eps))]["start_q"]) + rng.normal(0, 0.03, 6)
        return {"box_center": box_c.round(4).tolist(), "box_yaw": round(float(box_yaw), 4),
                "target_xy": target.round(4).tolist(), "other_xy": other.round(4).tolist(),
                "target_color": target_color, "start_q": start_q.round(4).tolist(),
                "end_q": end_q.round(4).tolist(), "lid_deg": round(float(rng.uniform(*LID_OPEN_DEG)), 1),
                "source_eps": {"box": eb["ep"], "target": et["ep"], "start": es["ep"]}}


def gripper_opening(robot):
    """Distance between the two inner pad faces (m), mid-pad."""
    pts = []
    for link, y in (("left_finger", -0.026), ("right_finger", 0.026)):
        p, q = (x.cpu().numpy() for x in robot.links[link].get_position_orientation())
        pts.append(Rotation.from_quat(q).apply([0.0, y, 0.03]) + p)
    return float(np.linalg.norm(pts[0] - pts[1]))


def gripper_raw(robot):
    """Real-robot gripper units: 0 closed .. 850 open (= opening in 0.1 mm)."""
    return float(np.clip(gripper_opening(robot) * 1e4, 0.0, GRIPPER_RAW_OPEN))


def command_to_raw(g):
    """Sim gripper command (0 open .. 1 closed) -> real raw command."""
    return float(GRIPPER_RAW_OPEN * (1.0 - g))


class RealEpisodeRecorder:
    """Collects one episode in the real recorder's episode.h5 layout (20 Hz, JPEG images, joint
    angles in rad, link6 pose xyz + quat xyzw, gripper in raw units)."""

    def __init__(self, kin_link6, dt, jpeg_quality=90):
        self.kin, self.dt, self.q = kin_link6, dt, jpeg_quality
        self.rows = {k: [] for k in ("robot_current_joint_angles", "robot_desired_joint_angles",
                                     "robot_current_joint_velocity", "robot_desired_joint_velocity",
                                     "robot_current_pose", "robot_desired_pose",
                                     "gripper_actual_position", "gripper_commanded_position",
                                     "mount_image_rgb_compressed", "gripper_image_rgb_compressed")}
        self._prev_cmd = None
        self.extras = {}  # sim-only per-step signals, written as extra datasets (sim_<name>)

    def _pose(self, q):
        p, quat = self.kin.fk_pose(np.asarray(q, dtype=float))
        return np.r_[p, quat].astype(np.float32)

    def _jpeg(self, rgb):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.q])
        assert ok
        return buf.ravel()

    def add(self, q_cur, dq_cur, q_cmd, grip_raw, grip_cmd_raw, front_rgb, wrist_rgb, extras=None):
        q_cmd = np.asarray(q_cmd, dtype=np.float32)
        dq_cmd = np.zeros(6, np.float32) if self._prev_cmd is None else (q_cmd - self._prev_cmd) / self.dt
        self._prev_cmd = q_cmd
        r = self.rows
        r["robot_current_joint_angles"].append(np.asarray(q_cur, np.float32))
        r["robot_desired_joint_angles"].append(q_cmd)
        r["robot_current_joint_velocity"].append(np.asarray(dq_cur, np.float32))
        r["robot_desired_joint_velocity"].append(dq_cmd.astype(np.float32))
        r["robot_current_pose"].append(self._pose(q_cur))
        r["robot_desired_pose"].append(self._pose(q_cmd))
        r["gripper_actual_position"].append(np.float32(grip_raw))
        r["gripper_commanded_position"].append(np.float32(grip_cmd_raw))
        r["mount_image_rgb_compressed"].append(self._jpeg(front_rgb))
        r["gripper_image_rgb_compressed"].append(self._jpeg(wrist_rgb))
        for k, v in (extras or {}).items():
            self.extras.setdefault(k, []).append(np.asarray(v, np.float32))

    def __len__(self):
        return len(self.rows["gripper_actual_position"])

    def write(self, path, attrs):
        n = len(self)
        with h5py.File(path, "w") as f:
            f.create_dataset("timestamps", data=(np.arange(n) + 1) * self.dt)
            for k, v in self.rows.items():
                if k.endswith("_compressed"):
                    ds = f.create_dataset(k, (n,), dtype=h5py.vlen_dtype(np.uint8))
                    for i, b in enumerate(v):
                        ds[i] = b
                else:
                    f.create_dataset(k, data=np.stack(v))
            for k, v in self.extras.items():
                f.create_dataset(f"sim_{k}", data=np.stack(v))
            for k, v in attrs.items():
                f.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
