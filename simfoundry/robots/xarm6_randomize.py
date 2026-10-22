# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode domain randomization for the xArm6 demo generator.

The sim floor stands in for a tabletop, and the real table is light oak, so a plain white floor
is a large visual gap for a vision policy. Each call to DomainRandomizer.apply() draws:

- tabletop: mostly procedural wood (tabletop_textures.py: oak / beech / pine / ash / walnut /
  cherry, random texture, tile scale, rotation, tint, roughness), sometimes a plain matte colour;
- lighting: dome-light intensity and colour temperature;
- cameras: front camera pose / focal length, wrist camera pose (small jitter around the nominal
  mount, covering calibration error between sim and the real rig).

Returns the drawn parameters so they can be logged with the episode. Import after OmniGibson.
"""

import math
import os

import numpy as np
import torch as th
from scipy.spatial.transform import Rotation

import omnigibson as og
from omnigibson.prims.material_prim import OmniPBRMaterialPrim

from simfoundry import ASSET_DIR
from simfoundry.robots.tabletop_textures import generate_wood_textures

TEXTURE_DIR = os.path.join(ASSET_DIR, "textures", "tabletop")
PLAIN_COLORS = {  # matte mats / painted tables
    "white": (0.85, 0.85, 0.85), "light_grey": (0.62, 0.62, 0.64), "dark_grey": (0.22, 0.22, 0.24),
    "black": (0.07, 0.07, 0.07), "green_mat": (0.20, 0.42, 0.28), "blue_mat": (0.20, 0.30, 0.50),
}


class DomainRandomizer:
    def __init__(self, env, rng, p_wood=0.8, cam_jitter=True, light_jitter=True,
                 front_jitter=(0.03, 3.0), wrist_jitter=(0.004, 2.0), focal_jitter=0.06, wood_species=None):
        """front_jitter / wrist_jitter: (position m, angle deg) half-ranges; focal_jitter: relative.
        The defaults cover a guessed rig; with calibrated cameras (sim_xarm6_real) use smaller ones.
        wood_species: restrict the wood textures to these palettes (e.g. the real table's light woods)."""
        self.env, self.rng = env, rng
        self.front_jitter, self.wrist_jitter, self.focal_jitter = front_jitter, wrist_jitter, focal_jitter
        self.p_wood, self.cam_jitter, self.light_jitter = p_wood, cam_jitter, light_jitter
        self.textures = generate_wood_textures(TEXTURE_DIR)
        if wood_species:
            self.textures = [t for t in self.textures if any(f"wood_{w}_" in os.path.basename(t) for w in wood_species)]
            assert self.textures, f"no wood textures for {wood_species}"
        self.floor_mat = OmniPBRMaterialPrim.get_material(
            scene=env.scene, name="rand_tabletop", prim_path=f"{env.scene.prim_path}/Looks/rand_tabletop")
        self._make_table_mesh(self.floor_mat)
        self._base_cams = {n: s.get_position_orientation(frame="parent") for n, s in env.external_sensors.items()}
        self._base_focal = {n: float(s.focal_length) for n, s in env.external_sensors.items()}
        self._base_light = (float(og.sim.skybox.intensity), th.as_tensor(og.sim.skybox.color).clone())

    TABLE_SIZE = 6.0  # metres; texture_scale = TABLE_SIZE / (tile size in metres)

    @staticmethod
    def _make_table_mesh(material, path="/World/rand_table"):
        """Visual-only UV quad 0.3 mm above the floor plane, whose own visuals are hidden.

        The stock ground-plane mesh has no UVs, and its material is bound on the parent prim, which
        wins over any binding we put on the mesh, so it can't be retextured in place."""
        from pxr import Gf, UsdGeom, Vt

        stage = og.sim.stage
        half = DomainRandomizer.TABLE_SIZE / 2
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(-half, -half, 3e-4), (half, -half, 3e-4), (half, half, 3e-4), (-half, half, 3e-4)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation("vertex")
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf_TexCoord2f(), UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
        material.bind(path)
        UsdGeom.Imageable(stage.GetPrimAtPath(og.sim.floor_plane.prim_path + "/geom")).MakeInvisible()

    def _tabletop(self):
        rng, m = self.rng, self.floor_mat
        if rng.random() < self.p_wood:
            tex = self.textures[rng.integers(len(self.textures))]
            m.diffuse_texture = tex
            tint = np.clip(rng.normal(1.0, 0.06, 3), 0.8, 1.2)
            m.diffuse_tint = th.tensor(tint, dtype=th.float32)
            desc = {"kind": "wood", "texture": os.path.basename(tex), "tint": tint.round(3).tolist()}
            from pxr import Gf

            tile_m = float(rng.uniform(0.25, 0.9))  # one texture repeat spans this many metres
            m.set_input("texture_scale", Gf.Vec2f(self.TABLE_SIZE / tile_m, self.TABLE_SIZE / tile_m))
            m.set_input("texture_rotate", float(rng.uniform(-15, 15) if rng.random() < 0.5 else rng.uniform(75, 105)))
        else:
            name = list(PLAIN_COLORS)[rng.integers(len(PLAIN_COLORS))]
            m.diffuse_texture = ""
            col = np.clip(np.asarray(PLAIN_COLORS[name]) * rng.uniform(0.85, 1.15), 0, 1)
            m.diffuse_color_constant = th.tensor(col, dtype=th.float32)
            desc = {"kind": "plain", "color": name}
        rough = float(rng.uniform(0.25, 0.8))
        m.set_input("reflection_roughness_constant", rough)
        desc["roughness"] = round(rough, 3)
        return desc

    def _lighting(self):
        rng = self.rng
        # The stock sky (2500) overexposes light surfaces; 0.25-0.9 of it keeps wood grain visible.
        intensity = self._base_light[0] * float(rng.uniform(0.25, 0.9))
        kelvin = float(rng.uniform(3500, 7500))            # warm indoor .. cool daylight
        r, g, b = _kelvin_to_rgb(kelvin)
        og.sim.skybox.intensity = intensity
        og.sim.skybox.color = (r, g, b)
        return {"intensity": round(intensity), "kelvin": round(kelvin)}

    def _cameras(self):
        rng, out = self.rng, {}
        for name, sensor in self.env.external_sensors.items():
            pos, quat = self._base_cams[name]
            dpos, dang = self.front_jitter if name.startswith("front") else self.wrist_jitter
            dp = rng.uniform(-1, 1, 3) * dpos
            ang = rng.uniform(-1, 1, 3) * math.radians(dang)
            new_quat = (Rotation.from_euler("xyz", ang) * Rotation.from_quat(quat.numpy())).as_quat()
            sensor.set_position_orientation(position=pos + th.tensor(dp, dtype=th.float32),
                                            orientation=th.tensor(new_quat, dtype=th.float32), frame="parent")
            scale = float(rng.uniform(1 - self.focal_jitter, 1 + self.focal_jitter))
            sensor.focal_length = self._base_focal[name] * scale
            out[name] = {"dpos_cm": (dp * 100).round(2).tolist(), "dang_deg": np.degrees(ang).round(2).tolist(),
                         "focal_scale": round(scale, 3)}
        return out

    def apply(self):
        info = {"tabletop": self._tabletop()}
        if self.light_jitter:
            info["light"] = self._lighting()
        if self.cam_jitter:
            info["cameras"] = self._cameras()
        for _ in range(3):
            og.sim.render()
        return info


def Sdf_TexCoord2f():
    from pxr import Sdf

    return Sdf.ValueTypeNames.TexCoord2fArray


def _kelvin_to_rgb(k):
    """Tanner Helland's blackbody approximation, normalised so the brightest channel is 1."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    rgb = np.clip([r, g, b], 0, 255) / 255.0
    return tuple((rgb / rgb.max()).tolist())
