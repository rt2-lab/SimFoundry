# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-import repair of the xArm6 USD produced by scripts/assets/import_custom_robot.py.

Isaac's URDF importer authors a mimic joint's ``referenceJoint`` and limits only if the joint it
mimics already exists. right_inner_knuckle_joint is created before right_outer_knuckle_joint,
so it comes out with a PhysxMimicJointAPI but no reference and no limits: a free, unbounded
hinge that flops inside the gripper (seen as a ±1000 limit and a joint stuck near 0 while its
siblings close). This copies the reference and limits over from its mirror-image joint.

It also fixes two importer defaults that break physical grasping:

- Drives are authored as "acceleration" drives, whose stiffness is scaled by the link
  inertia. The gripper links weigh 20-50 g, so the fingers close normally in free air but
  squeeze an object with ~1e-3 N m and it slides out on lift. All drives become "force" drives
  (stiffness in N m / rad, as on OmniGibson's own Franka); gains come from
  simfoundry/tasks/macros.py GAINS["XArm6"].
- Mimic joints are compliant springs (naturalFrequency 25, dampingRatio 0.005), which let the
  fingers rotate off the four-bar linkage under load. naturalFrequency 0 makes them rigid.

Runs with standalone usd-core (no Isaac Sim), e.g. in the simfoundry-editor env:

    mamba run -n simfoundry-editor python scripts/assets/fix_xarm6_usd.py \
        --usd assets/og_custom/objects/robot/xarm6/usd/xarm6.usda
"""

import argparse

from pxr import Sdf, Usd

ROBOT = "/xarm6"
REPAIRS = {
    # broken joint: (joint it mimics, mirror-image joint whose limits it shares)
    "right_inner_knuckle_joint": ("right_outer_knuckle_joint", "left_inner_knuckle_joint"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.usd)
    for broken, (reference, mirror) in REPAIRS.items():
        joint = stage.GetPrimAtPath(f"{ROBOT}/joints/{broken}")
        mirror_prim = stage.GetPrimAtPath(f"{ROBOT}/joints/{mirror}")
        assert joint.IsValid() and mirror_prim.IsValid(), f"missing {broken} or {mirror}"

        rel = joint.CreateRelationship("physxMimicJoint:rotX:referenceJoint", custom=False)
        rel.SetTargets([Sdf.Path(f"{ROBOT}/joints/{reference}")])
        for attr_name in ("physics:lowerLimit", "physics:upperLimit"):
            value = mirror_prim.GetAttribute(attr_name).Get()
            joint.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
        print(f"{broken}: referenceJoint -> {reference}, limits copied from {mirror}")

    n_drives = n_mimics = 0
    for prim in stage.Traverse():
        drive_type = prim.GetAttribute("drive:angular:physics:type")
        if drive_type and drive_type.IsValid():
            drive_type.Set("force")
            n_drives += 1
        natural_frequency = prim.GetAttribute("physxMimicJoint:rotX:naturalFrequency")
        if natural_frequency and natural_frequency.IsValid():
            natural_frequency.Set(0.0)
            prim.GetAttribute("physxMimicJoint:rotX:dampingRatio").Set(0.0)
            n_mimics += 1
    print(f"{n_drives} drives -> force, {n_mimics} mimic joints -> rigid")

    stage.GetRootLayer().Save()


if __name__ == "__main__":
    main()
