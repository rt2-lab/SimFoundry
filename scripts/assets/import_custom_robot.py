# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run OmniGibson's custom-robot importer, writing the USD under SimFoundry's assets/ tree.

The upstream example (omnigibson/examples/robots/import_custom_robot.py) calls
``import_og_asset_from_urdf`` without its required ``dataset_name`` argument, so it fails as
shipped. deps/ must not be edited (AGENTS.md), so this wrapper injects the argument instead.
``get_dataset_path`` is ``os.path.join(gm.DATA_PATH, dataset_name)``, so an absolute
``dataset_name`` places the asset outside the OmniGibson data tree.

Usage:
    python scripts/assets/import_custom_robot.py --config assets/robots/xarm6/import_xarm6.yaml \
        --dataset-root /abs/path/to/SimFoundry/assets/og_custom
"""

import argparse
import functools
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True, help="Absolute directory to write the imported asset into")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    os.makedirs(dataset_root, exist_ok=True)

    from omnigibson.examples.robots import import_custom_robot as importer

    importer.import_og_asset_from_urdf = functools.partial(
        importer.import_og_asset_from_urdf, dataset_name=dataset_root
    )
    sys.argv = [sys.argv[0], "--config", os.path.abspath(args.config)]
    importer.import_custom_robot()


if __name__ == "__main__":
    main()
