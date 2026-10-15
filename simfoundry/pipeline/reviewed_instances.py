# SPDX-License-Identifier: Apache-2.0
"""Validated, frame-bound SAM instance corrections for decomposition."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_reviewed_instances(manifest_path, source_path):
    """Load ordered masks; reject corrections authored against another frame."""
    path = Path(manifest_path)
    document = json.loads(path.read_text())
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"))
    digest = hashlib.sha256(source.tobytes()).hexdigest()
    if document.get("source_rgb_sha256") != digest:
        raise ValueError("Reviewed instances do not match the decomposition source frame")
    entries = document.get("instances", [])
    if not entries:
        raise ValueError("Reviewed instance manifest is empty")
    result = []
    for entry in entries:
        name = entry.get("name", "").strip()
        if not name:
            raise ValueError("Reviewed instance requires a name")
        mask_path = path.parent / entry["mask_path"]
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != source.shape[:2] or not mask.any():
            raise ValueError(f"Invalid reviewed mask dimensions or empty mask: {mask_path}")
        yy, xx = np.where(mask)
        result.append({
            "name": name, "mask": mask,
            "box": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
            "provenance": entry.get("provenance", {}),
        })
    return result
