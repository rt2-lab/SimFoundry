# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from simfoundry.pipeline.reviewed_instances import load_reviewed_instances


def make_inputs(tmp_path):
    rgb = np.full((10, 12, 3), 128, dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "source.png")
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    data = {"source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "instances": [{"name": "panel", "mask_path": "mask.png"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    return tmp_path / "manifest.json", tmp_path / "source.png"


def test_reviewed_mask_coordinates(tmp_path):
    entries = load_reviewed_instances(*make_inputs(tmp_path))
    assert entries[0]["box"] == [3, 2, 8, 7]
    assert entries[0]["mask"].sum() == 36


def test_wrong_frame_rejected(tmp_path):
    manifest, source = make_inputs(tmp_path)
    Image.new("RGB", (12, 10), "black").save(source)
    with pytest.raises(ValueError, match="source frame"):
        load_reviewed_instances(manifest, source)


@pytest.mark.parametrize("size", [(12, 10), (6, 5)])
def test_empty_or_resized_mask_rejected(tmp_path, size):
    inputs = make_inputs(tmp_path)
    Image.new("L", size, 0 if size == (12, 10) else 255).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="dimensions or empty"):
        load_reviewed_instances(*inputs)
