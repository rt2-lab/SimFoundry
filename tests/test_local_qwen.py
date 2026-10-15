# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from simfoundry.models.local_qwen import LocalQwen
from simfoundry.models.text_vlm import create_text_vlm


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setenv("SIMFOUNDRY_QWEN_MODEL_PATH", str(path))
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CACHE_MODE", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    return path


def test_local_provider_needs_no_gemini_credentials(model_dir, monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_TEXT_VLM", "qwen")
    model = create_text_vlm(project=None, model="gemini-3.1-pro-preview")
    assert isinstance(model, LocalQwen)
    with pytest.raises(NotImplementedError, match="text-output only"):
        model.get_result_images({})


def test_cache_replay_and_image_content_invalidation(model_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_MODE", "1")
    calls = []

    def infer(self, request):
        calls.append(request)
        return {"text": '{"count":6}', "finish_reason": "stop"}

    monkeypatch.setattr(LocalQwen, "_infer", infer)
    image = tmp_path / "image.png"
    image.write_bytes(b"first content")
    model = LocalQwen()
    first = model("count", image_paths=image)
    assert model("count", image_paths=image) == first
    assert len(calls) == 1
    image.write_bytes(b"changed content")
    model("count", image_paths=image)
    assert len(calls) == 2
    monkeypatch.delenv("CACHE_MODE")
    monkeypatch.setenv("TEST_MODE", "1")
    assert model("count", image_paths=image) == first
    assert len(calls) == 2


@pytest.mark.parametrize("response", [
    {"text": "partial", "finish_reason": "length"},
    {"text": "", "finish_reason": "stop"},
])
def test_incomplete_responses_are_not_cached(model_dir, monkeypatch, tmp_path, response):
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.setattr(LocalQwen, "_infer", lambda self, request: response)
    with pytest.raises(RuntimeError):
        LocalQwen()("inspect")
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_TEXT_VLM", "typo")
    with pytest.raises(ValueError, match="Unknown"):
        create_text_vlm()


def test_detection_prompt_names_coordinate_order_explicitly():
    from simfoundry.utils.prompt_utils import prompt_object_setlist_specific_floorplane_v3

    default = prompt_object_setlist_specific_floorplane_v3("table")
    local = prompt_object_setlist_specific_floorplane_v3("table", bbox_format=LocalQwen.bbox_format)
    assert "[ymin, xmin, ymax, xmax]" in default and '"box_2d"' in default
    assert "[xmin, ymin, xmax, ymax]" in local and '"bbox_xyxy"' in local
    assert "box_2d" not in local
