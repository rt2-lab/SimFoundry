# SPDX-License-Identifier: Apache-2.0

"""Select a text-output vision-language provider independently of image editing."""

import os


def create_text_vlm(**kwargs):
    """Keep Gemini as the default; opt into local Qwen for reasoning calls only."""
    backend = os.environ.get("SIMFOUNDRY_TEXT_VLM", "gemini").lower()
    if backend == "gemini":
        from simfoundry.models.vlm import Gemini
        return Gemini(**kwargs)
    if backend == "qwen":
        from simfoundry.models.local_qwen import LocalQwen
        return LocalQwen()
    raise ValueError(f"Unknown SIMFOUNDRY_TEXT_VLM: {backend!r}")
