# SPDX-License-Identifier: Apache-2.0

"""Local Qwen text/image-to-text inference with per-call GPU release and caching.

The worker uses its own Python environment so the model's recent Transformers
dependency does not replace the versions required by reconstruction stages.
It never produces images and must not be used as an image-editing backend.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from simfoundry.models.remote_cache import RemoteModelCache, image_digests


class LocalQwen:
    # Qwen localization uses x/y order. The stage-5 reader already accepts this
    # explicit key; never label these coordinates as Gemini's y/x box_2d.
    bbox_format = "xyxy"

    def __init__(self, model_path=None, python_bin=None):
        model_path = model_path or os.environ.get("SIMFOUNDRY_QWEN_MODEL_PATH")
        if not model_path:
            raise ValueError("Set SIMFOUNDRY_QWEN_MODEL_PATH to the local Qwen model directory")
        self.model_path = str(Path(model_path).resolve())
        if not (Path(self.model_path) / "config.json").is_file():
            raise FileNotFoundError(f"No model config under {self.model_path}")
        self.python_bin = python_bin or os.environ.get("SIMFOUNDRY_QWEN_PYTHON", sys.executable)
        self.max_tokens = int(os.environ.get("SIMFOUNDRY_QWEN_MAX_TOKENS", "2048"))
        self.max_side = int(os.environ.get("SIMFOUNDRY_QWEN_MAX_SIDE", "1024"))
        self.quantization = os.environ.get("SIMFOUNDRY_QWEN_QUANTIZATION", "none")
        if self.quantization not in {"none", "nf4"}:
            raise ValueError("SIMFOUNDRY_QWEN_QUANTIZATION must be none or nf4")
        if self.max_tokens < 1 or self.max_side < 32:
            raise ValueError("Invalid Qwen token or image size limit")
        self.timeout = int(os.environ.get("SIMFOUNDRY_QWEN_TIMEOUT_S", "600"))

    def _model_identity(self):
        root = Path(self.model_path)
        # Config/tokenizer contents and weight metadata distinguish local revisions
        # without reading several GB of weights on every request.
        files = sorted(p for p in root.iterdir() if p.is_file())
        identity = []
        for p in files:
            stat = p.stat()
            item = [p.name, stat.st_size, stat.st_mtime_ns]
            if p.suffix in {".json", ".jinja"}:
                item.append(hashlib.sha256(p.read_bytes()).hexdigest())
            identity.append(item)
        return identity

    def __call__(self, prompt, image_paths=None, temperature=0, top_p=0,
                 seed=0, n_retries=3, print_results=False, max_side=None):
        paths = [] if image_paths is None else (
            [image_paths] if isinstance(image_paths, (str, os.PathLike)) else list(image_paths)
        )
        paths = [str(Path(p).resolve()) for p in paths]
        temperature = float(os.environ.get("SIMFOUNDRY_QWEN_TEMPERATURE", temperature))
        top_p = float(os.environ.get("SIMFOUNDRY_QWEN_TOP_P", top_p))
        repetition_penalty = float(os.environ.get("SIMFOUNDRY_QWEN_REPETITION_PENALTY", "1.0"))
        if temperature < 0 or not 0 <= top_p <= 1 or repetition_penalty <= 0:
            raise ValueError("Invalid Qwen sampling settings")
        # Per-call override of self.max_side: stage 5's object-category detection runs
        # in the same GPU as FLUX/SAM3/a depth model (all resident in the parent process),
        # leaving only a sliver of free VRAM for this subprocess. The vision tower's
        # activation memory scales with image side length, so a caller that has observed
        # a CUDA OOM at the default 1024 can request a smaller side here without affecting
        # other call sites (e.g. stage 9's merge-parts step keeps the default).
        effective_max_side = self.max_side if max_side is None else int(max_side)
        if effective_max_side < 32:
            raise ValueError("Invalid Qwen sampling settings")
        request = {
            "prompt": prompt, "image_inputs": image_digests(paths),
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "max_tokens": self.max_tokens, "max_side": effective_max_side,
            "quantization": self.quantization, "model_identity": self._model_identity(),
            "enable_thinking": False, "adapter_version": 2,
            "repetition_penalty": repetition_penalty,
        }
        cache = RemoteModelCache.from_env()
        key = cache.key_for(provider="local_qwen", model=self.model_path, request=request)
        if cache.test_enabled:
            response = cache.load_response("local_qwen", key)
        else:
            response = cache.load_response_if_exists("local_qwen", key) if cache.cache_enabled else None
            if response is None:
                response = self._infer(dict(request, model_path=self.model_path, image_paths=paths))
                self.get_result_text(response)
                if cache.cache_enabled:
                    cache.store_response("local_qwen", self.model_path, key, request, response)
        if print_results:
            print(self.get_result_text(response), flush=True)
        return response

    def _infer(self, request):
        with tempfile.TemporaryDirectory(prefix="simfoundry-qwen-") as tmp:
            input_path = Path(tmp) / "request.json"
            output_path = Path(tmp) / "response.json"
            input_path.write_text(json.dumps(request))
            env = os.environ.copy()
            repo = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            subprocess.run(
                [self.python_bin, "-m", "simfoundry.models.local_qwen",
                 "--request", str(input_path), "--response", str(output_path)],
                env=env, check=True, timeout=self.timeout,
            )
            return json.loads(output_path.read_text())

    @staticmethod
    def get_result_text(result):
        if result.get("finish_reason") != "stop":
            raise RuntimeError("Qwen response was truncated; increase SIMFOUNDRY_QWEN_MAX_TOKENS")
        text = result.get("text", "").strip()
        if not text:
            raise RuntimeError("Qwen returned an empty response")
        return text

    @staticmethod
    def get_result_images(result):
        raise NotImplementedError("Qwen3.5 is text-output only; use an image-editing model")


def infer_local(request):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Local Qwen worker requires a CUDA GPU")
    set_seed(request["seed"])
    model_path = request["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
                  attn_implementation="sdpa")
    if request["quantization"] == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs).eval()
    content = []
    for p in request["image_paths"]:
        with Image.open(p) as source:
            image = source.convert("RGB")
        image.thumbnail((request["max_side"], request["max_side"]), Image.Resampling.LANCZOS)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": request["prompt"]})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    options = {"max_new_tokens": request["max_tokens"], "do_sample": request["temperature"] > 0,
               "repetition_penalty": request.get("repetition_penalty", 1.0)}
    if options["do_sample"]:
        options.update(temperature=request["temperature"], top_p=request["top_p"] or 1.0)
    with torch.inference_mode():
        generated = model.generate(**inputs, **options)
    tokens = generated[0, inputs["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    finished = bool(len(tokens)) and int(tokens[-1]) in eos
    return {
        "text": processor.decode(tokens, skip_special_tokens=True),
        "finish_reason": "stop" if finished else "length",
        "generated_tokens": len(tokens),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    response = infer_local(json.loads(Path(args.request).read_text()))
    Path(args.response).write_text(json.dumps(response, ensure_ascii=False, indent=2))
