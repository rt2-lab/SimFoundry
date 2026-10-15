# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent HTTP inference server for a LeRobot-format pi0.5 (pi05) checkpoint.

This process runs inside the isolated `.cache/pi05-serve-venv` venv (NOT the main
`simfoundry` conda env -- see AGENTS.md on why that env must not receive new
pip installs). It loads the pi05 policy + its pre/post-processor pipelines exactly
once at startup and then serves many inference requests for the lifetime of an
eval run, because reloading a 14GB checkpoint (PaliGemma VLM backbone + action
expert) per control step would dominate wall-clock time (see
`simfoundry/policies/pi05_lerobot.py` for why this differs from the LocalQwen
per-call-subprocess pattern).

Wire protocol (plain HTTP + JSON, no native `openpi` package involved -- that
package talks to a different, non-LeRobot checkpoint format and isn't installed
here):

  GET  /health
      -> {"status": "ok", "ready": bool}

  POST /reset
      Clears the policy's internal action-chunk queue (see PI05Policy.reset()).
      -> {"status": "ok"}

  POST /infer
      Body: {
        "images": {
            "base_0_rgb":        {"shape": [H, W, 3], "dtype": "uint8", "data": "<base64 raw bytes>"},
            "left_wrist_0_rgb":  {... same ...},
            "right_wrist_0_rgb": {... same ..., OPTIONAL -- see note below}
        },
        "state": [float, ...]   # zero-padded by this server up to the checkpoint's state dim
        "task": "<language instruction>",
        "action_dim": 8          # OPTIONAL, default 8: how many leading action dims to
                                  # return (this checkpoint pads actions to 32 regardless
                                  # of the real embodiment's DOF count)
      }
      -> {"action": [[float, ...], ...]}   # shape (chunk_size, action_dim)

  Missing camera note: this checkpoint expects 3 image views
  (base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb) but the OmniGibson sim this is
  evaluated against only has 2 cameras (one external, one wrist -- see
  OpenPIClient._extract_observation in simfoundry/policies/openpi.py). Rather than
  duplicating the single wrist image into both wrist slots (which would be an
  out-of-distribution "two identical wrist cameras" input), this server follows
  PI05Policy._preprocess_images's OWN built-in handling for a genuinely absent
  camera: any image key present in `policy.config.image_features` but absent from
  the batch dict is padded internally with an all -1 tensor AND a zero attention
  mask, which is the model's dedicated "no camera here" representation. So the
  client simply omits "right_wrist_0_rgb" instead of sending a stand-in image, and
  this server must NOT invent a placeholder for it.
"""

import argparse
import base64
import logging
import os
import sys
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="[pi05_serve] %(asctime)s %(message)s")
log = logging.getLogger("pi05_serve")

app = Flask(__name__)

# Populated by _load_policy() at startup; guarded by _lock since Flask's dev
# server can be multi-threaded and a 14GB model must only ever be loaded once
# and only ever run inference for one request at a time (no concurrent calls
# make sense for a single rollout anyway).
_state = {"policy": None, "preprocessor": None, "postprocessor": None, "device": None, "ready": False}
_lock = threading.Lock()

# Image keys come from the loaded checkpoint's config: pi05_base and DROID-based finetunes expect
# base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb, an xArm6 finetune such as
# ys903/pi05_xarm6_pick_place_50ep expects front/wrist. State is zero-padded to the width of the
# normalizer's state stats (the data the checkpoint was trained on), not the config's declared
# width: a DROID-based finetune declares 8 but was trained on 7-dim xArm state, and the model
# pads to max_state_dim itself.
IMAGE_PREFIX = "observation.images."


def _state_width(policy, preprocessor):
    for step in preprocessor.steps:
        stats = (getattr(step, "stats", None) or {}).get("observation.state") or {}
        for name in ("min", "mean", "q01"):
            if name in stats:
                return int(np.asarray(stats[name]).reshape(-1).shape[0])
    return policy.config.input_features["observation.state"].shape[0]


def _load_policy(checkpoint_path: str, device: str):
    """Load the PI05 policy + processors exactly once. See module docstring."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    log.info(f"Loading pi05 config from {checkpoint_path} ...")
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.device = device

    log.info(f"Loading pi05 policy weights (this reads ~14GB from disk once) ...")
    t0 = time.time()
    policy = PI05Policy.from_pretrained(checkpoint_path, config=cfg)
    policy.eval()
    log.info(f"Policy loaded in {time.time() - t0:.1f}s on device={device}")

    log.info("Building pre/post-processor pipelines (downloads the gated "
             "google/paligemma-3b-pt-224 tokenizer on first run; needs HF_TOKEN) ...")
    # The checkpoint's saved policy_preprocessor.json hardcodes its last step
    # (device_processor) to "cpu" -- true for whoever trained/exported it, but
    # not for us when --device cuda. Without this override the preprocessor
    # pipeline moves the tokenized/image batch to CPU while the model (per
    # cfg.device above) lives on the GPU, raising "Expected all tensors to be
    # on the same device" inside the language-embedding lookup (confirmed by
    # running a real eval with --device cuda). The postprocessor's own
    # device_processor is deliberately left at its saved "cpu" -- that one
    # exists to hand the action chunk back as plain CPU/numpy-friendly data,
    # which is what this server needs anyway to JSON-serialize it.
    overrides = {"device_processor": {"device": device}}
    # Checkpoints exported by lerobot >= 0.6.2 ship their tokenizer inside the checkpoint and
    # name it by the relative path "tokenizer"; 0.6.1 resolves that against the cwd instead.
    local_tokenizer = os.path.join(checkpoint_path, "tokenizer")
    if os.path.isdir(local_tokenizer):
        overrides["tokenizer_processor"] = {"tokenizer_name": os.path.abspath(local_tokenizer)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint_path,
        preprocessor_overrides=overrides,
    )
    log.info("Processors built OK")

    with _lock:
        _state["policy"] = policy
        _state["preprocessor"] = preprocessor
        _state["postprocessor"] = postprocessor
        _state["device"] = device
        _state["ready"] = True
    log.info("pi05 server ready to accept /infer requests")


def _decode_image(entry: dict) -> np.ndarray:
    shape = tuple(entry["shape"])
    dtype = np.dtype(entry["dtype"])
    raw = base64.b64decode(entry["data"])
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ready": bool(_state["ready"])})


@app.route("/reset", methods=["POST"])
def reset():
    with _lock:
        policy = _state["policy"]
        if policy is None:
            return jsonify({"status": "error", "message": "policy not loaded yet"}), 503
        policy.reset()
    log.info("policy.reset() called (action-chunk queue cleared for new episode)")
    return jsonify({"status": "ok"})


@app.route("/infer", methods=["POST"])
def infer():
    body = request.get_json(force=True)
    with _lock:
        policy = _state["policy"]
        preprocessor = _state["preprocessor"]
        postprocessor = _state["postprocessor"]
        device = _state["device"]
        if policy is None:
            return jsonify({"status": "error", "message": "policy not loaded yet"}), 503

        images = body.get("images", {})
        state = body.get("state", [])
        task = body.get("task", "")
        action_dim = int(body.get("action_dim", 8))

        batch = {}
        image_keys = [k[len(IMAGE_PREFIX):] for k in policy.config.image_features]
        for key in image_keys:
            if key not in images:
                # Deliberately NOT inserting a placeholder -- see module docstring.
                continue
            img = _decode_image(images[key])  # (H, W, C) uint8
            img_t = torch.from_numpy(img.copy()).to(torch.float32) / 255.0  # [0, 1]
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            batch[f"observation.images.{key}"] = img_t

        state_arr = np.asarray(state, dtype=np.float32)
        state_dim = _state_width(policy, preprocessor)
        if state_arr.shape[0] < state_dim:
            padded = np.zeros(state_dim, dtype=np.float32)
            padded[: state_arr.shape[0]] = state_arr
            state_arr = padded
        batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0)
        batch["task"] = [task]

        processed = preprocessor(batch)
        with torch.no_grad():
            action_chunk = policy.predict_action_chunk(processed)  # (1, chunk_size, 32)
            action_chunk = postprocessor(action_chunk)

        action_chunk = action_chunk[0, :, :action_dim].detach().cpu().numpy().tolist()

    return jsonify({"action": action_chunk})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the LeRobot-format pi05 checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--hf-token", default=None, help="HF token for the gated google/paligemma-3b-pt-224 tokenizer download")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if "HF_TOKEN" not in os.environ:
        log.warning("HF_TOKEN not set -- the gated paligemma-3b-pt-224 tokenizer download will 401 "
                    "unless it is already cached locally.")

    # Load synchronously before serving: the client waits on /health polling,
    # so there is no benefit to backgrounding this, and it avoids racing
    # /infer against an unfinished load.
    _load_policy(args.checkpoint, args.device)

    # threaded=False: only one inference should ever run at a time against a
    # single-episode rollout; avoids two /infer calls racing on the same
    # policy's internal state (e.g. RTC/queue state) if a client ever retries.
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
