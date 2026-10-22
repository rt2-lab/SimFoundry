# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for a LeRobot-format pi0.5 (pi05) checkpoint served by
`simfoundry/policies/pi05_serve/server.py`.

This is NOT `simfoundry/policies/openpi.py`: that client talks to the native
`openpi_client.websocket_client_policy.WebsocketClientPolicy`, which expects a
server built on the upstream `openpi` repo's checkpoint format. Our checkpoint
(`/home/tingting/Documents/allocator/models/pi05_base`) is a *LeRobot*-format
pi05 export (`lerobot.policies.pi05.PI05Policy`), which is a different Python
package with a different on-disk layout, so it needs its own server and client.

Why a persistent out-of-process server instead of the LocalQwen per-call
subprocess pattern (see `simfoundry/models/local_qwen.py`): LocalQwen is called
occasionally for text/classification work, where a ~5-10s process-startup +
model-load cost per call is a small fraction of total time. A VLA policy is
called every `execute_horizon` control steps throughout a live rollout -- many
times per second of simulated task time, for up to `n_episodes` episodes -- so
reloading a 14GB checkpoint (PaliGemma VLM backbone + action expert) on every
call would dominate wall-clock time. Instead this client starts ONE server
process (in the isolated `.cache/pi05-serve-venv`, which has `lerobot[pi]` but
is kept out of the shared, version-pinned `simfoundry` conda env -- see
AGENTS.md's warnings about not touching that env) before the eval episode(s)
begin, and re-uses it for the whole run via HTTP.

Design choice -- client starts its own server subprocess: the alternative is
requiring the user to manually start `pi05_serve/server.py` in a separate
terminal before running the eval script. Self-starting is more robust for this
single-process eval harness (no risk of "forgot to start the server" / stale
server from a previous checkpoint / port already bound to someone else's run),
at the cost of a ~1-2 minute startup delay baked into `__init__`. Set
`external_server=True` to instead connect to an already-running server (useful
for iterating on eval code without repaying the load cost each time).
"""

import base64
import os
import socket
import subprocess
import sys
import time

import numpy as np

from simfoundry.utils.processing_utils import resize_with_pad

from .abstract_client import InferenceClient

try:
    import requests
except ImportError as e:
    raise ImportError(
        "simfoundry.policies.pi05_lerobot requires the `requests` package in the "
        "main simfoundry environment (it should already be present as a transitive "
        "dependency; if not, this is a genuinely missing dep, not something to "
        "pip-install ad hoc into the simfoundry conda env without checking first)."
    ) from e

_SERVER_MODULE_PATH = os.path.join(os.path.dirname(__file__), "pi05_serve", "server.py")

# Default DROID/Franka single-arm action layout used throughout this eval
# script's other clients (see gr00t.py, openpi.py, and
# 1_eval_policy_og_scene.py's `action[7]` gripper post-processing): 7 arm
# joint positions + 1 gripper dim. The pi05 checkpoint pads actions/state to
# 32 regardless of embodiment, so this is how many of those 32 dims we treat
# as meaningful on the way back out.
DEFAULT_ACTION_DIM = 8


def _set_pdeathsig_sigterm():
    """subprocess.Popen preexec_fn: ask the kernel to SIGTERM this child if its
    parent dies for any reason (process exit, kill -9, or an unhandled native
    crash), via Linux's PR_SET_PDEATHSIG. Linux-only and best-effort -- a
    failure here must never block spawning the server, just leave the old
    orphan-on-crash risk in place.
    """
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, 15)  # 15 == SIGTERM
    except Exception:
        pass


class PI05LerobotClient(InferenceClient):
    def __init__(
        self,
        checkpoint_path: str,
        venv_python: str = None,
        host: str = "127.0.0.1",
        port: int = 8420,
        device: str = "cuda",
        hf_token: str = None,
        open_loop_horizon: int = 8,
        action_dim: int = DEFAULT_ACTION_DIM,
        external_server: bool = False,
        startup_timeout_s: float = 900.0,
    ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.action_dim = action_dim
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._proc = None

        if external_server:
            print(f"[PI05LerobotClient] Connecting to externally-managed pi05 server at {self.base_url}")
        else:
            venv_python = venv_python or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                ".cache", "pi05-serve-venv", "bin", "python",
            )
            if not os.path.isfile(venv_python):
                raise FileNotFoundError(
                    f"pi05-serve venv python not found at {venv_python}. Create it with "
                    f"`uv venv .cache/pi05-serve-venv --python 3.12` and install "
                    f"`lerobot[transformers-dep]` into it -- see simfoundry/policies/pi05_serve/README "
                    f"(or the task's setup notes) before using policy=pi05_lerobot."
                )
            if self._port_in_use(host, port):
                raise RuntimeError(
                    f"Port {port} is already in use. Either a stale pi05 server is still running "
                    f"(kill it first) or pass external_server=True with the right port to reuse it."
                )

            hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API")
            cmd = [
                venv_python, _SERVER_MODULE_PATH,
                "--checkpoint", checkpoint_path,
                "--host", host,
                "--port", str(port),
                "--device", device,
            ]
            if hf_token:
                cmd += ["--hf-token", hf_token]
            print(f"[PI05LerobotClient] Launching pi05 server: {' '.join(cmd)}")
            env = os.environ.copy()
            # 1_eval_policy_og_scene.py's host process is an Isaac Sim / Omniverse
            # Kit process, which has been observed to crash with a native SIGSEGV
            # (unrelated to this client -- a Kit shutdown-path crash in
            # omni.graph.core/omni.syntheticdata) without ever unwinding Python's
            # normal atexit/__del__ machinery. Without this, the pi05 server
            # subprocess survives as an orphan holding the port and ~16GB of GPU
            # memory, breaking the next launch attempt (observed twice in
            # practice -- see the eval run notes). PR_SET_PDEATHSIG makes the
            # kernel SIGTERM this child automatically the moment its parent dies
            # for any reason, crash included.
            self._proc = subprocess.Popen(cmd, env=env, preexec_fn=_set_pdeathsig_sigterm)
            self._wait_until_ready(startup_timeout_s)
            print(f"[PI05LerobotClient] pi05 server ready at {self.base_url}")

    @staticmethod
    def _port_in_use(host, port) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def _wait_until_ready(self, timeout_s: float):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"pi05 server process exited early with code {self._proc.returncode} "
                    f"before becoming ready -- check its stdout/stderr above for the error."
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2.0)
                if resp.ok and resp.json().get("ready"):
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2.0)
        raise TimeoutError(
            f"pi05 server did not become ready within {timeout_s}s (loading a 14GB checkpoint "
            f"can take a while on first run / cold disk cache -- consider raising startup_timeout_s)."
        )

    def reset(self):
        resp = requests.post(f"{self.base_url}/reset", timeout=30.0)
        resp.raise_for_status()

    def visualize(self, obs: dict):
        base_img = resize_with_pad(obs["exterior_image_1_left"], 224, 224)
        wrist_img = resize_with_pad(obs["wrist_image_left"], 224, 224)
        return np.concatenate([base_img, wrist_img], axis=1)

    def infer(self, obs: dict, instruction: str) -> dict:
        """Map the sim observation to pi05's 3-image + 32-dim-state input format,
        call the server, and return {"action", "viz"} in the same shape
        OpenPIClient.infer returns (so the eval loop in 1_eval_policy_og_scene.py
        needs no changes to consume it).

        IMPORTANT: `obs` here is NOT the raw per-step `obs_dict` OmniGibson
        returns from env.step()/env.reset() (that's keyed by
        obs_dict["policy"]["external_cam"/"wrist_cam"/...], as seen in
        OpenPIClient._extract_observation). By the time `policy.infer(curr_obs,
        prompt)` is called, 1_eval_policy_og_scene.py's run_episode() has
        already flattened that into a dict with keys "exterior_image_1_left",
        "exterior_image_2_left", "wrist_image_left", "joint_position",
        "gripper_position" (see OpenPIClient.infer, which reads exactly these
        keys and is the proven-working reference for this contract -- an
        earlier version of this file wrongly assumed infer() received the raw
        obs_dict and crashed with `KeyError: 'policy'` on the very first
        inference call of a real eval run; this is the fix).
        """
        base_img = resize_with_pad(obs["exterior_image_1_left"], 224, 224)
        wrist_img = resize_with_pad(obs["wrist_image_left"], 224, 224)
        # obs["exterior_image_2_left"] (a second external/base camera) exists
        # in this flattened dict but is deliberately NOT sent anywhere below:
        # pi05 has no second-external-view slot, only base_0_rgb (one base
        # view) + two *wrist* views, so there's nowhere semantically correct
        # to put it. See the right_wrist_0_rgb comment below for why we also
        # don't force it into a wrist slot.

        images = {
            "base_0_rgb": self._encode_image(base_img),
            "left_wrist_0_rgb": self._encode_image(wrist_img),
            # right_wrist_0_rgb deliberately omitted: the sim provides only one
            # true wrist camera. See pi05_serve/server.py's module docstring for
            # why we don't duplicate wrist_img (or stuff in the second external
            # view) here instead -- PI05Policy itself treats an absent image key
            # as "no camera" (all -1 tensor + zero attention mask), which is the
            # model's intended representation for a genuinely missing view,
            # rather than feeding it a duplicate/out-of-distribution image.
        }

        state = np.concatenate([
            np.asarray(obs["joint_position"], dtype=np.float32).flatten(),
            np.asarray(obs["gripper_position"], dtype=np.float32).flatten(),
        ])

        payload = {
            "images": images,
            "state": state.tolist(),
            "task": instruction,
            "action_dim": self.action_dim,
        }
        resp = requests.post(f"{self.base_url}/infer", json=payload, timeout=60.0)
        resp.raise_for_status()
        action_chunk = np.asarray(resp.json()["action"], dtype=np.float32)  # (chunk_size, action_dim)

        vis_images = np.concatenate([base_img, wrist_img], axis=1)
        return {"action": action_chunk, "viz": vis_images}

    @staticmethod
    def _encode_image(img: np.ndarray) -> dict:
        img = np.ascontiguousarray(img.astype(np.uint8))
        return {
            "shape": list(img.shape),
            "dtype": str(img.dtype),
            "data": base64.b64encode(img.tobytes()).decode("ascii"),
        }

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            print("[PI05LerobotClient] Terminating pi05 server subprocess")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PI05XArm6Client(PI05LerobotClient):
    """pi05 client for xArm6 finetunes in the UFACTORY LeRobot layout (e.g.
    ys903/pi05_xarm6_pick_place_50ep): cameras "front" + "wrist" (square frames), state and
    action both [J1..J6 (rad), gripper (0 = open, 1 = closed)], actions absolute.

    `obs` holds "front" and "wrist" (H, W, 3) uint8 images and "state" (7,); @camera_keys picks
    the names they are sent under. Image slots the checkpoint has beyond these two (empty_camera_0,
    right_wrist_0_rgb) are left out so the policy masks them, as it did in training. A checkpoint
    trained with relative actions converts them back to absolute in its postprocessor.
    """

    # Camera names a checkpoint expects for the sim's front / wrist views.
    CAMERA_KEYS = {
        "ufactory": {"front": "front", "wrist": "wrist"},
        # DROID-based finetunes (scripts/xarm6/finetune_pi05.sh renames onto these).
        "droid": {"front": "base_0_rgb", "wrist": "left_wrist_0_rgb"},
    }

    def __init__(self, *args, image_size: int = 224, camera_keys: str = "ufactory", droid_layout: bool = False,
                 **kwargs):
        # droid_layout: the checkpoint was trained on 8-dim [J1..J6, 0, gripper] state/action
        # (convert_demos_to_lerobot.py --droid-layout); obs["state"] and the returned actions stay
        # 7-dim [J1..J6, gripper] either way.
        kwargs.setdefault("action_dim", 8 if droid_layout else 7)
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.camera_keys = self.CAMERA_KEYS[camera_keys]
        self.droid_layout = droid_layout

    def _state(self, state):
        state = np.asarray(state, dtype=np.float32).flatten()
        if self.droid_layout:
            state = np.concatenate([state[:6], [0.0], state[6:7]]).astype(np.float32)
        return state

    def visualize(self, obs: dict):
        return np.concatenate([resize_with_pad(obs[k], self.image_size, self.image_size)
                               for k in ("front", "wrist")], axis=1)

    def infer(self, obs: dict, instruction: str) -> dict:
        images = {k: resize_with_pad(obs[k], self.image_size, self.image_size) for k in ("front", "wrist")}
        payload = {
            "images": {self.camera_keys[k]: self._encode_image(v) for k, v in images.items()},
            "state": self._state(obs["state"]).tolist(),
            "task": instruction,
            "action_dim": self.action_dim,
        }
        resp = requests.post(f"{self.base_url}/infer", json=payload, timeout=60.0)
        resp.raise_for_status()
        action_chunk = np.asarray(resp.json()["action"], dtype=np.float32)
        if self.droid_layout:  # (chunk, 8) -> (chunk, 7): drop the J7 pad
            action_chunk = np.concatenate([action_chunk[:, :6], action_chunk[:, 7:8]], axis=1)
        return {"action": action_chunk, "viz": np.concatenate(list(images.values()), axis=1)}
