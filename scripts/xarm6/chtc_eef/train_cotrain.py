# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`lerobot-train` with the real episodes of a real+sim dataset oversampled.

The dataset's meta/sources.jsonl (one line per episode, in episode order, written by
convert_real_h5_to_lerobot.py) marks each episode "real" or sim. LeRobot's EpisodeAwareSampler
draws frames from a list of episode [from, to) ranges; listing every real range REAL_REPEAT times
makes each real frame REAL_REPEAT times as likely per epoch. Everything else is stock lerobot-train
(same CLI flags); with REAL_REPEAT=1 this is exactly lerobot-train.

    REAL_REPEAT=4 python train_cotrain.py --dataset.root=<dataset> ...
"""

import json
import os
import sys

import numpy as np
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.scripts import lerobot_train

REPEAT = int(os.environ.get("REAL_REPEAT", "1"))


def _dataset_root():
    for a in sys.argv[1:]:
        if a.startswith("--dataset.root="):
            return a.split("=", 1)[1]
    raise SystemExit("train_cotrain.py needs --dataset.root=...")


def _real_episodes(root):
    with open(os.path.join(root, "meta", "sources.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [i for i, r in enumerate(rows) if r.get("source", "real") == "real"], len(rows)


class RealOversampler(EpisodeAwareSampler):
    def __init__(self, dataset_from_indices, dataset_to_indices, episode_indices_to_use=None, **kw):
        lo, hi = np.asarray(dataset_from_indices), np.asarray(dataset_to_indices)
        real, n_eps = _real_episodes(_dataset_root())
        assert n_eps == len(lo), f"sources.jsonl lists {n_eps} episodes, dataset has {len(lo)}"
        assert episode_indices_to_use is None, "episode subsets are not supported with oversampling"
        extra = np.repeat(np.asarray(real, dtype=np.int64), REPEAT - 1)
        lo, hi = np.concatenate([lo, lo[extra]]), np.concatenate([hi, hi[extra]])
        super().__init__(lo, hi, episode_indices_to_use=None, **kw)
        n_real = int(sum(hi[i] - lo[i] for i in real)) * REPEAT
        print(f"[cotrain] {len(real)} real episodes x{REPEAT}: real frames are {n_real / self._num_frames:.1%} "
              f"of the {self._num_frames} samples per epoch", flush=True)


if REPEAT > 1:
    lerobot_train.EpisodeAwareSampler = RealOversampler

if __name__ == "__main__":
    sys.exit(lerobot_train.main())
