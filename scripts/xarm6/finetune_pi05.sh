#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Finetune pi0.5 on the scripted xArm6 sim demos (LeRobot dataset from
# scripts/xarm6/convert_demos_to_lerobot.py). Run inside tmux; uses the LeRobot venv.
#
#   bash scripts/xarm6/finetune_pi05.sh <dataset_root> <output_dir> [base_checkpoint]
#
# Default base: DAVIAN-Robotics/pi05_droid_jointpos (OpenPI's pi05_droid_jointpos, pi0.5 finetuned
# on DROID with joint-position actions). It predicts joint *deltas* relative to the current state
# (gripper absolute), so training uses policy.use_relative_actions with the gripper excluded; the
# dataset's action stats must be recomputed in relative space with the same chunk size
# (convert_demos_to_lerobot.py --relative-action-chunk 15). Its cameras are DROID's, so the
# dataset's front / wrist views are renamed onto base_0_rgb / left_wrist_0_rgb; right_wrist_0_rgb
# stays empty, and the 7-dim state is zero-padded to its max_state_dim of 8.
#
# Environment overrides: LEROBOT_TRAIN (trainer executable, default: the local LeRobot venv's),
# BATCH_SIZE / STEPS / SAVE_FREQ, and TRAIN_EXTRA_ARGS (extra lerobot-train flags, e.g.
# "--policy.train_expert_only=false" for a full finetune on an 80 GB GPU).
#
# Logging: set WANDB_API_KEY (and optionally WANDB_ENTITY / WANDB_PROJECT) to log to Weights &
# Biases; without a key, wandb is off.
#
# Memory: the vision encoder is frozen and only the action expert is trained (as in ys903's
# xArm6 finetune), with gradient checkpointing, which keeps a 32 GB GPU within budget.
set -euo pipefail

DATASET_ROOT=${1:?dataset root}
OUTPUT_DIR=${2:?output dir}
BASE=${3:-DAVIAN-Robotics/pi05_droid_jointpos}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC2086  # TRAIN_EXTRA_ARGS is a list of flags
"${LEROBOT_TRAIN:-${REPO_DIR}/.cache/pi05-serve-venv/bin/lerobot-train}" \
  --dataset.repo_id="local/$(basename "${DATASET_ROOT}")" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.path="${BASE}" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.action_feature_names='["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "gripper.pos"]' \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --batch_size="${BATCH_SIZE:-8}" \
  --steps="${STEPS:-20000}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="pi05_xarm6_cube_in_box_sim" \
  --wandb.enable="$([ -n "${WANDB_API_KEY:-}" ] && echo true || echo false)" \
  --wandb.project="${WANDB_PROJECT:-pi05_xarm6_sim}" \
  ${WANDB_ENTITY:+--wandb.entity="${WANDB_ENTITY}"} \
  --wandb.disable_artifact=true \
  ${TRAIN_EXTRA_ARGS:-}
