#!/bin/bash
# pi0.5 finetune on a task-space (TCP) xArm6 dataset from make_eef_dataset.py, real episodes
# oversampled by train_cotrain.py. Runs inside the CHTC container (run_train_eef.sh) or locally
# with the LeRobot venv:
#
#   bash finetune_eef.sh <dataset_root> <output_dir> <base_checkpoint_dir>
#
# Env knobs: BATCH_SIZE (32) STEPS (16000) SAVE_FREQ (4000) REAL_REPEAT (4) WANDB_PROJECT
# (pi05_xarm6_eef) TRAIN_EXTRA_ARGS. The LR schedule decays over exactly STEPS (earlier rounds
# decayed over 30k and stopped near the peak LR).
set -euo pipefail

DATASET_ROOT=${1:?dataset root}
OUTPUT_DIR=${2:?output dir}
BASE=${3:?base checkpoint dir}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
STEPS="${STEPS:-16000}"
ACTION_NAMES="$("${PYTHON}" -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))["features"]["action"]["names"]))' "${DATASET_ROOT}/meta/info.json")"

REAL_REPEAT="${REAL_REPEAT:-4}" "${PYTHON}" "${HERE}/train_cotrain.py" \
  --dataset.repo_id="local/$(basename "${DATASET_ROOT}")" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.path="${BASE}" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.action_feature_names="${ACTION_NAMES}" \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="${STEPS}" \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --batch_size="${BATCH_SIZE:-32}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ:-4000}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="pi05_xarm6_eef" \
  --wandb.enable="$([ -n "${WANDB_API_KEY:-}" ] && echo true || echo false)" \
  --wandb.project="${WANDB_PROJECT:-pi05_xarm6_eef}" \
  ${WANDB_ENTITY:+--wandb.entity="${WANDB_ENTITY}"} \
  --wandb.disable_artifact=true \
  ${TRAIN_EXTRA_ARGS:-}
