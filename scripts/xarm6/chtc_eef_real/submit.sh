#!/bin/bash
# Stage the real-only task-space dataset and submit its pi0.5 finetune on CHTC.
# Run from the SimFoundry repo root. Same trainer as ../chtc_eef (run_train_eef.sh,
# finetune_eef.sh, train_cotrain.py are copied from there), with this package's submit file,
# access-point directory ~/simfoundry_chtc_eef_real and W&B project pi05_xarm6_eef_real.
#
#   bash scripts/xarm6/chtc_eef_real/submit.sh [exp_name]      # default xarm6_eef_real_8k
#
# Env overrides as in ../chtc_eef/stage_and_submit.sh (STEPS, BATCH, SAVE_FREQ, DRY_RUN=1, ...).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB_FILE="${HERE}/train_eef_real.sub" JOB_DIR=simfoundry_chtc_eef_real \
WANDB_PROJECT="${WANDB_PROJECT:-pi05_xarm6_eef_real}" STEPS="${STEPS:-8000}" SAVE_FREQ="${SAVE_FREQ:-2000}" REAL_REPEAT=1 \
    bash "${HERE}/../chtc_eef/stage_and_submit.sh" "${1:-xarm6_eef_real_8k}" Data/cube2_scene/lerobot/xarm6_real_eef
