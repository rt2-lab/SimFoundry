#!/bin/bash
# Stage a LeRobot dataset to CHTC (if not already there) and submit one pi0.5 finetune, from the lab
# machine. Reuses the ssh ControlMaster that stage_to_chtc.sh opens (one password + Duo per host);
# W&B credentials come from WANDB_API_KEY / WANDB_ENTITY in ~/.bashrc on the submit node (train.sub
# uses getenv).
#
#   bash scripts/xarm6/chtc/submit_run.sh <dataset_dir> <base_name> <exp_name> [steps] [batch] [save_freq]
#
#   # real data only (50 autoalloc episodes), pi05_base:
#   bash scripts/xarm6/chtc/submit_run.sh Data/cube2_scene/lerobot/xarm6_real pi05_base xarm6_real_base 8000 32 2000
#
# The base checkpoint must already be staged (stage_to_chtc.sh ... base), and job files must be
# current on the submit node (stage_to_chtc.sh ... jobfiles). exp_name is the checkpoint tar name in
# <staging>/checkpoints/ and should be unique (train.sub's default exp_name uses $(Cluster), which
# condor does not expand in transfer_output_remaps when passed through -append).
set -euo pipefail

DATASET_DIR="${1:?dataset_dir}"
BASE_NAME="${2:?base_name}"
EXP_NAME="${3:?exp_name}"
STEPS="${4:-8000}"
BATCH="${5:-32}"
SAVE_FREQ="${6:-2000}"
NETID="${NETID:-tdu35}"
SUBMIT_HOST="${SUBMIT_HOST:-ap2002.chtc.wisc.edu}"
STAGING="${STAGING:-/staging/t/tdu35/simfoundry}"
WANDB_PROJECT="${WANDB_PROJECT:-pi05_xarm6_real}"
MIN_GPU_MEM="${MIN_GPU_MEM:-80000M}"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${HOME}/.ssh/cm-%r@%h:%p" -o ControlPersist=30m)
DATASET_NAME="$(basename "${DATASET_DIR}")"
CHTC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

test -f "${DATASET_DIR}/meta/stats.json" || { echo "missing dataset ${DATASET_DIR}"; exit 1; }
remote() { ssh "${SSH_OPTS[@]}" "${NETID}@${SUBMIT_HOST}" "$@"; }

remote "test -f ${STAGING}/${BASE_NAME}_base.tar" || { echo "base ${BASE_NAME} not staged: run stage_to_chtc.sh ... base"; exit 1; }
if remote "test -f ${STAGING}/${DATASET_NAME}_dataset.tar"; then
    echo "dataset ${DATASET_NAME} already staged"
else
    DATASET_DIR="${DATASET_DIR}" bash "${CHTC_DIR}/stage_to_chtc.sh" "${NETID}" "${SUBMIT_HOST}" dataset
fi
remote "test ! -e ${STAGING}/checkpoints/${EXP_NAME}.tar" || { echo "exp_name ${EXP_NAME} already has a checkpoint tar"; exit 1; }

remote "source ~/.bashrc >/dev/null 2>&1; test -n \"\${WANDB_API_KEY:-}\" || { echo 'WANDB_API_KEY not set on submit node'; exit 1; }; \
  cd ~/simfoundry_chtc && condor_submit train.sub \
  -append 'dataset_name=${DATASET_NAME}' -append 'base_name=${BASE_NAME}' -append 'exp_name=${EXP_NAME}' \
  -append 'batch_size=${BATCH}' -append 'steps=${STEPS}' -append 'save_freq=${SAVE_FREQ}' \
  -append 'min_gpu_mem=${MIN_GPU_MEM}' -append 'wandb_project=${WANDB_PROJECT}'"
echo "Follow: ssh ${NETID}@${SUBMIT_HOST} 'tail -c 2000 ~/simfoundry_chtc/logs/train_<cluster>.err'"
