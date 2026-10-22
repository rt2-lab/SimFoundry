#!/bin/bash
# Upload the task-space dataset and this package to CHTC, then submit one training job.
# Run from the SimFoundry repo root on the lab machine. Reuses the ssh ControlMaster sockets
# (~/.ssh/cm-<netid>@<host>:22); without one, ssh asks for the password + Duo.
#
#   bash scripts/xarm6/chtc_eef/stage_and_submit.sh <exp_name> [dataset_dir]
#   bash scripts/xarm6/chtc_eef/stage_and_submit.sh xarm6_eef_r4
#
# Env: STEPS (16000) BATCH (32) SAVE_FREQ (4000) REAL_REPEAT (4) BASE_NAME (pi05_base, must
# already be staged by ../chtc/stage_to_chtc.sh ... base) WANDB_PROJECT (pi05_xarm6_eef)
# MIN_GPU_MEM (80000M; training uses ~22 GB, 40000M also admits L40S / A100-40GB)
# DRY_RUN=1 stages but does not submit. SUB_FILE / JOB_DIR / WANDB_PROJECT let another package
# (../chtc_eef_sim, ../chtc_eef_real) reuse this script and the shared trainer files with its own
# submit file, access-point directory and W&B project.
set -euo pipefail

EXP_NAME="${1:?Usage: stage_and_submit.sh <exp_name> [dataset_dir]}"
DATASET_DIR="${2:-Data/cube2_scene/lerobot/xarm6_real_sim_eef}"
NETID="${NETID:-tdu35}"
SUBMIT_HOST="${SUBMIT_HOST:-ap2002.chtc.wisc.edu}"
TRANSFER_HOST="transfer.chtc.wisc.edu"
STAGING="${STAGING:-/staging/t/tdu35/simfoundry}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB_FILE="${SUB_FILE:-${HERE}/train_eef.sub}"
JOB_DIR="${JOB_DIR:-simfoundry_chtc_eef}"
BASE_NAME="${BASE_NAME:-pi05_base}"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${HOME}/.ssh/cm-%r@%h:%p" -o ControlPersist=30m)
DATASET_NAME="$(basename "${DATASET_DIR}")"

submit() { ssh "${SSH_OPTS[@]}" "${NETID}@${SUBMIT_HOST}" "$@"; }
transfer() { ssh "${SSH_OPTS[@]}" "${NETID}@${TRANSFER_HOST}" "$@"; }

test -f "${DATASET_DIR}/meta/stats.json" || { echo "missing dataset ${DATASET_DIR}"; exit 1; }
test -f "${DATASET_DIR}/meta/sources.jsonl" || { echo "missing ${DATASET_DIR}/meta/sources.jsonl (real/sim labels)"; exit 1; }
transfer "test -f ${STAGING}/${BASE_NAME}_base.tar" || { echo "base ${BASE_NAME} not staged"; exit 1; }
transfer "test ! -e ${STAGING}/checkpoints/${EXP_NAME}.tar" || { echo "exp_name ${EXP_NAME} already has a checkpoint tar"; exit 1; }

if transfer "test -f ${STAGING}/${DATASET_NAME}_dataset.tar"; then
    echo "dataset ${DATASET_NAME} already staged"
else
    du -shL "${DATASET_DIR}"
    echo "streaming ${DATASET_NAME}_dataset.tar -> ${TRANSFER_HOST}:${STAGING}/"
    tar -chf - -C "$(dirname "${DATASET_DIR}")" --transform "s,^${DATASET_NAME},local/${DATASET_NAME}," "${DATASET_NAME}" \
        | transfer "cat > ${STAGING}/${DATASET_NAME}_dataset.tar.part && mv ${STAGING}/${DATASET_NAME}_dataset.tar.part ${STAGING}/${DATASET_NAME}_dataset.tar"
fi

submit "mkdir -p ~/${JOB_DIR}/logs"
scp -q "${SSH_OPTS[@]}" "${SUB_FILE}" "${HERE}/run_train_eef.sh" "${HERE}/finetune_eef.sh" "${HERE}/train_cotrain.py" \
    "${NETID}@${SUBMIT_HOST}:~/${JOB_DIR}/"
echo "job files -> ${SUBMIT_HOST}:~/${JOB_DIR}/"
[ "${DRY_RUN:-0}" = 1 ] && { echo "DRY_RUN=1: not submitting"; exit 0; }

submit "source ~/.bashrc >/dev/null 2>&1; test -n \"\${WANDB_API_KEY:-}\" || { echo 'WANDB_API_KEY not set on submit node'; exit 1; }; \
  cd ~/${JOB_DIR} && condor_submit $(basename "${SUB_FILE}") \
  -append 'dataset_name=${DATASET_NAME}' -append 'base_name=${BASE_NAME}' -append 'exp_name=${EXP_NAME}' \
  -append 'batch_size=${BATCH:-32}' -append 'steps=${STEPS:-16000}' -append 'save_freq=${SAVE_FREQ:-4000}' \
  -append 'real_repeat=${REAL_REPEAT:-4}' -append 'wandb_project=${WANDB_PROJECT:-pi05_xarm6_eef}' \
  -append 'min_gpu_mem=${MIN_GPU_MEM:-80000M}'"
echo "Follow: ssh ${NETID}@${SUBMIT_HOST} 'tail -c 2000 ~/${JOB_DIR}/logs/train_<cluster>.err'"
echo "Fetch:  ssh ${NETID}@${TRANSFER_HOST} 'cat ${STAGING}/checkpoints/${EXP_NAME}.tar' | tar -xf - -C outputs/chtc ${EXP_NAME}/checkpoints/<step>/pretrained_model"
