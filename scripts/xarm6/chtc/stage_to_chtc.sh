#!/bin/bash
# Upload what train.sub needs to CHTC. Run from the SimFoundry repo root on the lab machine; scp
# asks for your CHTC password + Duo.
#
#   bash scripts/xarm6/chtc/stage_to_chtc.sh <netid> <submit_host> [what]
#     submit_host: your CHTC access point, e.g. ap2002.chtc.wisc.edu
#     what:        all (default) | dataset | base | jobfiles
#
# Staged under $STAGING (default /staging/t/tdu35/simfoundry):
#   <dataset>_dataset.tar  local/<dataset>/ (LeRobot dataset)
#   <base>_base.tar        <base>/ (checkpoint, processor JSONs patched for relative actions)
#                          + hf_home/hub/models--google--paligemma-3b-pt-224 (gated tokenizer cache)
# Job files go to ~/simfoundry_chtc on the submit node.
set -euo pipefail

NETID="${1:?Usage: stage_to_chtc.sh <netid> <submit_host> [all|dataset|base|jobfiles]}"
SUBMIT_HOST="${2:?Usage: stage_to_chtc.sh <netid> <submit_host> [all|dataset|base|jobfiles]}"
WHAT="${3:-all}"
STAGING="${STAGING:-/staging/t/tdu35/simfoundry}"
DATASET_DIR="${DATASET_DIR:-Data/cube2_scene/lerobot/xarm6_cube_in_box_sim}"
BASE_DIR="${BASE_DIR:-checkpoints/pi05_droid_jointpos}"
TOKENIZER_CACHE="${HOME}/.cache/huggingface/hub/models--google--paligemma-3b-pt-224"
TRANSFER_HOST="transfer.chtc.wisc.edu"
CHTC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

DATASET_NAME="$(basename "${DATASET_DIR}")"
BASE_NAME="$(basename "${BASE_DIR}")"
upload=()

if [[ "${WHAT}" == all || "${WHAT}" == dataset ]]; then
    test -f "${DATASET_DIR}/meta/stats.json" || { echo "missing dataset ${DATASET_DIR}"; exit 1; }
    tar -cf "${TMP}/${DATASET_NAME}_dataset.tar" -C "$(dirname "${DATASET_DIR}")" \
        --transform "s,^${DATASET_NAME},local/${DATASET_NAME}," "${DATASET_NAME}"
    upload+=("${TMP}/${DATASET_NAME}_dataset.tar")
fi
if [[ "${WHAT}" == all || "${WHAT}" == base ]]; then
    test -f "${BASE_DIR}/model.safetensors" || { echo "missing checkpoint ${BASE_DIR}"; exit 1; }
    test -d "${TOKENIZER_CACHE}" || { echo "missing tokenizer cache ${TOKENIZER_CACHE}"; exit 1; }
    mkdir -p "${TMP}/hf_home/hub"
    cp -a "${TOKENIZER_CACHE}" "${TMP}/hf_home/hub/"
    tar -cf "${TMP}/${BASE_NAME}_base.tar" -C "$(dirname "${BASE_DIR}")" "${BASE_NAME}" -C "${TMP}" hf_home
    upload+=("${TMP}/${BASE_NAME}_base.tar")
fi
if (( ${#upload[@]} )); then
    du -sh "${upload[@]}"
    ssh "${NETID}@${TRANSFER_HOST}" "mkdir -p ${STAGING}/checkpoints"
    scp "${upload[@]}" "${NETID}@${TRANSFER_HOST}:${STAGING}/"
fi
if [[ "${WHAT}" == all || "${WHAT}" == jobfiles ]]; then
    ssh "${NETID}@${SUBMIT_HOST}" "mkdir -p ~/simfoundry_chtc/logs"
    scp "${CHTC_DIR}/train.sub" "${CHTC_DIR}/run_train_chtc.sh" "${CHTC_DIR}/../finetune_pi05.sh" \
        "${NETID}@${SUBMIT_HOST}:~/simfoundry_chtc/"
fi
echo "Done. On ${SUBMIT_HOST}: cd ~/simfoundry_chtc && condor_submit train.sub dockerhub_user=tdu35"
