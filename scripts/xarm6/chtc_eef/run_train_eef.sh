#!/bin/bash
# CHTC job wrapper for finetune_eef.sh (task-space actions, real+sim co-training); same image as
# ../chtc/train.Dockerfile (simfoundry_lerobot_train:0.6.1).
#
# Usage (from train_eef.sub): run_train_eef.sh <exp_name> <dataset_name> <base_name>
#
# Inputs HTCondor transfers into the scratch dir (made by stage_to_chtc.sh):
#   <dataset_name>_dataset.tar  -> local/<dataset_name>/   (LeRobot dataset, relative-action stats)
#   <base_name>_base.tar        -> <base_name>/            (base pi0.5 checkpoint)
#                                     hf_home/hub/...          (PaliGemma tokenizer cache, gated on the Hub)
#   finetune_eef.sh, train_cotrain.py
# Output: checkpoint_bundle.tar with every saved step's pretrained_model/ (+ train_config), packed
# on normal exit and on eviction / termination.
set -euo pipefail

EXP_NAME="${1:?Usage: run_train_eef.sh <exp_name> <dataset_name> <base_name>}"
DATASET_NAME="${2:?Usage: run_train_eef.sh <exp_name> <dataset_name> <base_name>}"
BASE_NAME="${3:?Usage: run_train_eef.sh <exp_name> <dataset_name> <base_name>}"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
OUT_DIR="${SCRATCH}/outputs/${EXP_NAME}"
BUNDLE="checkpoint_bundle.tar"

echo "SimFoundry CHTC job: exp=${EXP_NAME} dataset=${DATASET_NAME} base=${BASE_NAME} host=$(hostname)"
nvidia-smi || true

for tarball in "${DATASET_NAME}_dataset.tar" "${BASE_NAME}_base.tar"; do
    echo "Extracting ${tarball}"
    tar -xf "${tarball}" -C "${SCRATCH}"
    rm -f "${tarball}"
done

# The container runs as the submitter's uid, which has no passwd entry; torch's getpass.getuser()
# raises KeyError without these.
export USER="${USER:-chtc}" LOGNAME="${LOGNAME:-chtc}"
export HOME="${SCRATCH}" XDG_CACHE_HOME="${SCRATCH}/.cache"
# Everything the trainer needs is local: no Hub downloads (the tokenizer is gated), only wandb.
export HF_HOME="${SCRATCH}/hf_home"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_DIR="${SCRATCH}/.cache/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEROBOT_TRAIN="$(command -v lerobot-train)"
SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE
mkdir -p "${WANDB_DIR}"

package_checkpoints() {
    [ -f "${BUNDLE}" ] && return 0
    if [ ! -d "${OUT_DIR}/checkpoints" ]; then
        echo "No checkpoints under ${OUT_DIR}; nothing to bundle."
        return 0
    fi
    echo "Packaging checkpoints into ${BUNDLE}"
    # Model weights for every saved step; optimizer state (training_state/) is left out.
    (cd "${OUT_DIR}/.." && tar -cf "${SCRATCH}/${BUNDLE}" --exclude='training_state' "${EXP_NAME}")
}
NORMAL_COMPLETION=0
on_exit_or_signal() {
    [ "${NORMAL_COMPLETION}" -eq 1 ] || package_checkpoints
}
trap on_exit_or_signal EXIT TERM INT

# Fail fast on a bad GPU node (one A100 node reported "CUDA device busy or unavailable" at the
# first allocation, after the 7 GB of inputs had already been transferred).
if ! python -c "import torch; x = torch.zeros(1, device='cuda'); print('CUDA ok:', torch.cuda.get_device_name(0))"; then
    echo "CUDA self-test failed on $(hostname)" >&2
    nvidia-smi -q -d COMPUTE | grep -i "compute mode" >&2 || true
    exit 3
fi

echo "Starting training"
bash finetune_eef.sh "${SCRATCH}/local/${DATASET_NAME}" "${OUT_DIR}" "${SCRATCH}/${BASE_NAME}"

package_checkpoints
NORMAL_COMPLETION=1
trap - EXIT TERM INT
