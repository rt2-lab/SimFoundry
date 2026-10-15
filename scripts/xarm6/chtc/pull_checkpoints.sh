#!/bin/bash
# Fetch a finished CHTC run's checkpoint bundle into outputs/chtc/<exp_name>/. Run from the
# SimFoundry repo root; scp asks for your CHTC password + Duo.
#
#   bash scripts/xarm6/chtc/pull_checkpoints.sh <netid> <exp_name>
#
# Then serve a step locally, e.g.:
#   .cache/pi05-serve-venv/bin/python simfoundry/policies/pi05_serve/server.py --device cuda \
#       --checkpoint outputs/chtc/<exp_name>/checkpoints/020000/pretrained_model
set -euo pipefail

NETID="${1:?Usage: pull_checkpoints.sh <netid> <exp_name>}"
EXP_NAME="${2:?Usage: pull_checkpoints.sh <netid> <exp_name>}"
STAGING="${STAGING:-/staging/t/tdu35/simfoundry}"
DEST="outputs/chtc"

mkdir -p "${DEST}"
scp "${NETID}@transfer.chtc.wisc.edu:${STAGING}/checkpoints/${EXP_NAME}.tar" "${DEST}/${EXP_NAME}.tar"
tar -xf "${DEST}/${EXP_NAME}.tar" -C "${DEST}"
rm -f "${DEST}/${EXP_NAME}.tar"
ls "${DEST}/${EXP_NAME}/checkpoints"
