#!/usr/bin/env bash
# Distributed evaluation of a checkpoint.
#   bash scripts/dist_test.sh <config.yml> <checkpoint.pth> [num_gpus] [extra train.py args...]
# e.g.
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_test.sh configs/dome/Dome-S-VisDrone.yml weight/dome-s-visdrone_converted.pth 2
# Set SAVE_TEST_VISUALIZE_RESULT=True to dump ground truth / prediction pairs under visualize_all/.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?usage: bash scripts/dist_test.sh <config.yml> <checkpoint.pth> [num_gpus] [extra train.py args...]}
CHECKPOINT=${2:?usage: bash scripts/dist_test.sh <config.yml> <checkpoint.pth> [num_gpus] [extra train.py args...]}
NGPU=${3:-2}
shift $(($# >= 3 ? 3 : 2))

echo "config: $CONFIG  checkpoint: $CHECKPOINT  gpus: $NGPU  extra args: $*"

torchrun --master_port="${MASTER_PORT:-7778}" --nproc_per_node="$NGPU" train.py \
    -c "$CONFIG" --test-only -r "$CHECKPOINT" "$@"
