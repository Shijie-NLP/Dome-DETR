#!/usr/bin/env bash
# Distributed training.
#   bash scripts/dist_train.sh <config.yml> [num_gpus] [extra train.py args...]
# e.g.
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh configs/dome/DFine-S-VisDrone.yml 2
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh configs/dome/Dome-S-VisDrone.yml 4 --use-amp
# The log goes to logs/<config name>-<timestamp>.log. MASTER_PORT defaults to 7789.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?usage: bash scripts/dist_train.sh <config.yml> [num_gpus] [extra train.py args...]}
NGPU=${2:-2}
shift $(($# >= 2 ? 2 : 1))

EXP_NAME=$(basename "$CONFIG" .yml)
mkdir -p logs
echo "config: $CONFIG  gpus: $NGPU  extra args: $*"

torchrun --master_port="${MASTER_PORT:-7789}" --nproc_per_node="$NGPU" train.py \
    -c "$CONFIG" --seed=0 "$@" 2>&1 | tee "logs/${EXP_NAME}-$(date +%Y%m%d_%H%M%S).log"
