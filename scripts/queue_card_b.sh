#!/usr/bin/env bash
# The queue of card B (2026-09-13). Start it on a free card:
#   bash scripts/queue_card_b.sh <gpu>            # inside tmux; e.g. bash scripts/queue_card_b.sh 1
#
# Trains, one after another:
#   abl_aitod_2_fine          the stride ablation, full fine level (row 2)
#   dfine_l_visdrone          the L baselines; AI-TOD L last, the heaviest run (36 h, 22 GB),
#   dfine_l_aitod             so the card stays held
#   dfine_l_aitod (seed 3)    a repeat to keep holding it; kill it when the card is needed
# Nothing here depends on the light-versus-heavy decision: the ours_* runs and rows 3 to 5 wait
# for it. A failed run does not stop the queue. NOTIFY_URL defaults to the dfine-saturn topic.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${1:?usage: bash scripts/queue_card_b.sh <gpu>}
export REPORT=1
export NOTIFY_URL=${NOTIFY_URL:-https://ntfy.sh/dfine-saturn}

bash scripts/experiments.sh abl_aitod_2_fine dfine_l_visdrone dfine_l_aitod
SEED=3 bash scripts/experiments.sh dfine_l_aitod
SEED=4 bash scripts/experiments.sh dfine_l_aitod
echo "card B queue done"
