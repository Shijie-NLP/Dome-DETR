#!/usr/bin/env bash
# The queue of card A (2026-09-13). Start it on a free card:
#   bash scripts/queue_card_a.sh <gpu>            # inside tmux; e.g. bash scripts/queue_card_a.sh 0
#
# Trains, one after another:
#   dfine_s_aitod             the S AI-TOD baseline (row 1 of the ablation)
#   abl_aitod_2b_fine_light   the stride ablation, light fine level (row 2b)
#   dfine_s_visdrone          the S VisDrone baseline (on the server for the first time)
#   dfine_m_visdrone          the M baselines
#   dfine_m_aitod
#   dfine_l_aitod (seed 1)    the heaviest run last, so the card stays held for 36 h at a time
#   dfine_l_aitod (seed 2)
# Nothing here depends on the light-versus-heavy decision: the ours_* runs and rows 3 to 5 wait
# for it. A failed run does not stop the queue. NOTIFY_URL defaults to the dfine-saturn topic.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${1:?usage: bash scripts/queue_card_a.sh <gpu>}
export REPORT=1
export NOTIFY_URL=${NOTIFY_URL:-https://ntfy.sh/dfine-saturn}

bash scripts/experiments.sh dfine_s_aitod abl_aitod_2b_fine_light dfine_s_visdrone dfine_m_visdrone dfine_m_aitod
SEED=1 bash scripts/experiments.sh dfine_l_aitod
SEED=2 bash scripts/experiments.sh dfine_l_aitod
echo "card A queue done"
