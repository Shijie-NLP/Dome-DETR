#!/usr/bin/env bash
# The queue of card A (2026-09-13): the card that is training dfine_s_aitod now.
#   bash scripts/queue_card_a.sh <gpu>            # inside tmux; e.g. bash scripts/queue_card_a.sh 0
#
# Takes the card over the moment dfine_s_aitod ends (checked every 2 seconds), then trains, one
# after another:
#   abl_aitod_2b_fine_light   the stride ablation, light fine level (row 2b)
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

# the baseline's processes carry its config on their command line; [.] keeps this loop from matching itself
while pgrep -f "DFine-S-AITOD[.]yml" >/dev/null; do sleep 2; done

bash scripts/experiments.sh abl_aitod_2b_fine_light dfine_m_visdrone dfine_m_aitod
SEED=1 bash scripts/experiments.sh dfine_l_aitod
SEED=2 bash scripts/experiments.sh dfine_l_aitod
echo "card A queue done"
