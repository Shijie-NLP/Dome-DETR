#!/usr/bin/env bash
# The training runs of the paper, by name, one after another in the order given.
#
#   bash scripts/experiments.sh --list                      # the names and their configs
#   bash scripts/experiments.sh dfine_s_visdrone ours_s_aitod
#   bash scripts/experiments.sh baselines                   # every D-FINE baseline (S, M, L x VisDrone, AI-TOD)
#   bash scripts/experiments.sh ours                        # our method (S, M, L x VisDrone, AI-TOD)
#   bash scripts/experiments.sh all                         # baselines, then ours
#   bash scripts/experiments.sh --dry-run all               # print the commands only
#
# Environment:
#   GPUS=2      torchrun on that many GPUs (default 1: plain python; CUDA_VISIBLE_DEVICES picks the card)
#   SEED=0      the seed of every run
#   EXTRA=...   more train.py arguments for every run, e.g. EXTRA="-u train_dataloader.num_workers=4"
#   REPORT=1    after each run, write its RESULTS.md and curves.png with tools/analysis/run_report.py
#
# Each run writes into its config's output_dir/<date>_<time> (train.py does that) and its console
# into logs/<name>-<date>_<time>.log. A failing run stops the sequence.
set -euo pipefail
cd "$(dirname "$0")/.."

# ------------------------------------------------------------------ the registry
declare -A CONFIGS=(
    [dfine_s_visdrone]=configs/dome/DFine-S-VisDrone.yml
    [dfine_m_visdrone]=configs/dome/DFine-M-VisDrone.yml
    [dfine_l_visdrone]=configs/dome/DFine-L-VisDrone.yml
    [dfine_s_aitod]=configs/dome/DFine-S-AITOD.yml
    [dfine_m_aitod]=configs/dome/DFine-M-AITOD.yml
    [dfine_l_aitod]=configs/dome/DFine-L-AITOD.yml
    [ours_s_visdrone]=configs/dome/DFine-S-VisDrone-Ours.yml
    [ours_m_visdrone]=configs/dome/DFine-M-VisDrone-Ours.yml
    [ours_l_visdrone]=configs/dome/DFine-L-VisDrone-Ours.yml
    [ours_s_aitod]=configs/dome/DFine-S-AITOD-Ours.yml
    [ours_m_aitod]=configs/dome/DFine-M-AITOD-Ours.yml
    [ours_l_aitod]=configs/dome/DFine-L-AITOD-Ours.yml
)
BASELINES=(dfine_s_visdrone dfine_m_visdrone dfine_l_visdrone dfine_s_aitod dfine_m_aitod dfine_l_aitod)
OURS=(ours_s_visdrone ours_m_visdrone ours_l_visdrone ours_s_aitod ours_m_aitod ours_l_aitod)

GPUS=${GPUS:-1}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}
REPORT=${REPORT:-0}
DRY_RUN=0

# ------------------------------------------------------------------ functions
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

list_experiments() {
    for name in "${BASELINES[@]}" "${OURS[@]}"; do printf '  %-18s %s\n' "$name" "${CONFIGS[$name]}"; done
}

# the output_dir of a config, resolved through its includes, so the run's directory can be found afterwards
output_dir_of() {
    python - "$1" <<'PY'
import sys
from src.core.yaml_utils import load_config
print(load_config(sys.argv[1])["output_dir"])
PY
}

# the run directory train.py created: the newest under the config's output_dir
latest_run_of() {
    local out
    out=$(output_dir_of "$1")
    ls -1dt "$out"/*/ 2>/dev/null | head -1 | sed 's#/$##'
}

# train one config: python or torchrun, the console tee'd into logs/
train() {
    local name=$1 config=$2 stamp
    stamp=$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p logs
    local -a cmd
    if [ "$GPUS" -gt 1 ]; then
        cmd=(torchrun --master_port="${MASTER_PORT:-7789}" --nproc_per_node="$GPUS" train.py)
    else
        cmd=(python train.py)
    fi
    # shellcheck disable=SC2206  # EXTRA is meant to split into arguments
    cmd+=(-c "$config" --use-amp --seed "$SEED" $EXTRA)
    echo "== $name: ${cmd[*]}"
    if [ "$DRY_RUN" = 1 ]; then return 0; fi
    "${cmd[@]}" 2>&1 | tee "logs/${name}-${stamp}.log"
}

# summarize the newest run of a config into RESULTS.md and curves.png
report() {
    local run
    run=$(latest_run_of "$1")
    if [ -n "$run" ]; then
        echo "== report: $run"
        python tools/analysis/run_report.py "$run"
    fi
}

run_experiment() {
    local name=$1
    local config=${CONFIGS[$name]:-}
    if [ -z "$config" ]; then
        echo "unknown experiment '$name'; the names are:" >&2
        list_experiments >&2
        exit 2
    fi
    train "$name" "$config"
    if [ "$REPORT" = 1 ] && [ "$DRY_RUN" = 0 ]; then report "$config"; fi
}

# a name, or a group of names, to the names it stands for
expand() {
    case "$1" in
        all) echo "${BASELINES[@]}" "${OURS[@]}" ;;
        baselines) echo "${BASELINES[@]}" ;;
        ours) echo "${OURS[@]}" ;;
        *) echo "$1" ;;
    esac
}

# ------------------------------------------------------------------ main
if [ $# -eq 0 ]; then
    usage
    exit 1
fi
names=()
for arg in "$@"; do
    case "$arg" in
        -h | --help)
            usage
            exit 0
            ;;
        --list)
            list_experiments
            exit 0
            ;;
        --dry-run) DRY_RUN=1 ;;
        *)
            read -r -a more <<<"$(expand "$arg")"
            names+=("${more[@]}")
            ;;
    esac
done
if [ ${#names[@]} -eq 0 ]; then
    usage
    exit 1
fi

echo "runs, in order: ${names[*]}  (gpus $GPUS, seed $SEED${EXTRA:+, extra: $EXTRA})"
for name in "${names[@]}"; do run_experiment "$name"; done
echo "done: ${names[*]}"
