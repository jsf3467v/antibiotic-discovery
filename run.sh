#!/usr/bin/env bash
# Results pipeline for the antibiotic-discovery project, on CPU, across seeds.
# Reward-integrity checks live in verify.sh; run that after this completes.
#
#   bash run.sh --fresh          wipe per-seed outputs, then run seeds 42 43 44
#   bash run.sh --fresh 42       wipe + run a single seed
#   bash run.sh 43 44            resume: no wipe, finished work is skipped
#
# train_rl resumes from its last checkpoint, so a crashed run can be restarted
# without --fresh and it continues from where it stopped.

set -euo pipefail
cd "$(dirname "$0")"

export PROJECT_DEVICE="${PROJECT_DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
# Unbuffered stdout so `| tee run.log` shows progress live instead of block-buffering.
export PYTHONUNBUFFERED=1

FRESH=0
if [ "${1:-}" = "--fresh" ]; then
    FRESH=1
    shift
fi
if [ "$#" -eq 0 ]; then
    SEEDS=(42 43 44)
else
    SEEDS=("$@")
fi
# stat_tests builds rl_episode_props.csv; dynamics consumes it.
STEPS=(train_rl eval_rl baselines eval_baselines stat_tests agreement dynamics)

stamp() { date +%H:%M:%S; }

if [ "$FRESH" -eq 1 ]; then
    echo "[$(stamp)] clearing outputs for seeds ${SEEDS[*]}"
    for s in "${SEEDS[@]}"; do rm -rf "runs/seed$s"; done
    rm -rf results/summary
fi

for s in "${SEEDS[@]}"; do
    for step in "${STEPS[@]}"; do
        echo "[$(stamp)] seed $s : $step"
        RL_SEED="$s" python -m "src.$step"
    done
done

echo "[$(stamp)] cross-seed summary (collect)"
python -m src.collect

echo "[$(stamp)] done: seeds ${SEEDS[*]}"
