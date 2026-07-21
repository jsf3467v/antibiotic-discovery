#!/usr/bin/env bash
# Reward-integrity checks for the antibiotic-discovery pipeline. Run after run.sh.
# Reads existing artifacts, scores nothing new with the GNN, finishes in seconds.
#
#   bash verify.sh              check seeds 42 43 44
#   bash verify.sh 42           check a single seed
#
# diagnose_rewards writes the seed-invariant probe file that gate_check reads,
# so it runs once up front; gate_check then runs per seed against each run dir.
# Both print only their verdict and evidence lines; run the modules without
# --quiet for the full section tables (which are also written to CSV regardless).

set -euo pipefail
cd "$(dirname "$0")"

export PROJECT_DEVICE="${PROJECT_DEVICE:-cpu}"
export PYTHONUNBUFFERED=1

if [ "$#" -eq 0 ]; then
    SEEDS=(42 43 44)
else
    SEEDS=("$@")
fi

stamp() { date +%H:%M:%S; }

diag="${SEEDS[0]}"
echo "[$(stamp)] reward diagnostics (diagnose_rewards)"
RL_SEED="$diag" python -m src.diagnose_rewards --quiet

for s in "${SEEDS[@]}"; do
    echo "[$(stamp)] seed $s : gate_check"
    RL_SEED="$s" python -m src.gate_check --quiet
done

echo "[$(stamp)] done: verified seeds ${SEEDS[*]}"
