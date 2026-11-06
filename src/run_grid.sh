#!/usr/bin/env bash
# One stage = train -> eval -> plots for the given seeds (default: seed 0 only, to check the 10k-step plateau before spending on more seeds).
# Needs the `rdp` conda env active. Logs and stage markers go to $RDP_ROOT/logs. Resumable: grid.py skips finished runs.
set -euo pipefail
cd "$(dirname "$0")"
R="${RDP_ROOT:-/Data/shash/red_dec}"; L="$R/logs"; mkdir -p "$L"
T="${TAG:-}"; export OUT="${OUT:-$R/runs$T}"  # TAG=_30k -> runs_30k/, figures_30k/, results_30k/ ; STEPS=30000 overrides train.steps
stage() {  # name, seeds
  SEEDS="$2" python grid.py train ${STEPS:+train.steps=$STEPS} 2>&1 | tee -a "$L/grid_$1_train.log" | grep -E "timing|Error|Traceback" || true
  SEEDS="$2" python grid.py eval  > "$L/grid_$1_eval.log" 2>&1
  python plots.py "$OUT" "$R/figures$T" "$R/results$T" > "$L/plots_$1.log" 2>&1
  touch "$L/stage_$1_done"; date > "$L/stage_$1_done"
}
rm -f "$L"/stage_*_done
stage "${1:-seed0}" "${2:-0}"  # usage: ./run_grid.sh [stage-name] ["seeds"], e.g. ./run_grid.sh seed12 "1 2"
