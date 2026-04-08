#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

configs=(
  "$ROOT_DIR/configs/experiment/exp_debug_stb5x_latefus_wo_onl_cal_128_epi.yaml"
  "$ROOT_DIR/configs/experiment/exp_debug_stb5x_latefus_wo_stb_rwd_128_epi.yaml"
  "$ROOT_DIR/configs/experiment/exp_debug_stb5x_latefus_wo_tac_rwd_128_epi.yaml"
  "$ROOT_DIR/configs/experiment/exp_debug_stb5x_latefus_wo_tac_sem_n_rwd_128_epi.yaml"
)

cd "$ROOT_DIR"

for cfg in "${configs[@]}"; do
  echo "=== Training with: $cfg"
  python "$ROOT_DIR/scripts/train.py" --experiment "$cfg"
done
