#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GROUP="${1:-main}"
ROOT_DIR="/rl-grasp-refine/outputs/unseen_test_formal"
OUT_DIR="${SCRIPT_DIR}/generated"
DPI="330"
PRINT_DATA_FORMAT="table"

usage() {
  cat <<'EOF'
Usage:
  plot_scripts/plot_group.sh [main|ablation] [options]

Options:
  --root PATH              Formal eval root directory.
  --out-dir PATH           Base output directory. The group subdir is appended by each plot script.
  --dpi N                  PNG export DPI.
  --print-data-format FMT  table or csv.
  -h, --help               Show this help.

Outputs:
  <out-dir>/<group>/*.png
  <out-dir>/<group>/plot_data_<group>.txt
EOF
}

if [[ "${GROUP}" == "-h" || "${GROUP}" == "--help" ]]; then
  usage
  exit 0
fi

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --dpi)
      DPI="$2"
      shift 2
      ;;
    --print-data-format)
      PRINT_DATA_FORMAT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${GROUP}" in
  main|ablation)
    ;;
  *)
    echo "Unsupported group: ${GROUP}. Expected main or ablation." >&2
    exit 2
    ;;
esac

case "${PRINT_DATA_FORMAT}" in
  table|csv)
    ;;
  *)
    echo "Unsupported --print-data-format: ${PRINT_DATA_FORMAT}. Expected table or csv." >&2
    exit 2
    ;;
esac

GROUP_OUT_DIR="${OUT_DIR%/}/${GROUP}"
mkdir -p "${GROUP_OUT_DIR}"
DATA_LOG="${GROUP_OUT_DIR}/plot_data_${GROUP}.txt"

SCRIPTS=(
  fig01_main_overall_performance.py
  fig02_main_risk_return.py
  fig03_risk_return_scatter.py
  fig04_mechanism_triplet.py
  fig06_object_stability_boxplot.py
  fig07_object_stability_bar.py
  fig08_per_object_rank_curve.py
  fig09_per_run_overlay.py
)

{
  echo "# Plot Data Summary"
  echo "group=${GROUP}"
  echo "root=${ROOT_DIR}"
  echo "out_dir=${OUT_DIR}"
  echo "dpi=${DPI}"
  echo "print_data_format=${PRINT_DATA_FORMAT}"
  echo "generated_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo
} > "${DATA_LOG}"

for script in "${SCRIPTS[@]}"; do
  {
    echo "## ${script}"
    python "${SCRIPT_DIR}/${script}" \
      --root "${ROOT_DIR}" \
      --group "${GROUP}" \
      --out-dir "${OUT_DIR}" \
      --dpi "${DPI}" \
      --print-data-format "${PRINT_DATA_FORMAT}"
    echo
  } 2>&1 | tee -a "${DATA_LOG}"
done

echo "Saved plot data log: ${DATA_LOG}"
