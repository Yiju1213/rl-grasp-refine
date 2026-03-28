#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/train_formal_from_scratch.sh [EXPERIMENT_PATH] [-- additional train.py args]

Examples:
  bash scripts/train_formal_from_scratch.sh
  bash scripts/train_formal_from_scratch.sh configs/experiment/exp_debug.yaml
EOF
  exit 0
fi

EXPERIMENT_PATH="${1:-configs/experiment/exp_debug.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

cd "${ROOT_DIR}"
python scripts/train.py --experiment "${EXPERIMENT_PATH}" "$@"
