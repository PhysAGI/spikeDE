#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"
python run.py --backbone DRSGNN \
    --dataset Cora \
    --integrator fdeint \
    --tau 5.0 \
    --alpha 0.9 \
    --method gl \
    --tau_learnable