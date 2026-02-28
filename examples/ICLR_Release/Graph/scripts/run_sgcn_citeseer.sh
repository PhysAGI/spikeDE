#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"
python run.py --backbone SGCN \
    --dataset Citeseer \
    --integrator fdeint \
    --tau 2.0 \
    --alpha 0.3 \
    --method pred \
    --tau_learnable