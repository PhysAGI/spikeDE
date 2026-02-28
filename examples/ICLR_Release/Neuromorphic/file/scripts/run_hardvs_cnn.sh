#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$ROOT_DIR/file/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"
nohup python train_hardvs_cnn.py --alpha 0.8 -device cuda:7 -b 32 > "$LOG_DIR/train_dvs_alpha0.8.log" 2>&1 &
echo "Started: train_hardvs_cnn.py (log: $LOG_DIR/train_dvs_alpha0.8.log)"
