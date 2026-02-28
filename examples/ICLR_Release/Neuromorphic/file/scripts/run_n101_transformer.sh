#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$ROOT_DIR/file/logs"
TRAIN_OUT_DIR="$ROOT_DIR"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"
nohup python train_n101_transformer.py --model spikformer --alpha 0.1 --device cuda:5 -b 32 --output-dir "$TRAIN_OUT_DIR" > "$LOG_DIR/train_n101_alpha0.1_b32.log" 2>&1 &
echo "Started: train_n101_transformer.py (log: $LOG_DIR/train_n101_alpha0.1_b32.log)"
