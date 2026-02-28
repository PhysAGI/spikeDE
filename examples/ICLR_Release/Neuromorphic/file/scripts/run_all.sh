#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_n101_transformer.sh"
bash "$SCRIPT_DIR/run_hardvs_cnn.sh"
