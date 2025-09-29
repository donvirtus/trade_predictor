#!/usr/bin/env bash
# Wrapper to ensure the 'projects' conda environment is active before running a Python script.
# Usage: ./run_with_env.sh scripts/predict_enhanced.py --timeframe 15m --model auto --single
set -euo pipefail

# Explicit python path for determinism (faster than full activation)
PYTHON_BIN="/home/donvirtus/miniconda3/envs/projects/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$@"
