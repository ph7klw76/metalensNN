#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
phase="$1"
mkdir -p "artifacts/phase${phase}"
python3 -m fiber_metalens_nn.cli audit --output "artifacts/phase${phase}/hardware.json" --require-ready
timeout=3600
if (( phase >= 6 )); then timeout=21600; fi
.venv-gpu/bin/python -m fiber_metalens_nn.cli guard --hardware "artifacts/phase${phase}/hardware.json" --log "artifacts/phase${phase}/memory.jsonl" --timeout "$timeout" -- .venv-gpu/bin/python -m fiber_metalens_nn.trial "$phase"
