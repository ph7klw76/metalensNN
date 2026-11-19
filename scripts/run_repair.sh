#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
mkdir -p artifacts/acceptance-repair
job="$1"
shift
python3 -m fiber_metalens_nn.cli audit --output artifacts/acceptance-repair/hardware.json --require-ready
.venv-gpu/bin/python -m fiber_metalens_nn.cli guard --hardware artifacts/acceptance-repair/hardware.json --log "artifacts/acceptance-repair/$job-memory.jsonl" --timeout 21600 -- .venv-gpu/bin/python "scripts/$job.py" "$@"
