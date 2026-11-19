#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
mkdir -p artifacts/1550nm_collimation
stage="${1:?stage required: phase8, train, validate, package, all}"
shift
stamp="$(date -u +%Y%m%dT%H%M%S%N)"
case "$stage" in
  train|validate|transit|refine) phase=9 ;;
  package) phase=10 ;;
  *) phase=8 ;;
esac
hardware="artifacts/1550nm_collimation/hardware-${stage}-${stamp}.json"
.venv-gpu/bin/python -m fiber_metalens_nn.cli audit --phase "$phase" --output "$hardware" --require-ready
.venv-gpu/bin/python -m fiber_metalens_nn.cli guard --hardware "$hardware" --log "artifacts/1550nm_collimation/${stage}-${stamp}-memory.jsonl" --timeout 21600 -- .venv-gpu/bin/python -m fiber_metalens_nn.collimation_run "$stage" "$@"
