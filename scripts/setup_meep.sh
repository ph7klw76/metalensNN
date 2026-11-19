#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAMBA_DOWNLOAD_THREADS=2 MAMBA_EXTRACT_THREADS=1
mkdir -p artifacts/phase6 locks
python3 -m fiber_metalens_nn.cli audit --output artifacts/phase6/setup-hardware.json --require-ready
if [[ -f locks/meep-linux-64-explicit.txt ]]; then
  packages=(--file locks/meep-linux-64-explicit.txt)
else
  packages=(--override-channels -c conda-forge --strict-channel-priority 'python=3.12' 'pymeep=1.35.0=nompi_py312h7a316d3_100' numpy scipy pyyaml psutil)
fi
python3 -m fiber_metalens_nn.cli guard --hardware artifacts/phase6/setup-hardware.json --log artifacts/phase6/setup-memory.jsonl --timeout 1800 -- .tools/bin/micromamba --no-rc create -y -r "$PWD/.cache/mamba-root" -p "$PWD/.venv-meep" "${packages[@]}"
python3 scripts/lock_meep.py
.venv-meep/bin/python -c 'import meep; print("Meep", meep.__version__)'
