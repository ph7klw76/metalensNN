#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_CACHE=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_PYTHON_DOWNLOADS=never
export UV_CONCURRENT_BUILDS=1
export UV_LINK_MODE=copy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Use the pinned wheel libraries rather than an inherited toolkit override.
unset LD_LIBRARY_PATH
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p locks artifacts/phase1
python3 -m fiber_metalens_nn.cli audit --output hardware.json --require-ready
uv --version > artifacts/phase1/uv-version.txt
if [[ ! -e locks/rtx4060-wsl2-py312-cuda12.txt ]]; then
    uv pip compile --quiet --python-version 3.12 --generate-hashes requirements-rtx4060.in -o locks/rtx4060-wsl2-py312-cuda12.txt
fi
if [[ ! -e .venv-gpu/bin/python ]]; then
    uv venv --python "$(readlink -f "$(command -v python3.12)")" .venv-gpu
fi
if [[ ! -f artifacts/phase1/wheel-downloads.json ]]; then
    echo 'STOP: Fetch the hash-verified wheels first with scripts/download_locked_wheels.py.' >&2
    exit 2
fi
python3 -m fiber_metalens_nn.cli guard --hardware hardware.json --log artifacts/phase1/install-bounded-memory.jsonl --timeout 1800 -- python3 scripts/install_one_package_at_a_time.py
uv pip freeze --python .venv-gpu/bin/python > locks/rtx4060-installed.txt
.venv-gpu/bin/python -m pytest -q | tee artifacts/phase1/pytest.log
for backend in cpu gpu; do
    .venv-gpu/bin/python -m fiber_metalens_nn.cli audit --output hardware.json --require-ready
    .venv-gpu/bin/python -m fiber_metalens_nn.cli guard --hardware hardware.json --log "artifacts/phase1/${backend}-memory.jsonl" --timeout 240 -- .venv-gpu/bin/python -m fiber_metalens_nn.cli smoke --backend "$backend" --output "artifacts/phase1/${backend}-smoke.json"
done
python3 scripts/phase1_report.py
