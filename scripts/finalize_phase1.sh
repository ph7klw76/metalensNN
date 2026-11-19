#!/usr/bin/env bash
# Collect evidence after a gate stop. Does not install or import JAX.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -p test_safety_stdlib.py -v > artifacts/phase1/safety-stdlib.log 2>&1
python3 -m compileall -q src scripts tests
python3 -m fiber_metalens_nn.cli audit --output hardware.json
python3 scripts/phase1_report.py
