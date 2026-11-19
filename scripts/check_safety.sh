#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -p test_safety_stdlib.py -v
python3 -m fiber_metalens_nn.cli audit --output hardware.json --require-ready
