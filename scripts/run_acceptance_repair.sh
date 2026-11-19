#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_repair.sh diagnose_acceptance
bash scripts/run_repair.sh diagnose_proxy
bash scripts/run_repair.sh diagnose_profile
bash scripts/run_repair.sh validate_thick_proxy
bash scripts/run_repair.sh train_repair
bash scripts/run_repair.sh check_trained_state_resume --config configs/finite_thickness_repair.json --folder artifacts/acceptance-repair/phase5
PYTHONPATH=src .venv-gpu/bin/python scripts/reuse_unchanged_references.py
status=0
bash scripts/run_repair.sh validate_repair || status=$?
bash scripts/run_repair.sh check_repaired_sampling || status=$?
PYTHONPATH=src .venv-gpu/bin/python scripts/report_repair.py
exit "$status"
