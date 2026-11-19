# Research workflow

The scripts cover the initial focusing trial, the 1550 nm fiber collimation study, the height-conditioned neural inverse design, independent Maxwell calculations and SLC fabrication geometry. Run commands from the repository root with the corresponding environment active. Use a new output directory when running experiments.

## Reproduce height-conditioned training

The original training used Python 3.12, CUDA 12, an NVIDIA RTX 4060, and the versions recorded in `locks/rtx4060-wsl2-py312-cuda12.txt`. The driver checks the GPU, limits VRAM use, and requires a successful pilot before the training stage. The commands below copy the study configuration into a new run directory.

```bash
study=artifacts/height-family-reproduction
mkdir -p "$study"
cp artifacts/1550nm_height_family_n15263_v2/config.json "$study/config.json"
.venv-gpu/bin/python -B scripts/run_height_family_n15263_v2.py --folder "$study" --stage pilot
.venv-gpu/bin/python -B scripts/run_height_family_n15263_v2.py --folder "$study" --stage train
```

The differentiable model supplies training gradients. A separate Meep 1.35.0 environment performs cylindrical FDTD. Its environment is recorded in `locks/meep-linux-64-explicit.txt`. The solver and field-analysis entry points are `scripts/height_family_maxwell_worker_v2.py` and `scripts/height_family_maxwell_analysis_v2.py`. Both provide `--help` for geometry, reference and analysis arguments.

## Revisit the earlier phases

In the original WSL workflow, `bash scripts/run_phase.sh N` launched phases 1–7 and `bash scripts/run_acceptance_repair.sh` ran the finite-thickness and mesh-convergence repairs. `bash scripts/run_collimation.sh phase8`, `train`, `validate`, and `package` were the initial 1550 nm collimation entry points. Read the applicable configuration and use new output folders before executing these compute stages.

Frozen CPU inference and desktop export use the selected NN weight arrays and reference files shipped in this checkout. Historical training-state replay, validation, fixture tests and manuscript regeneration may require original checkpoints, raw fields, campaign receipts or figure data from the study archive.
