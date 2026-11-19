# Fiber metalens inverse design

This repository contains the neural inverse design, optical simulation and SLC export code for a fiber-mounted polymer collimator at 1550 nm. The height-conditioned network predicts corrections to a phase-derived annular relief. A differentiable optical model estimates useful modal efficiency, and Meep provides an independent full-wave comparison.

The reference design uses a 10.4 µm Gaussian fiber mode-field diameter, a 50 µm diameter cylinder with polymer index 1.5263, and 50 radial bands each 0.5 µm wide. The frozen model accepts cylinder heights of 40–100 µm. The desktop tool selects heights at 0.5 µm intervals and produces 50 nm and 500 nm relief profiles, with a 500 nm layer SLC export of the connected cylinder and relief.

## Run the frozen model or the desktop app

Use Python 3.12 and install the CPU dependencies in a virtual environment. Tk is required for the desktop window. The locks in `locks/` describe the original CUDA and CPU Meep environments. Frozen inference requires the study's checkpoint and reference files in `artifacts/1550nm_height_family_n15263_v2/`. Obtain these separately before running the inference or desktop commands.

```text
python -m pip install -r requirements-cpu.txt
python scripts/fiber_metalens_desktop_inference.py --height-um 67.5 --backend cpu
python scripts/fiber_metalens_desktop.py
```

Run these commands from the repository root. In the window, enter `67.5`, select **Generate design**, and then select **Save 500 nm SLC**. Export writes a geometry audit and a source manifest alongside the SLC. The input indices are modeled assumptions, and the displayed efficiencies are forward-model predictions.

## Source and reproducibility

- `src/fiber_metalens_nn/` contains the optical models, neural training support, geometry and metrics.
- [`scripts/run_height_family_n15263_v2.py`](scripts/run_height_family_n15263_v2.py) contains the original height-conditioned model, decoder and training loop. It uses JAX 0.6.2, Flax 0.10.6 and Optax 0.2.5.
- `scripts/height_family_maxwell_worker_v2.py` and `scripts/height_family_maxwell_analysis_v2.py` implement the Meep 1.35.0 cylindrical simulations and analysis. Some archived validation and report drivers require the original raw fields or campaign ledgers.
- `scripts/fiber_metalens_desktop_slc.py`, `scripts/audit_metalens_slc.py` and `vendor/` contain the export route, independent contour audit and the extracted SLC writer with provenance.
- `configs/`, `locks/` and `tests/` contain physical configurations, dependency locks and regression checks. Tests for historical fabrication binaries or saved field arrays require their original fixtures. See [research workflow](docs/research-workflow.md) for the original phase entry points and [project assumptions](assumptions.md) for modeling scope.

The frozen inference entry point verifies source, configuration and checkpoint hashes before loading the selected parameters. Load checkpoints only from a trusted source.
