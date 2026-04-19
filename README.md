# Learned preconditioning for Conjugate Gradient solvers

BSc Final Year Project, University of Greenwich, 2025-26

**A controlled factorial evaluation of training objectives for the Poisson equation**

Supervised by Dr Peter Soar

## Project overview

This project asks a narrow question and tests it directly: does the training objective determine whether a learned neural preconditioner actually helps Conjugate Gradient on the Poisson equation?

A U-Net is integrated into a Flexible Conjugate Gradient (FCG) solver and compared under matched conditions using two training objectives:
- mean squared error (MSE)
- a condition-loss surrogate estimated with the Hutchinson trace estimator

The core study is an eight-case factorial design that isolates training objective while holding the architecture, solver family, and evaluation setup fixed.

## Core finding

Under matched architecture, solver, initial guess, and evaluation conditions, the MSE-trained preconditioner did not converge at any tested grid size. The condition-loss-trained preconditioner reduced iteration counts by 78 to 84 percent across `N=16`, `32`, and `64`. In this setup, the training objective determined whether the learned preconditioner worked at all.

## Repository structure

```text
src/
  data/         Poisson assembly, source generation, datasets
  solvers/      CG, PCG, FCG, direct solver, preconditioners
  models/       Dimension-generic U-Net (2D/3D) and CNN warm-start baseline
  training/     MSE and condition-loss training pipelines
  evaluation/   NN preconditioner wrapper and evaluation helpers
experiments/    Reproducible experiment scripts and figure builders
tests/          161 automated tests across 22 files
results/        Committed artefacts, figures, checkpoints, and summaries
```

## Headline results

| Case | Preconditioner | Loss | N=32 | Reduction |
|------|----------------|------|------|-----------|
| 1 | None (CG) | — | 80.6 | baseline |
| 4 | IC(0) | — | 28.2 | 65% |
| 6 | U-Net | MSE | 1000 | fails to converge |
| 7 | U-Net | Condition | 12.7 | 84% |

## Full core factorial results

Source: `results/factorial/results.json` (Cases 1–5, 7, 8)

| Case | x0 | Preconditioner | Loss | Solver | N=16 | N=32 | N=64 | Reduction |
|------|----|----------------|------|--------|------|------|------|-----------|
| 1 | Zero | None | — | CG | 40.6 | 80.6 | 161.5 | baseline |
| 2 | Warm-start | None | MSE | CG | 43.6 | 86.8 | 175.9 | -7 to -9% |
| 3 | Zero | Jacobi | — | PCG | 40.6 | 80.6 | 161.5 | 0% |
| 4 | Zero | IC(0) | — | PCG | 15.7 | 28.2 | 54.0 | 61-67% |
| 5 | Warm-start | IC(0) | MSE | PCG | 16.2 | 29.8 | 56.2 | 60-65% |
| 7 | Zero | U-Net | Condition | FCG | 9.0 | 12.7 | 27.0 | 78-84% |
| 8 | Warm-start | U-Net | Condition | FCG | 9.0 | 11.0 | 45.4 | 72-86% |

**Case 6 (MSE-trained U-Net preconditioner):** evaluated separately via `results/nn_precond/mse_results.json`. It hit the 1000-iteration cap at `N=16`, `32`, and `64`.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Requirements:
- Python 3.12
- PyTorch 2.6+
- SciPy
- NumPy
- matplotlib
- pyamg (optional, for the AMG baseline)

## Running the checks

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Running the main experiments

These are the scripts that matter for the dissertation results:

```bash
python experiments/run_factorial.py
python experiments/run_condition_loss.py
python experiments/run_curriculum.py
python experiments/run_3d.py
python experiments/run_spectral_analysis.py
python experiments/run_statistical_analysis.py
```

The exact outputs depend on the script, but the committed artefacts already used in the dissertation are in `results/`.

## Artefact map

The main committed artefacts are:

- `results/factorial/results.json` — core eight-case factorial results used in the main comparison
- `results/nn_precond/mse_results.json` — separate MSE evaluation used to document Case 6 failure
- `results/curriculum/2d/` — curriculum transfer artefacts, including the `N=128` extension
- `results/spectral_analysis/spectrum_N16.json` — eigenvalue evidence used to interpret the MSE failure
- `results/statistical_analysis.json` — Wilcoxon and Holm-Bonferroni results
- `results/3d/v4_3d_results_N32.png` — committed 3D evaluation figure at `N=32`

## Caveats

This repo preserves the same caveats as the dissertation:

- The strongest evidence is the 2D factorial study. It is backed by committed JSON artefacts.
- The 3D `N=32` extension is reported from a committed evaluation figure, not a separate JSON file.
- The learned preconditioner reduced iteration counts but did not beat classical baselines on wall-clock time at the tested grid sizes.
- The variable-coefficient extension failed on all three tested patterns.
- `N=256` and `N=512` scaling work is exploratory and is not part of the committed core evaluation chain.

## Colab notebooks

Two exploratory notebooks are tracked at the repo root:

- `colab_n512.ipynb` — hosted-cloud exploratory run for `N=512`
- `colab_3d_n64.ipynb` — hosted-cloud exploratory 3D `N=64` run

These are not part of the core committed evaluation used for the main factorial result.

## Report note

This repository focuses on code and committed experimental artefacts. The dissertation PDF and LaTeX submission files were handled separately from the tracked `main` branch.
