# Learned Preconditioning for Conjugate Gradient Solvers

BSc Final Year Project — University of Greenwich, 2025-26

**A Controlled Factorial Evaluation of Training Objectives for the Poisson Equation**

Supervised by Dr Peter Soar

## Project Overview

This project investigates whether the choice of training objective determines the effectiveness of a learned neural network preconditioner for Conjugate Gradient solvers applied to the Poisson equation. A U-Net neural network is integrated as a preconditioner within a Flexible Conjugate Gradient (FCG) framework, and two training objectives — mean squared error (MSE) and a condition-loss surrogate estimated via the Hutchinson trace estimator — are compared under matched conditions through a controlled factorial experimental design.

## Core Finding

Under matched architecture, solver, initial guess, and evaluation conditions, the MSE-trained preconditioner did not converge at any tested grid size, while the condition-loss-trained preconditioner reduced iteration counts by 78 to 84 percent across N=16, 32, and 64. The training objective was the factor that determined whether the preconditioner converged.

## Repository Structure

```
src/
  data/         - Poisson assembly, source generation, dataset
  solvers/      - CG, PCG, FCG, direct solver, preconditioners
  models/       - Dimension-generic U-Net (2D/3D)
  training/     - MSE and condition-loss training pipelines
  evaluation/   - NN preconditioner wrapper, evaluation metrics
experiments/    - Experiment scripts (factorial, curriculum, 3D, analysis)
tests/          - 161 automated tests across 22 files
results/        - Committed JSON artefacts, figures, checkpoints
```

## Headline Findings

| Case | Preconditioner | Loss | N=32 | Reduction |
|------|---------------|------|------|-----------|
| 1 | None (CG) | — | 80.6 | baseline |
| 4 | IC(0) | — | 28.2 | 65% |
| 6 | U-Net | MSE | 1000 | FAILS |
| 7 | U-Net | Condition | 12.7 | 84% |

## Full Core Factorial Results (committed)

Source: `results/factorial/results.json` (Cases 1–5, 7, 8)

| Case | x0 | Preconditioner | Loss | Solver | N=16 | N=32 | N=64 | Reduction |
|------|-----|---------------|------|--------|------|------|------|-----------|
| 1 | Zero | None | — | CG | 40.6 | 80.6 | 161.5 | baseline |
| 2 | Warm-start | None | MSE | CG | 43.6 | 86.8 | 175.9 | -7 to -9% |
| 3 | Zero | Jacobi | — | PCG | 40.6 | 80.6 | 161.5 | 0% |
| 4 | Zero | IC(0) | — | PCG | 15.7 | 28.2 | 54.0 | 61-67% |
| 5 | Warm-start | IC(0) | MSE | PCG | 16.2 | 29.8 | 56.2 | 60-65% |
| 7 | Zero | U-Net | Condition | FCG | 9.0 | 12.7 | 27.0 | 78-84% |
| 8 | Warm-start | U-Net | Condition | FCG | 9.0 | 11.0 | 45.4 | 72-86% |

**Case 6 (MSE-trained U-Net preconditioner):** evaluated separately via `results/nn_precond/mse_results.json`. Did not converge at any grid size (1000-iteration cap reached at N=16, 32, and 64).

## Testing

```bash
python -m pytest tests/ -v
# 161 passed (verified 2026-04-12)
```

## Requirements

Python 3.12, PyTorch 2.6+, SciPy, NumPy, matplotlib, pyamg (optional for AMG baseline)
