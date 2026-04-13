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

## Key Results (from committed artefacts)

| Case | Preconditioner | Loss | N=16 | N=32 | N=64 | Reduction | Source |
|------|---------------|------|------|------|------|-----------|--------|
| 1 | None (CG) | — | 40.6 | 80.6 | 161.5 | baseline | `results/factorial/results.json` |
| 4 | IC(0) | — | 15.7 | 28.2 | 54.0 | 61-67% | `results/factorial/results.json` |
| 6 | U-Net | MSE | 1000 | 1000 | 1000 | FAILS | `results/nn_precond/mse_results.json` |
| 7 | U-Net | Condition | 9.0 | 12.7 | 27.0 | 78-84% | `results/factorial/results.json` |

Note: Case 6 is not in `results/factorial/results.json` because it was run through a separate MSE training and evaluation script. Its results are in `results/nn_precond/mse_results.json`.

## Testing

```bash
python -m pytest tests/ -v
# 161 passed (verified 2026-04-12)
```

## Requirements

Python 3.12, PyTorch 2.6+, SciPy, NumPy, matplotlib, pyamg (optional for AMG baseline)
