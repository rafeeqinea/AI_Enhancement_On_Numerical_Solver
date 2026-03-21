# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v2 — Learned preconditioner with condition loss

This branch is the core of the project. It implements classical and learned preconditioners, plugs them into PCG and FCG solvers, and runs the 8-case factorial experiment that compares everything.

### The 8-case factorial

Each case turns on or off a specific component so we can measure its individual contribution.

| Case | x0 | Preconditioner | Loss | Solver | N=32 | N=64 | N=128 |
|------|-----|---------------|------|--------|------|------|-------|
| 1 | Zero | None | — | CG | 81 | 162 | 324 |
| 2 | U-Net | None | MSE | CG | 87 | 176 | — |
| 3 | Zero | Jacobi | — | PCG | 81 | 162 | 324 |
| 4 | Zero | IC(0) | — | PCG | 28 | 54 | 105 |
| 5 | U-Net | IC(0) | MSE | PCG | 30 | 56 | — |
| 6 | Zero | U-Net | MSE | FCG | >1000 | >1000 | — |
| 7 | Zero | U-Net | Condition | FCG | 13 | 27 | 20 |
| 8 | U-Net | U-Net | Condition | FCG | 11 | 45 | — |

### What the numbers say

**Jacobi does nothing for this problem** (Case 3 = Case 1). The 5-point Poisson stencil has a constant diagonal, so Jacobi just scales everything by 1/4. Condition number stays the same.

**IC(0) is the real classical baseline** (Case 4). Cuts iterations by 61-67%. This is what the learned preconditioner has to beat.

**MSE-trained preconditioner fails** (Case 6). The U-Net learns to predict A^{-1}r in pixel space, which only captures low-frequency patterns. After CG removes those in the first few iterations, the remaining residual is high-frequency. The U-Net outputs near-constant values for high-frequency input and FCG cannot converge.

**Condition-loss preconditioner works** (Case 7). Trained with the Hutchinson estimator of ||I - AM||^2_F using 32 random probes per gradient step. Random probes cover all frequencies, so the U-Net has to handle them all. At N=128: CG takes 324 iterations, the condition-loss preconditioner takes 20. That is a 93.9% reduction.

**Warm-start adds nothing to preconditioning** (Case 5 vs 4, Case 8 vs 7 at N=64). The v1 warm-start model was trained for a different task (source-to-solution with MSE). Its prediction introduces A-norm error that the preconditioner then has to fight against.

### What's here

- `src/solvers/preconditioners.py` — Jacobi (diagonal inverse) and IC(0) (manual incomplete Cholesky)
- `src/solvers/pcg.py` — Standard preconditioned CG
- `src/solvers/fcg.py` — Flexible CG for nonlinear preconditioners (from FCG-NO, Algorithm 1)
- `src/training/losses.py` — Hutchinson condition loss with sparse A as torch buffer
- `src/data/precond_dataset.py` — Krylov subspace training data (residual-error pairs from CG iterations)
- `src/training/train_precond.py` — MSE training pipeline for preconditioner
- `src/evaluation/nn_preconditioner.py` — Wraps trained U-Net as callable for FCG
- `experiments/run_factorial.py` — Runs all 8 cases, outputs the comparison table

### Tests

109 tests.

```bash
python -m pytest tests/ -v
python experiments/run_factorial.py
```
