# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## What this is

This project uses neural networks to speed up the Conjugate Gradient (CG) method for solving the 2D Poisson equation. Instead of replacing the solver, the neural network works alongside it — either by predicting a good starting point (warm-start) or by acting as a preconditioner that reshapes the problem so CG converges faster.

The core experiment is an 8-case factorial that isolates each component's contribution: does the starting point help? does the preconditioner help? does the loss function matter? what happens when you combine them?

## Results

Tested on the 2D Poisson equation with 5-point FDM stencil, Dirichlet boundaries, random Gaussian blob source terms.

| Case | Initial guess | Preconditioner | Loss | Solver | N=32 iters | N=64 iters | N=128 iters |
|------|--------------|----------------|------|--------|-----------|-----------|------------|
| 1 | Zero | None | — | CG | 81 | 162 | 324 |
| 2 | U-Net | None | MSE | CG | 87 | 176 | — |
| 3 | Zero | Jacobi | — | PCG | 81 | 162 | 324 |
| 4 | Zero | IC(0) | — | PCG | 28 | 54 | 105 |
| 6 | Zero | U-Net | MSE | FCG | >1000 | >1000 | — |
| 7 | Zero | U-Net | Condition | FCG | **13** | **27** | **20** |
| 8 | U-Net | U-Net | Condition | FCG | **11** | 45 | — |

The condition-loss preconditioner (Case 7) reduces iterations by 78-94%. The MSE-trained preconditioner (Case 6) fails completely — spectral bias makes it only handle low-frequency residuals, while the condition loss forces uniform spectral coverage via Hutchinson random probes.

## How it works

**Warm-start (v1):** A U-Net predicts the solution field from the source term. CG starts from this prediction instead of zero. On small grids this actually hurts (-7 to -9%) because the prediction error in the A-norm is larger than the benefit. Consistent with the NOWS paper (Eshaghi 2025), which shows warm-start helps more at larger grid sizes.

**Learned preconditioner (v2):** A separate U-Net is trained to approximate the inverse operator A^{-1}. At each FCG iteration, the residual is fed through the U-Net to produce a preconditioned search direction. The training loss matters more than the architecture:

- MSE loss: the U-Net learns to match A^{-1}r in pixel space. This captures low-frequency patterns but ignores high-frequency structure. After a few CG iterations remove the low-frequency content, the U-Net outputs near-constant values. FCG cannot converge.

- Condition loss: ||I - AM||^2_F estimated via Hutchinson trace with 32 random probes per step. This directly measures how close the preconditioned system is to identity. Random probes span all frequencies, so the U-Net has to handle all of them. FCG converges in 9-27 iterations.

**Variable coefficients (v3):** The preconditioner trained on one operator cannot handle a different operator — it only sees the residual, not the coefficient field D(x). IC(0) adapts automatically because it is recomputed from each A. MD-PNOP's equation recast approach would fix this by decomposing parameter changes into additional source terms (in progress).

## Project structure

```
src/
  data/          poisson.py, generate.py, dataset.py, precond_dataset.py
  solvers/       cg.py, pcg.py, fcg.py, preconditioners.py, direct.py
  models/        unet.py, cnn.py
  training/      train.py, train_precond.py, losses.py
  evaluation/    evaluate.py, evaluate_precond.py, nn_preconditioner.py
  utils/         metrics.py, visualize.py

experiments/
  run_baseline.py          Case 1
  run_warmstart.py         Case 2
  run_preconditioned.py    Cases 3-4
  run_nn_precond.py        Case 6
  run_condition_loss.py    Case 7
  run_factorial.py         All cases

tests/                     118 tests
```

## Branch and tag structure

| Branch | What it contains |
|--------|-----------------|
| main | Base |
| v0-rebuild | Poisson assembly, CG solver, baseline experiments |
| v1-warmstart-rebuild | U-Net/CNN warm-start training and evaluation |
| v2-preconditioner | PCG, FCG, IC(0), learned preconditioner, 8-case factorial |
| v3-variable-coefficient | Variable-coefficient Poisson, diffusion coefficient generators |

Tags follow the pattern `vX.Y-rebuild` (e.g. `v2.4-rebuild` for the full factorial).

## Requirements

- Python 3.12
- PyTorch (CUDA)
- NumPy, SciPy, Matplotlib

## Running

```bash
python experiments/run_factorial.py
python -m pytest tests/ -v
```

## Papers this builds on

- NOWS (Eshaghi 2025) — neural operator warm-starts for CG
- NPO (Li 2025) — condition loss ||I-AM||^2_F via Hutchinson for learned preconditioners
- FCG-NO (Rudikov 2024) — flexible CG with neural operator preconditioner
- Azulay & Treister (2022) — U-Net as multigrid V-cycle analogy
- MD-PNOP (Cheng 2025) — equation recast for parametric generalisation
