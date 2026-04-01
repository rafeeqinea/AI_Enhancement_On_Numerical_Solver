# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## What This Is

This project uses neural networks to speed up the Conjugate Gradient (CG) method for solving the Poisson equation in 2D and 3D. Instead of replacing the solver, the neural network works alongside it — either by predicting a good starting point (warm-start) or by acting as a preconditioner that reshapes the problem so CG converges faster.

The core experiment is a factorial design that isolates each component's contribution: does the starting point help? Does the preconditioner help? Does the loss function matter? What happens when you combine them?

The key finding: the loss function matters more than the architecture. The same U-Net trained with MSE loss fails completely. Trained with condition loss, it reduces iterations by 78-94%. This comparison has not been published before (confirmed across 73 papers).

## Results Summary

### Completed Cases (v0-v3, 2D)

| Case | Warm-Start | Preconditioner | Loss | Solver | N=32 | N=128 | Reduction | Verdict |
|------|-----------|----------------|------|--------|------|-------|-----------|---------|
| 1 | No | None | — | CG | 81 | 324 | baseline | Raw CG. No help. Slow. |
| 2 | Yes | None | — | CG | 75 | — | 7-9% | Better start, same speed. |
| 3 | No | Jacobi | — | PCG | 81 | 324 | 0% | Useless for constant Poisson. |
| 4 | No | IC(0) | — | PCG | 16 | ~130 | 61-80% | Strong classical baseline. |
| 5 | Yes | IC(0) | — | PCG | 16 | ~130 | 61-80% | WS adds nothing on top of IC(0). |
| 6 | No | U-Net | MSE | FCG | FAIL | FAIL | FAILS | Spectral bias. Only learns low-freq. |
| 7 | No | U-Net | Condition | FCG | 12 | 20 | 85-94% | Beats IC(0) by 2x. Nearly O(1) scaling. |
| 8 | Yes | U-Net | Condition | FCG | 7 | ~20 | 86-94% | Marginal extra from warm-start. |

### 3D Results (v5)

| Case | Method | N=32 (32,768 DOFs) | Reduction |
|------|--------|-------------------|-----------|
| 1 | CG (baseline) | 96 iters | — |
| 4 | IC(0) + PCG | 34 iters | 64.6% |
| 7 | U-Net (condition loss) + FCG | 17 iters | 82.3% |

### Planned Combinations (v4-combined)

| Case | Warm-Start | Preconditioner | Recast | Target Problem | Status |
|------|-----------|----------------|--------|---------------|--------|
| 9 | Yes | IC(0) + U-Net | No | Constant coeff | TODO |
| 10 | No | IC(0) + U-Net | No | Constant coeff | TODO |
| 11 | No | U-Net | Yes | Variable coeff | TODO |
| 12 | No | IC(0) + U-Net | Yes | Variable coeff | TODO |
| 13 | Yes | U-Net | Yes | Variable coeff | TODO |
| 14 | Yes | IC(0) + U-Net | Yes | Variable coeff (full stack) | TODO |

---

## Detailed Case Explanations

### Case 1: Baseline CG (No Help)

```
Warm-start:      OFF (start from x = all zeros)
Preconditioner:  NONE
Solver:          Plain CG
```

This establishes the baseline. CG solves the linear system Au = b by iteratively improving a guess. Each iteration computes one matrix-vector multiply (A × direction vector) and takes an optimal step. The number of iterations depends on the condition number κ(A), which for the Poisson matrix grows as O(N²). This means CG iterations grow as O(N) — double the grid, double the iterations.

```
N=16:   41 iterations
N=32:   81 iterations
N=64:  162 iterations
N=128: 324 iterations
```

Without preconditioning, CG is too slow for large problems. At N=128 in 3D (2 million unknowns), CG would need ~600+ iterations. Each iteration touches every nonzero in the sparse matrix. This is the scaling problem the entire project addresses.

---

### Case 2: Warm-Start Only

```
Warm-start:      ON (trained U-Net predicts initial guess x₀)
Preconditioner:  NONE
Solver:          Plain CG (starting from NN prediction instead of zeros)
```

A U-Net is trained on thousands of Poisson problems to predict what the solution looks like given the source term f. Instead of starting CG from x = 0, we start from x = U-Net(f). The hope is that starting closer to the answer means fewer iterations.

Result: 7-9% reduction. Almost nothing.

The reason is fundamental. CG's convergence rate depends on the condition number κ(A), not on the starting point. A better starting point means the initial residual is smaller, so CG skips the first few easy iterations. But the hard iterations — the ones where convergence slows down because of the condition number — take just as long.

Think of it as getting an Uber to a mountain trailhead instead of walking from the car park. You saved 500 metres but the mountain is still 10 km tall and just as steep. The Uber didn't flatten the mountain.

This result is consistent with the NOWS paper (Eshaghi 2025), the Super-Fidelity paper (Zhou 2025), and Grementieri (2022). Warm-start has a ceiling because it changes WHERE you start but not HOW FAST you converge.

---

### Case 3: Jacobi Preconditioner

```
Warm-start:      OFF
Preconditioner:  Jacobi (divide residual by diagonal of A)
Solver:          PCG (Preconditioned CG)
```

Jacobi is the simplest preconditioner. At each iteration, instead of using the raw residual to pick the search direction, divide it element-wise by the diagonal of A.

For our constant-coefficient Poisson matrix, the diagonal is all 4's (every interior point has the same stencil weight). Dividing everything by 4 doesn't change the ratio between eigenvalues. The condition number κ(A/4) = κ(A). Nothing changes.

Result: 0% reduction. Completely useless for this problem.

Jacobi would help if the diagonal varied (some entries large, some small). It rescales uneven rows to be roughly equal. For our uniform Poisson stencil, there is nothing to rescale. Included for completeness — it shows that not all preconditioners are useful.

---

### Case 4: IC(0) Preconditioner

```
Warm-start:      OFF
Preconditioner:  Incomplete Cholesky IC(0)
Solver:          PCG
```

IC(0) computes an approximate factorization of A: A ≈ L × Lᵀ, where L is lower triangular. "Incomplete" means only entries where A has nonzeros are kept — no fill-in allowed. This keeps L sparse and cheap.

Applying IC(0) at each iteration means solving two triangular systems (forward then backward substitution), each O(n). The result is a preconditioned residual z = (LLᵀ)⁻¹r that gives CG a much smarter search direction.

```
N=16:   10 iterations (was 41)
N=32:   16 iterations (was 81)
N=64:   55 iterations (was 162)
N=128: ~130 iterations (was 324)
```

Reduction: 61-80% depending on grid size.

This is the standard classical baseline. Every preconditioning paper compares against IC(0). If a new method cannot beat IC(0), it has no practical value.

Note: IC(0) can fail on some matrices (negative diagonal during factorization). We experienced this with scipy's spilu, which produced garbage for the Poisson matrix. A custom implementation was needed for reliability.

Also note: IC(0) iterations still grow with N (10 → 16 → 55 → 130). IC(0) improves the constant factor but does not change the scaling class. The NN preconditioner (Case 7) will need to do better.

---

### Case 5: Warm-Start + IC(0)

```
Warm-start:      ON (NN predicts x₀)
Preconditioner:  IC(0)
Solver:          PCG (starting from NN prediction)
```

Case 2 showed warm-start alone gives 7-9%. Case 4 showed IC(0) alone gives 61-80%. This case tests whether they stack: is the combined effect 87%? Or something less?

Result: same as Case 4. 61-80%. Warm-start adds zero benefit on top of IC(0).

IC(0) makes CG converge in 16 iterations at N=32. The warm-start puts x₀ maybe 7% closer to the answer. But when IC(0) reaches tolerance in 16 iterations from ANY starting point, shaving 7% off the starting distance saves less than one iteration. Rounded to the nearest integer: same count.

Conclusion: warm-start and preconditioning are NOT additive. The preconditioner dominates. Once you have a good preconditioner, the starting point does not matter.

---

### Case 6: U-Net Preconditioner Trained with MSE Loss

```
Warm-start:      OFF
Preconditioner:  U-Net trained with MSE loss ||z - e||²
Solver:          FCG (Flexible CG — needed because NN is nonlinear)
```

A U-Net is trained as a preconditioner using supervised learning. Training data: for each random Poisson problem, run CG and record (residual r, error e = u_exact - x_current) pairs at each iteration. Train the U-Net to predict e from r. Loss = ||NN(r) - e||².

This is the obvious approach. It fails completely.

Result: does not converge. Hits maximum iterations.

The failure mechanism is spectral bias. Neural networks learn low-frequency patterns first (smooth bumps) and high-frequency patterns last (sharp wiggles). The training data comes from early CG iterations where residuals are smooth. The NN learns to correct smooth residuals well.

But when FCG starts iterating with this preconditioner, the smooth errors get removed (because the NN IS correcting them). What remains is high-frequency residual — sharp oscillations the NN has never seen during training. The NN's output becomes nearly constant regardless of input (measured: ||output|| ≈ 0.27 for any input). FCG receives a useless "correction" every iteration and cannot converge.

This is not a bug. It is a fundamental mismatch between the training distribution (smooth residuals) and the evaluation distribution (oscillatory residuals after smooth content is removed).

---

### Case 7: U-Net Preconditioner Trained with Condition Loss

```
Warm-start:      OFF
Preconditioner:  U-Net trained with condition loss ||I - AM||²_F
Solver:          FCG
```

Instead of MSE, the U-Net is trained with the Hutchinson estimator of the Frobenius norm ||I - AM||²_F. No training data is needed. No exact solutions. 128 random probe vectors w ~ N(0,1) are generated per step. The loss measures ||w - A × NN(w)||² — how far is A × NN from the identity matrix.

If NN = A⁻¹ perfectly, then A × NN(w) = w for every probe, and loss = 0.

Random probes from a standard normal distribution contain ALL frequencies equally — smooth AND oscillatory. The NN is forced to handle the entire spectrum during training. There is no distribution mismatch during evaluation because the training distribution is universal.

```
N=16:    9 iterations (was 41)   → 78% reduction
N=32:   12 iterations (was 81)   → 85% reduction
N=64:   21 iterations (was 162)  → 87% reduction
N=128:  20 iterations (was 324)  → 94% reduction
N=32 3D: 17 iterations (was 96)  → 82% reduction
```

The iteration count is nearly constant across grid sizes: 9 → 12 → 21 → 20. Compare with CG: 41 → 81 → 162 → 324 (doubles every time). The NN preconditioner achieves O(1)-like iteration scaling — the same behaviour as classical multigrid.

It beats IC(0) at every grid size, and the gap widens at larger N:
- N=32: 12 vs 16 (NN wins by 25%)
- N=64: 21 vs 55 (NN wins by 62%)
- N=128: 20 vs ~130 (NN wins by 85%)

This is the core result of the project.

---

### Case 8: Warm-Start + Condition Loss U-Net

```
Warm-start:      ON (NN predicts initial guess x₀)
Preconditioner:  U-Net trained with condition loss
Solver:          FCG (starting from warm-start prediction)
```

Case 7 is the best single method. This tests whether adding warm-start on top helps.

```
N=32:    7 iterations (was 12 in Case 7 → saved 5)
N=64:  ~21 iterations (was 21 → saved 0)
N=128: ~20 iterations (was 20 → saved 0)
```

Marginal improvement at small grids (5 iterations saved at N=32), zero at large grids. Same conclusion as Case 5: once the preconditioner is strong enough, the starting point is irrelevant. The warm-start model costs extra training time for almost no benefit at large scales.

---

### Cases 9-14: Combined System (Planned, v4-combined branch)

Cases 1-8 tested each ingredient in isolation. Cases 9-14 combine them. See the v4-combined branch for full details and implementation plan.

**Case 9:** IC(0) + U-Net (condition loss) + warm-start on constant coefficients. IC(0) handles the bulk, U-Net handles the leftover. Expected: 10-15 iterations.

**Case 10:** IC(0) + U-Net without warm-start. Isolates the combination effect.

**Case 11:** Equation recast + U-Net on variable coefficients. Extends the breakthrough to harder problems using MD-PNOP perturbation decomposition.

**Case 12:** Equation recast + IC(0) + U-Net on variable coefficients. Full classical+NN stack.

**Case 13:** Warm-start + equation recast + U-Net on variable coefficients.

**Case 14:** Everything combined. All three literature techniques (NOWS warm-start + NPO condition loss + MD-PNOP equation recast) in one pipeline.

---

## Project Structure

```
src/
  data/          poisson.py, generate.py, dataset.py, precond_dataset.py
  solvers/       cg.py, pcg.py, fcg.py, preconditioners.py, direct.py
  models/        unet.py (dimension-generic: 2D and 3D), cnn.py
  training/      train.py, train_precond.py, losses.py
  evaluation/    evaluate.py, evaluate_precond.py, nn_preconditioner.py
  utils/         metrics.py, visualize.py

experiments/
  run_baseline.py          Case 1
  run_warmstart.py         Case 2
  run_preconditioned.py    Cases 3-4
  run_nn_precond.py        Case 6
  run_condition_loss.py    Case 7
  run_factorial.py         All 2D cases
  run_3d.py                3D experiments (Cases 1, 4, 7)

tests/                     144 tests
```

## Branch Structure

```
v0-rebuild (baseline CG, 48 tests)
 └── v1-warmstart-rebuild (warm-start, CNN vs U-Net)
      └── v2-preconditioner (8-case factorial, 109 tests)
           ├── v3-variable-coefficient (variable D(x), equation recast, 123 tests)
           ├── v4-combined (combined system plan, Cases 9-14)
           └── v5-3d (3D extension, 82.3% reduction, 144 tests)
```

Tags follow the pattern `vX.Y-rebuild`.

## Requirements

- Python 3.12
- PyTorch 2.6+ with CUDA
- NumPy, SciPy, Matplotlib

## Running

```bash
python experiments/run_factorial.py          # 2D factorial (Cases 1-8)
python -m experiments.run_3d --all --N 32    # 3D (Cases 1, 4, 7)
python -m pytest tests/ -v                   # all tests
```

## Papers This Builds On

- NOWS (Eshaghi 2025) — neural operator warm-starts for CG
- NPO (Li 2025) — condition loss ||I-AM||²_F via Hutchinson for learned preconditioners
- FCG-NO (Rudikov 2024) — flexible CG with neural operator preconditioner
- Azulay & Treister (2022) — U-Net as multigrid V-cycle analogy
- MD-PNOP (Cheng 2025) — equation recast for parametric generalisation
