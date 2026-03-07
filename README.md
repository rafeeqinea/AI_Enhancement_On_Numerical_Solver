# AI Enhancement of Numerical Solvers

**M34 Final Year Project — University of Greenwich**

---

## What This Project Does

Uses neural networks to accelerate the Conjugate Gradient (CG) solver for the 2D Poisson equation through two independent mechanisms:

1. **Warm-start (MD-PNOP concept)** — A U-Net predicts the solution field, which is fed as the initial guess x₀ to CG, reducing the number of iterations needed to converge.

2. **Learned preconditioner (NPO concept)** — A U-Net trained with condition loss ||I - AM||²_F acts as a nonlinear preconditioner inside Flexible CG (FCG), clustering the eigenvalues of the preconditioned system near 1 for faster convergence.

The core contribution is a **controlled factorial experiment** isolating each mechanism's effect independently, then combining them to demonstrate multiplicative iteration savings — something no prior work has done.

---

## The Problem

2D Poisson equation on the unit square with homogeneous Dirichlet boundary conditions:

```
-∇²u = f    on [0,1]²
   u = 0    on ∂Ω
```

Discretised using the **5-point finite difference stencil**, assembled via Kronecker products into a sparse SPD system **Ax = b**, and solved with Conjugate Gradient.

**Why CG slows down:** The condition number κ(A) = O(N²), so CG iterations grow linearly with grid refinement. At N=64, CG takes ~167 iterations. At N=128, ~329 iterations. At N=256, ~650+ iterations.

**Why not multigrid?** For constant-coefficient Poisson on a regular grid, geometric multigrid is O(N) — the gold standard. ML targets problems where multigrid is harder: variable coefficients, complex geometries, unstructured meshes. This project uses constant-coefficient Poisson as a controlled testbed to validate the methodology before extending to harder problems.

---

## The Experiment: 8-Case Factorial Design

Two axes, fully crossed: **initial guess** (zero vs NN warm-start) × **preconditioner** (none, classical, NN-MSE, NN-condition-loss).

| Case | Initial Guess (x₀) | Preconditioner (M⁻¹) | Loss Function | Solver | What It Proves |
|------|--------------------|-----------------------|---------------|--------|----------------|
| 1 | Zero vector | None | N/A | CG | Pure baseline — no ML |
| 2 | **U-Net warm-start (MD-PNOP concept)** — NN predicts solution, feeds as x₀ | None | MSE on solution | CG | Warm-start boost in isolation |
| 3 | Zero vector | **Jacobi** — M = diag(A), trivial classical | N/A | PCG | Weak classical preconditioner baseline |
| 4 | Zero vector | **IC(0)** — Incomplete Cholesky, standard classical | N/A | PCG | Strong classical preconditioner baseline |
| 5 | **U-Net warm-start (MD-PNOP concept)** | **IC(0)** | MSE on solution | PCG | Warm-start + strong classical. Does warm-start still help with a good preconditioner? |
| 6 | Zero vector | **U-Net preconditioner** — NN maps residual r → correction z ≈ A⁻¹r | MSE: \|\|z - A⁻¹r\|\|² | FCG | NN preconditioner with naive loss. FCG needed because NN is nonlinear (ReLU) |
| 7 | Zero vector | **U-Net preconditioner (NPO concept)** — same architecture, different training | **Condition loss: \|\|I - AM\|\|²_F** via Hutchinson trace estimator | FCG | NPO's key claim: condition loss beats MSE. Case 6 vs 7 isolates this |
| 8 | **U-Net warm-start (MD-PNOP concept)** | **U-Net preconditioner (NPO concept)** | MSE (warm-start) + Condition loss (preconditioner) | FCG | **Full stack — MD-PNOP + NPO combined.** Multiplicative savings: warm-start reduces \|\|e₀\|\|, condition loss reduces κ |

**Stretch cases (if time permits):**

| Case | Initial Guess (x₀) | Preconditioner (M⁻¹) | Loss Function | Solver | What It Proves |
|------|--------------------|-----------------------|---------------|--------|----------------|
| 9 | Zero vector | **IC(0) + U-Net correction** — hybrid classical-neural (PreCorrector concept) | A⁻¹-weighted Frobenius | FCG | Does correcting a classical preconditioner beat learning from scratch? |
| 10 | **U-Net warm-start (MD-PNOP concept)** | **IC(0) + U-Net correction** | MSE + A⁻¹-weighted Frobenius | FCG | Ultimate combination — all three innovations |

All cases tested at **N = {64, 128, 256}** reporting:
- Iteration count
- Wall-time (NN inference + solver time, honest breakdown)
- Eigenvalue spectrum of M⁻¹A
- Residual convergence curves (log-residual vs iteration)
- Error bars over 30+ test problems

---

## Why This Is Novel

Confirmed by exhaustive search across 73 papers + 8 academic database queries + 5 independent review agents:

| Gap | Status |
|-----|--------|
| Full factorial warm-start × preconditioner × loss function comparison? | **No paper does this** |
| Condition loss combined with warm-start? | **No paper does this** |
| MSE vs condition loss on same architecture, same problem? | **No clean comparison exists** |
| NN warm-start + NN preconditioner in Flexible CG? | **No paper does this** |

The closest work is Li et al. (ICML 2023), which produces both a preconditioner and initial guess from a single GNN — but does not isolate their contributions, does not test condition loss, and uses unstructured meshes with standard PCG (not FCG).

---

## Key Papers Backing This Work

| Paper | Year | Concept Used |
|-------|------|-------------|
| **FCG-NO** (Rudikov et al.) | 2024 | Flexible CG framework for nonlinear NN preconditioner |
| **Azulay & Treister** | 2022 | U-Net ≈ multigrid V-cycle (architectural justification) |
| **NPO** (Li et al.) | 2025 | Condition loss \|\|I - AM\|\|²_F, NAMG architecture |
| **MD-PNOP** (Cheng et al.) | 2025 | Equation-recast warm-start framework |
| **NOWS** (Eshaghi et al.) | 2025 | Neural operator warm-starts for Krylov solvers |
| **Solver-in-the-Loop** (Um et al., NeurIPS) | 2020 | U-Net warm-start for CG on Poisson (smoke simulation) |
| **RBWS** (Hou et al.) | 2025 | Convergence theory: warm-start (reduces \|\|e₀\|\|) × preconditioner (reduces κ) = multiplicative savings |
| **PreCorrector** (Trifonov et al.) | 2025 | A⁻¹-weighted Frobenius loss, correction-based preconditioner |
| **HINTS** (Zhang et al.) | 2024 | Spectral bias framework — NN captures low-freq, solver handles high-freq |

Full literature review: 73 papers read, 20 HIGH relevance, 23 MEDIUM, 28 LOW/SKIP.

---

## Project Structure

```
src/
  data/
    poisson.py        -- FDM assembly: A matrix (Kronecker), b vector, grid points
    generate.py       -- Random dataset generation (Gaussian blob sources)
    dataset.py        -- PyTorch Dataset: (source, solution) pairs [v1]
  models/
    cnn.py            -- BaselineCNN (~195K params, 7 layers) [v1]
    unet.py           -- U-Net (~600K params, base_features=16, 3 levels) [v1]
  solvers/
    direct.py         -- Exact solver (SciPy spsolve)
    cg.py             -- Conjugate Gradient with per-iteration diagnostics
  utils/
    metrics.py        -- Error metrics, speedup calculations, timing
    visualize.py      -- Solution heatmaps, convergence plots, comparison figures
experiments/
    run_baseline.py   -- v0 baseline experiment runner
    train.py          -- Model training script [v1]
    evaluate.py       -- Warm-start evaluation script [v1]
tests/               -- 48 unit tests (all passing)
data/processed/      -- Generated datasets (sources + solutions)
results/figures/     -- Output plots from experiments
```

---

## Progress

### v0 — Baseline Infrastructure: COMPLETE

| Tag | What | Tests | Status |
|-----|------|-------|--------|
| v0.1 | `poisson.py`, `direct.py`, `cg.py`, `generate.py` — core FDM + solvers | 20 | Done |
| v0.2 | `metrics.py`, `visualize.py`, `run_baseline.py` — evaluation infrastructure | 37 | Done |
| v0.3 | Full experiment runner, results, figures, README | 37 | Done |

**Key numbers:** CG ~167 iters (N=64), ~329 iters (N=128) at tol=1e-6.

### v1 — ML Warm-Start (MD-PNOP concept): IN PROGRESS

| Tag | What | Status |
|-----|------|--------|
| v1.1 | PyTorch Dataset + BaselineCNN architecture | Files exist, need rewrite |
| v1.2 | U-Net warm-start + CNN vs U-Net comparison | Not started |
| v1.3 | Ablations: tolerance sweep, data size sweep, OOD, failure cases, spectral bias | Not started |

### v2 — ML Preconditioner (NPO concept): NOT STARTED

| Tag | What | Status |
|-----|------|--------|
| v2.1 | Jacobi + IC(0) classical baselines, FCG solver implementation | Not started |
| v2.2 | U-Net preconditioner with MSE loss in FCG | Not started |
| v2.3 | Condition loss \|\|I - AM\|\|²_F via Hutchinson estimator, MSE vs condition loss comparison | Not started |
| v2.4 | Combined warm-start + preconditioner — full 8-case factorial comparison table | Not started |

### Stretch — Hybrid Classical-Neural: NOT STARTED

| Tag | What | Status |
|-----|------|--------|
| v2.5 | IC(0) + NN correction (PreCorrector concept), cases 9-10 | Not started |

---

## Branch / Tag Structure

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Full project overview and plan | Up to date |
| `v0-baseline` | Classical solvers, no ML | Complete (v0.1, v0.2, v0.3) |
| `v1-cnn-warmstart` | Warm-start models | In progress (v1.1 placeholder) |
| `v2-preconditioner` | Learned preconditioner + combined | Future |

### Viewing Tags

On GitHub: click the branch dropdown → **Tags** tab → select v0.1 / v0.2 / v0.3 / v1.1

Locally:
```bash
git checkout v0.1          # browse code at that stage
git checkout v1-cnn-warmstart  # return to current work
```

---

## Quick Start

```bash
# Run all tests (48 tests, all passing)
python -m pytest tests/ -v

# Run baseline experiments
python experiments/run_baseline.py

# Generate a dataset
python -c "from src.data.generate import generate_dataset; generate_dataset(N=32, num_samples=100, seed=42)"
```

---

## Technical Details

### Why Flexible CG (FCG) instead of standard PCG?

Standard PCG requires the preconditioner M⁻¹ to be a **fixed linear SPD operator**. A neural network with ReLU activations is nonlinear — applying it twice to the same input gives the same output, but M⁻¹(r₁ + r₂) ≠ M⁻¹(r₁) + M⁻¹(r₂). This breaks PCG's conjugacy property and can cause divergence.

FCG (Notay 2000, Rudikov et al. 2024) handles this by re-orthogonalising search directions against all previous directions, at the cost of O(k·N²) extra storage and inner products. This is the approach validated by FCG-NO (Rudikov 2024) and Azulay (2022).

### Why U-Net for both warm-start and preconditioner?

Azulay & Treister (2022) proved that a U-Net's encoder-decoder structure mirrors a multigrid V-cycle:
- Downsampling = restriction operator
- Upsampling = prolongation operator
- Skip connections = grid transfer operators
- Convolutions at each level = smoothing iterations

This makes U-Net a **learned multigrid operator** — architecturally suited for both predicting solutions (warm-start) and approximating A⁻¹ (preconditioner).

### Why condition loss instead of MSE for the preconditioner?

MSE \|\|z - A⁻¹r\|\|² optimises for pointwise accuracy of the correction vector. But CG convergence depends on the **condition number** κ(M⁻¹A), not pointwise accuracy. A preconditioner that perfectly predicts z at some points but misses others can still have poor spectral properties.

Condition loss \|\|I - AM\|\|²_F directly measures how close the preconditioned system is to the identity — if M⁻¹A = I, CG converges in 1 iteration. NPO (Li et al. 2025) demonstrated this produces better eigenvalue clustering and fewer CG iterations than MSE or residual loss.

### The FLOP Budget Problem

At N=64 (4096 unknowns), one CG iteration costs ~20K FLOPs (one SpMV with 5-point stencil). A 9.4M-parameter U-Net costs ~18.8M FLOPs per forward pass — equivalent to ~940 CG iterations. This means the NN inference alone exceeds the cost of the entire baseline CG solve (167 iterations).

**Solution:** Use a smaller U-Net (~600K params, base_features=16, 3 levels) and test at N={64, 128, 256}. At N=256, CG takes 650+ iterations, and the NN inference cost becomes a small fraction of total solve time.

---

## Future Work (mentioned in report, not implemented)

- Solver-in-the-Loop training: backpropagate through differentiable CG (Um et al. 2020)
- Variable-coefficient Poisson: -div(D(x)∇u) = f — where multigrid genuinely struggles
- Resolution transfer: train on N=64, deploy on N=128 without retraining (FCG-NO demonstrated this)
- 3D extension: linear elasticity (supervisor's stretch goal)
- Deflation-based acceleration: DeepONet-generated deflation subspaces (Kopanicakova et al. 2025)
- Spectral Neural Operator as preconditioner (FCG-NO architecture)
