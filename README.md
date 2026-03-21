# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v0 — Baseline CG solver

This branch builds the foundation: a custom Conjugate Gradient solver for the 2D Poisson equation on a structured grid.

### What's here

- **Poisson assembly** (`src/data/poisson.py`) — 5-point FDM stencil via Kronecker product. The system matrix A is built as `T x I + I x T` where T is tridiag(-1, 2, -1). Works for any grid size N.

- **CG solver** (`src/solvers/cg.py`) — Written from scratch (not scipy) so we can track per-iteration residuals and plug in warm-starts later. Returns a `CGResult` with the solution, iteration count, convergence flag, residual history, and timing.

- **Direct solver** (`src/solvers/direct.py`) — Wraps scipy's sparse direct solve. Used as ground truth to verify CG gives the right answer.

- **Data generation** (`src/data/generate.py`) — Random source terms from superpositions of 3-8 Gaussian blobs. Seeded for reproducibility.

- **Baseline experiments** (`experiments/run_baseline.py`) — Runs CG on grids N=8 to N=128. At N=64, CG takes about 167 iterations. At N=128, about 329. Iterations scale as O(N), matching the theory since the condition number grows as O(N^2) and CG convergence goes as O(sqrt(kappa)).

### Results

| N | DOF | CG iterations | Time (s) |
|---|-----|--------------|----------|
| 8 | 64 | 11 | 0.0001 |
| 16 | 256 | 40 | 0.0003 |
| 32 | 1024 | 81 | 0.001 |
| 64 | 4096 | 167 | 0.005 |
| 128 | 16384 | 329 | 0.03 |

### Tests

48 tests covering the solver, assembly, data generation, metrics, and visualisation.

```bash
python -m pytest tests/ -v
```
