# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v3 — Variable-coefficient Poisson

This branch extends the solver to handle spatially varying diffusion coefficients: -div(D(x) grad u) = f. When D(x) is not constant, the stencil changes at every grid point and the system becomes harder to solve.

### What changed from v2

The constant-coefficient Poisson has a uniform stencil: every node sees the same [-1, 4, -1] pattern. The condition-loss U-Net preconditioner trained on this operator reduces CG iterations by 78-94%.

Variable coefficients break this. The stencil weights now depend on the local D(x) values (using harmonic averaging at cell faces). The operator A changes for every different D(x) field. A preconditioner trained on one A cannot invert a different A.

### Three coefficient patterns

- **Smooth** — Gaussian bump in D(x). Condition number around 700. CG needs ~130 iterations at N=32.
- **Discontinuous** — Circular inclusion with contrast up to 67:1. Condition number around 18,700. CG needs ~490 iterations.
- **Layered** — Piecewise constant bands. Condition number around 2,700. CG needs ~280 iterations.

### Results at N=32

| Pattern | CG | IC(0)+PCG | NN+FCG |
|---------|-----|-----------|--------|
| Smooth | 130 | 30 (77%) | >1000 (fails) |
| Discontinuous | 491 | 36 (93%) | >1000 (fails) |
| Layered | 141 | 29 (80%) | >1000 (fails) |

IC(0) adapts per-problem because it is recomputed from each A. The U-Net preconditioner fails because it only sees the residual vector, not the coefficient field D(x). It has no way to know which operator it should be inverting.

### What this means

The preconditioner needs to either (a) take D(x) as an additional input channel so it can condition on the coefficient, or (b) use MD-PNOP's equation recast approach where you train on a reference D0 and decompose new D(x) as D0 plus perturbation source terms.

Both are future work. The negative result here shows where the constant-coefficient approach breaks down and why parametric generalisation matters.

### What's here

- `src/data/poisson.py` — now includes `assemble_variable_poisson_2d(N, D)` with harmonic face averaging
- `src/data/poisson.py` — `generate_diffusion_coefficient(X, Y, rng, pattern)` for smooth/discontinuous/layered D(x)
- `experiments/run_variable_coeff.py` — trains per-pattern condition-loss models, evaluates CG vs IC(0) vs NN
- `tests/test_variable_poisson.py` — symmetry, positive definiteness, constant-D equivalence

### Tests

118 tests.

```bash
python -m pytest tests/ -v
```
