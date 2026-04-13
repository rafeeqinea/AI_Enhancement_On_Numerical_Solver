# FYP Results Summary

## Constant-coefficient Poisson (v0-v2)

8-case factorial experiment on 2D Poisson with 5-point FDM stencil.

| Case | x0 | Preconditioner | Loss | Solver | N=16 | N=32 | N=64 | N=128 | Reduction |
|------|-----|---------------|------|--------|------|------|------|-------|-----------|
| 1 | Zero | None | — | CG | 41 | 81 | 162 | 324 | baseline |
| 2 | U-Net | None | MSE | CG | 44 | 87 | 176 | — | -7 to -9% |
| 3 | Zero | Jacobi | — | PCG | 41 | 81 | 162 | 324 | 0% |
| 4 | Zero | IC(0) | — | PCG | 16 | 28 | 54 | 105 | 61-67% |
| 5 | U-Net | IC(0) | MSE | PCG | 16 | 30 | 56 | — | 60-65% |
| 6 | Zero | U-Net | MSE | FCG | >1000 | >1000 | >1000 | — | FAIL |
| 7 | Zero | U-Net | Condition | FCG | 9 | 13 | 27 | 20 | 78-94% |
| 8 | U-Net | U-Net | Condition | FCG | 9 | 11 | 45 | — | 72-86% |

### Key findings

Case 2: Warm-start alone hurts at small grids. MSE-trained prediction has high A-norm error.

Case 3: Jacobi does nothing for constant-diagonal Poisson. Just divides by 4.

Case 4: IC(0) is the real classical baseline. 61-67% reduction.

Case 5: Warm-start adds nothing to IC(0). The warm-start model introduces error that IC(0) has to fight.

Case 6: MSE-trained preconditioner fails completely. Spectral bias — the U-Net only learns low-frequency residual mappings. After CG removes low-freq content, the U-Net outputs near-constant values.

Case 7: Condition loss (Hutchinson ||I-AM||^2_F) achieves 78-94% reduction. At N=128: 324 iterations down to 20. The NN iteration count stays nearly flat with grid size while CG grows linearly and IC(0) grows as sqrt(N).

Case 8: Combining warm-start with condition preconditioner helps at N=32 (11 vs 13) but hurts at N=64 (45 vs 27). The v1 warm-start model quality is the bottleneck.

### Novel contribution

The Case 6 vs Case 7 comparison (MSE vs condition loss on the same architecture, same problem) has not been published. The result is dramatic: FAIL vs 78-94% reduction.

## Variable-coefficient Poisson (v3)

-div(D(x) grad u) = f with three coefficient patterns.

### v3.1: Naive application (trained on D=1, tested on D(x))

| Pattern | D range | CG | IC(0)+PCG | NN+FCG |
|---------|---------|-----|-----------|--------|
| Smooth | 1.4-5.0 | 130 | 30 (77%) | >1000 (FAIL) |
| Discontinuous | 1.0-67 | 491 | 36 (93%) | >1000 (FAIL) |
| Layered | 0.2-6.9 | 141 | 29 (80%) | >1000 (FAIL) |

NN fails because it only sees the residual, not D(x). IC(0) adapts per-problem.

### v3.2: MD-PNOP equation recast

Decompose A(D) = A(D0) + perturbation. Solve iteratively using the reference preconditioner.

| Pattern | Contrast | CG | IC(0) direct | Recast+IC(0) | Recast+NN |
|---------|----------|-----|-------------|-------------|-----------|
| Layered | ~3:1 | 103 | 29 | 266 | 106 |
| Smooth | ~5:1 | 117 | 29 | diverges | diverges |
| Discontinuous | ~67:1 | 362 | 34 | diverges | diverges |

Equation recast works for small perturbations. NN is 2.5x faster than IC(0) inside the recast (layered case). Diverges when the coefficient contrast exceeds ~3:1.

## Test coverage

161 tests across 22 test files. All passing.

## Architecture

- U-Net preconditioner: base_features=16, levels=3, ~483K params
- Warm-start U-Net: base_features=16, levels=2
- Condition loss: 32 Hutchinson probes, 100 gradient steps/epoch
- FCG: m_max=20, Algorithm 1 from FCG-NO (Rudikov 2024)
- IC(0): manual incomplete Cholesky (scipy spilu was unreliable)

## Papers referenced

- NOWS (Eshaghi 2025) — warm-start for CG
- NPO (Li 2025) — condition loss via Hutchinson
- FCG-NO (Rudikov 2024) — flexible CG with neural preconditioner
- Azulay & Treister (2022) — U-Net as V-cycle
- MD-PNOP (Cheng 2025) — equation recast for parametric generalisation
