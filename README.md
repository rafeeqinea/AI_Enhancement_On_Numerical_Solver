# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v4-combined — Combined Preconditioner System

This branch is where everything comes together. Versions 0 through 3 tested each ingredient separately. Version 5 extended to 3D. This branch stacks them into one unified system and tests every meaningful combination.

### What We've Proven So Far (Cases 1-8)

Each case turns on different ingredients so we can isolate what each one contributes.

| Case | Warm-Start | Preconditioner | Loss | Solver | N=32 | N=128 | Reduction | What It Showed |
|------|-----------|----------------|------|--------|------|-------|-----------|----------------|
| 1 | No | None | — | CG | 81 | 324 | baseline | Iterations grow as O(N). The problem. |
| 2 | Yes | None | — | CG | 75 | — | 7-9% | Warm-start has a ceiling. Better start, same speed. |
| 3 | No | Jacobi | — | PCG | 81 | 324 | 0% | Useless for constant-diagonal Poisson. |
| 4 | No | IC(0) | — | PCG | 16 | ~130 | 61-80% | Strong classical baseline. Industry standard. |
| 5 | Yes | IC(0) | — | PCG | 16 | ~130 | 61-80% | Warm-start adds nothing on top of IC(0). |
| 6 | No | U-Net | MSE | FCG | FAIL | FAIL | FAILS | Spectral bias. Only learns low-frequency corrections. |
| 7 | No | U-Net | Condition | FCG | 12 | 20 | 85-94% | Beats IC(0) by 2x. Nearly O(1) scaling. |
| 8 | Yes | U-Net | Condition | FCG | 7 | ~20 | 86-94% | Marginal extra from warm-start. |

The key finding: the loss function matters more than anything else. Same U-Net, same solver, but condition loss succeeds where MSE fails. This comparison has not been published (confirmed across 73 papers).

### New Cases: The Combinations (To Be Implemented)

| Case | Warm-Start | Preconditioner | Recast | Target Problem | Expected Result | Status |
|------|-----------|----------------|--------|---------------|-----------------|--------|
| 9 | Yes | IC(0) + U-Net | No | Constant coeff | 10-15 iters (95-97%) | TODO |
| 10 | No | IC(0) + U-Net | No | Constant coeff | 10-15 iters (95-97%) | TODO |
| 11 | No | U-Net | Yes | Variable coeff | Better than v3 | TODO |
| 12 | No | IC(0) + U-Net | Yes | Variable coeff | IC(0) base + NN correction | TODO |
| 13 | Yes | U-Net | Yes | Variable coeff | WS + recast + NN | TODO |
| 14 | Yes | IC(0) + U-Net | Yes | Variable coeff | Full stack, all ingredients | TODO |

### Case 9: IC(0) + U-Net + Warm-Start (Constant Coefficients)

The full stack for constant-coefficient Poisson. This is the original plan from the start of the project: test each piece separately, verify it works, then combine everything.

**How it works at each FCG iteration:**
1. Compute residual r (how wrong is the current guess)
2. Apply IC(0) to r → get z1 (the classical correction — what math alone can do)
3. Compute leftover: r2 = r - A × z1 (what IC(0) missed)
4. Apply U-Net to r2 → get z2 (the learned correction — what the NN picks up)
5. Combined correction: z = z1 + z2 (classical + learned, fed into FCG)

The U-Net's job is easier here than in Case 7. In Case 7 it had to learn all of A⁻¹ by itself. Here IC(0) handles ~65% of the work. The U-Net only learns the remaining ~35%.

### Case 10: IC(0) + U-Net, No Warm-Start (Constant Coefficients)

Same as Case 9 but without warm-start. Comparing Case 9 vs Case 10 tells us whether warm-start adds anything on top of the combined IC(0)+NN preconditioner.

Based on Cases 5 and 8, the answer is probably no. But we test it to be sure.

### Case 11: Equation Recast + U-Net (Variable Coefficients)

The NN trained on constant D=1 failed on variable D(x) in v3 because it never sees the coefficient field. Equation recast (from MD-PNOP, Cheng et al. 2025) splits the problem:

- A(D) = A(D0) + delta_A, where D0 = 1 is the reference the NN knows
- Inner solve: use the NN preconditioner on the known A(D0)
- Outer iteration: handle the perturbation delta_A with relaxation

v3 showed this works for small perturbations (~3:1 contrast) but diverges for large (~67:1). A purpose-trained model for the recast framework might handle larger contrasts.

### Case 12: Equation Recast + IC(0) + U-Net (Variable Coefficients, Full Stack)

IC(0) adapts to variable coefficients automatically because it recomputes from each matrix. In v3, IC(0) alone got 77-93% reduction on variable coefficients. Adding the NN on top should push further.

### Case 13: Warm-Start + Equation Recast + U-Net (Variable Coefficients)

For the recast framework, warm-start could help the outer iteration converge faster. Each outer step solves a full inner problem. If warm-start gives a good initial guess for each inner solve, the total cost drops.

### Case 14: Everything Combined (The Final Product)

All three techniques from the literature stacked:
- NOWS (Eshaghi 2025) → warm-start
- NPO (Li 2025) → condition loss via Hutchinson
- MD-PNOP (Cheng 2025) → equation recast for variable coefficients

Combined into one pipeline. This is the final product.

### Branch Structure

```
v0-rebuild (baseline CG)
 └── v1-warmstart-rebuild (warm-start)
      └── v2-preconditioner (8-case factorial)
           ├── v3-variable-coefficient (variable D(x) + equation recast)
           ├── v4-combined (THIS BRANCH — combined system, Cases 9-14)
           └── v5-3d (3D extension, 82.3% reduction at N=32)
```

### Implementation Priority

1. Case 9 + 10 (constant coeff combined) — low effort, high impact
2. Case 11 + 12 (variable coeff with recast) — medium effort, high impact
3. Cases 13 + 14 (add warm-start) — low effort, low impact (WS is always marginal)

### Tests

123 tests (inherited from v3).

```bash
python -m pytest tests/ -v
```
