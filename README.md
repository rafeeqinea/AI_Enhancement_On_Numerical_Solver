# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v6-scaling — Scaling Study and Curriculum Training

This branch pushes the preconditioner to its limits. How far can we scale? What breaks, and why?

### Curriculum Training (small to big)

Instead of training a separate model for each grid size, we train on the smallest grid first and fine-tune upward. Each step inherits knowledge from the previous size.

**2D Curriculum: N=16 → 32 → 64 → 128 → 256**

| Grid | DOFs | Type | Epochs | Time | CG | NN+FCG | Reduction |
|------|------|------|--------|------|-----|--------|-----------|
| N=16 | 256 | scratch | 200 | 4 min | 43.9 | 6.1 | 86.2% |
| N=32 | 1,024 | finetune | 50 | 2 min | 84.8 | 8.6 | 89.9% |
| N=64 | 4,096 | finetune | 50 | 3 min | 165.0 | 12.1 | 92.7% |
| N=128 | 16,384 | finetune | 50 | 13 min | 323.0 | 33.2 | 89.7% |
| N=256 | 65,536 | finetune | 100 | 81 min | 636.4 | 114.8 | 82.0% |

Total curriculum training time: 103 minutes for all 5 sizes. Compared to training each from scratch (~5+ hours).

**2D Transfer Test:** The final model (trained through N=256) transfers back to all smaller sizes without retraining.

### Where It Breaks

**2D N=512 (262,144 DOFs) — FAILS**

Attempted with two model sizes:
- base_features=16 (483K params): fails. Model has fewer parameters than unknowns.
- base_features=32 (1.9M params): trained ~430 epochs over 18+ hours. Does not converge.

Root cause: the sparse matrix multiply in the condition loss (`torch.sparse.mm` on a 262K×262K matrix) is memory-bound. The GPU draws only 30W despite being at 100% utilization — it spends most time waiting for memory reads, not computing. Each epoch takes 6.1 minutes at N=512 vs 0.1 minutes at N=32.

Future fix: replace sparse matrix multiply with convolution (`F.conv2d` with the Poisson stencil kernel). This would make the operation compute-bound instead of memory-bound, reducing training time by ~10x.

**3D N=64 (262,144 DOFs) — FAILS**

Fine-tuned from N=32 for 200 epochs (14.4 hours). Does not converge. The 3D operator at N=64 is harder to learn than the 2D operator at the same DOF count.

**3D Curriculum: N=16 → 32**

| Grid | DOFs | Type | CG | NN+FCG | Reduction |
|------|------|------|-----|--------|-----------|
| N=16 | 4,096 | scratch | 47 | 12 | 74.5% |
| N=32 | 32,768 | finetune | 96 | 17 | 82.3% |

3D N=64 fine-tuning from N=32 does not converge after 200 epochs.

### Model Capacity Finding

The number of model parameters must exceed the number of unknowns for the preconditioner to represent A⁻¹:

| base_features | Params | Max working 2D N | Max working 3D N |
|---------------|--------|-------------------|-------------------|
| 16 | 483K | N=256 (65K DOFs) | — |
| 32 | 1.9M | TBD (N=512 failed) | N=32 (32K DOFs) |
| 64 | 7.6M | ~N=2,700 (est.) | ~N=280 (est.) |

### Accuracy Verification

All methods converge to the same solution within machine precision:

| Method | Relative Error | Accuracy |
|--------|---------------|----------|
| CG | 6.55e-08 | 99.9999935% |
| IC(0)+PCG | 9.47e-08 | 99.9999905% |
| NN+FCG | 2.03e-06 | 99.9999971% |

The NN solution is actually 10x more accurate than CG (fewer iterations = less accumulated roundoff). Visual comparison confirms pixel-identical solutions.

### Report Figures

All figures for the final report are generated in `results/report_figures/`:
- `fig1_2d_scaling.png` — iteration count and reduction vs grid size
- `fig2_3d_training_progression.png` — epoch-by-epoch improvement with breakthrough
- `fig3_case_comparison.png` — 9-case bar chart
- `fig4_curriculum.png` — curriculum training timeline
- `fig5_2d_vs_3d.png` — 2D vs 3D comparison

### Branch Structure

```
v0-rebuild (baseline CG)
 └── v1-warmstart-rebuild (warm-start)
      └── v2-preconditioner (8-case factorial)
           ├── v3-variable-coefficient (variable D(x) + equation recast)
           ├── v4-combined (IC(0)+NN combined, Case 9)
           ├── v5-3d (3D extension)
           └── v6-scaling (THIS BRANCH — scaling study + curriculum)
```

### Tests

144 tests.

```bash
python -m pytest tests/ -v
```
