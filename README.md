# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v5-3d — 3D Poisson Extension

This branch extends the entire pipeline from 2D to 3D. The Poisson equation -∇²u = f is now solved on the unit cube [0,1]³ with a 7-point stencil and N³ unknowns.

### What Changed from 2D

The 2D version uses a 5-point stencil on an N×N grid (N² unknowns). The 3D version uses a 7-point stencil on an N×N×N grid (N³ unknowns). At N=32, that's 32,768 unknowns instead of 1,024.

**Dimension-generic U-Net:** The same `unet.py` file handles both 2D and 3D. Pass `dim=2` for Conv2d or `dim=3` for Conv3d. Same architecture, same skip connections, just more spatial dimensions. The 3D model has 5.6M parameters (vs 483K for 2D) because Conv3d kernels are 3×3×3 = 27 weights vs Conv2d's 3×3 = 9.

**Structured IC(0):** The dense IC(0) from v2 would need 32768×32768 = 8 GB for N=32 in 3D. A new diagonal-based implementation exploits the known stencil structure to compute IC(0) in O(n) time and O(n) memory.

**GPU optimisations:** Condition loss training uses AMP (mixed precision), GradScaler (prevents float16 overflow), gradient checkpointing (reduces VRAM), and probe accumulation (process probes in batches). All four were needed to fit N=128 3D (2M DOFs) in 8.6 GB VRAM.

### 3D Results at N=32 (32,768 DOFs)

| Case | Method | Iterations | Reduction |
|------|--------|-----------|-----------|
| 1 | CG (baseline) | 96 | — |
| 4 | IC(0) + PCG | 34 | 64.6% |
| 7 | U-Net (condition loss) + FCG | 17 | 82.3% |

The condition loss U-Net preconditioner beats IC(0) by 2x in 3D, consistent with the 2D results. The approach extends to higher dimensions.

### Training Details

- Model: UNet(base_features=32, levels=3, dim=3), 5.6M parameters
- Loss: Condition loss ||I - AM||²_F via Hutchinson with 128 random probes
- Training: 300 epochs × 100 steps/epoch, Adam with cosine annealing
- GPU: RTX 4060 Laptop (8.6 GB VRAM), ~5 hours
- Checkpoints saved every 50 epochs (epoch_0050 through epoch_0300)

### Training Progression

The model improved throughout training, with a breakthrough between epoch 150 and 200:

```
Epoch  50:  1000 iters  (model not ready — random garbage)
Epoch 100:    38 iters  (60.4% reduction — working, close to IC(0))
Epoch 150:    38 iters  (plateaued)
Epoch 200:    28 iters  (70.8% — broke through, beats IC(0))
Epoch 250:    19 iters  (80.2% — still improving)
Epoch 300:    17 iters  (82.3% — final result)
```

### 3D Visualisations

Solution slices through the centre of the cube and an interactive 3D view:

- `results/3d/slice_xy.png` — XY plane at z=0.5
- `results/3d/slice_xz.png` — XZ plane at y=0.5
- `results/3d/slice_yz.png` — YZ plane at x=0.5
- `results/3d/v4_3d_results_N32.png` — combined results (slices + convergence curves)
- `results/3d/v4_3d_interactive.html` — interactive 3D isosurface (open in browser, rotate/zoom)

### What's Here

- `src/models/unet.py` — dimension-generic U-Net (dim=2 or dim=3)
- `src/data/poisson.py` — 3D Poisson assembly, grid points, RHS
- `src/data/generate.py` — 3D source term generation
- `src/training/losses.py` — condition loss with AMP, GradScaler, checkpointing, probe batching
- `src/evaluation/nn_preconditioner.py` — 3D preconditioner wrapper with unit-norm scaling
- `src/solvers/preconditioners.py` — structured IC(0) for 3D (diagonal-based, O(n))
- `experiments/run_3d.py` — train and evaluate 3D experiments
- `tests/test_poisson_3d.py` — 21 new tests for 3D

### Branch Structure

```
v0-rebuild (baseline CG)
 └── v1-warmstart-rebuild (warm-start)
      └── v2-preconditioner (8-case factorial)
           ├── v3-variable-coefficient (variable D(x) + equation recast)
           ├── v4-combined (combined system plan, Cases 9-14)
           └── v5-3d (THIS BRANCH — 3D extension)
```

### Tests

144 tests (123 from v3 + 21 new 3D tests).

```bash
python -m pytest tests/ -v
```
