# AI Enhancement of Numerical Solvers

BSc Final Year Project — University of Greenwich, 2025-26

Supervised by Dr Peter Soar

## v1 — Warm-start with neural networks

This branch adds neural network warm-starting on top of the v0 baseline. A CNN and U-Net are trained to predict the Poisson solution from the source term, and CG starts from that prediction instead of zero.

### The idea

CG converges from whatever starting point you give it. If the starting point is already close to the answer, CG should need fewer iterations to reach the tolerance. A neural network that has seen thousands of (source, solution) pairs should be able to predict a reasonable starting guess.

### What happened

It didn't work. At grid sizes N=16 to N=64, warm-starting with the neural network prediction made CG take MORE iterations, not fewer.

| N | CG (cold) | CNN warm-start | U-Net warm-start |
|---|-----------|---------------|-----------------|
| 16 | 41 | 44 (+7.9%) | 44 (+7.3%) |
| 32 | 81 | 90 (+10.8%) | 87 (+7.4%) |
| 64 | 167 | 186 (+11.6%) | 180 (+7.9%) |

The CNN generalised well (val_loss=0.081, gap=1.1x) but warm-start still hurt. The U-Net overfit badly (val_loss=0.85, gap=227x).

### Why it failed

The NOWS paper (Eshaghi 2025) reports the same pattern: at N=64, their warm-start only gets 23.7% speedup even with a spectral neural operator. Our negative results are consistent with the literature for small grids. The reasons:

1. CG convergence depends on the A-norm error, not pixel MSE. A prediction with low MSE can have high A-norm error if it gets high-frequency modes wrong.
2. At small N, CG only needs 40-167 iterations. The warm-start savings are tiny in absolute terms.
3. The inference cost of running the U-Net eats whatever iteration savings exist.

This negative result motivates the preconditioner approach in v2, where we change the convergence rate itself rather than just the starting point.

### What's here

- `src/models/cnn.py` — 7-layer CNN baseline (~195K params)
- `src/models/unet.py` — U-Net with configurable depth/width
- `src/data/dataset.py` — PyTorch dataset with normalisation and padding
- `src/training/train.py` — Training pipeline with early stopping, LR scheduling, snapshots
- `src/evaluation/evaluate.py` — Cold vs warm CG comparison
- `experiments/run_warmstart.py` — Full experiment runner

### Tests

69 tests.

```bash
python -m pytest tests/ -v
```
