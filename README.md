# AI Enhancement of Numerical Solvers

**M34 Final Year Project — BSc Computer Science, University of Greenwich**

Investigating whether machine learning can accelerate iterative PDE solvers
by providing warm-start predictions and learned preconditioners for the
Conjugate Gradient method.

## Branches

| Branch | Description | Status |
|--------|-------------|--------|
| `v0-baseline` | Classical FDM + CG solvers, metrics, experiments (no ML) | Complete |
| `v1-cnn-warmstart` | CNN/U-Net predicts initial guess for CG | Planned |
| `v2-mdpnop-warmstart` | MD-PNOP neural operator warm-start | Planned |
| `v3-npo-preconditioner` | NPO learned preconditioner for CG | Planned |

## Getting Started

Switch to `v0-baseline` to see the current implementation:

```bash
git checkout v0-baseline
```

Each branch uses **tags** (v0.1, v0.2, v0.3, etc.) to show progressive development stages. See the branch README for full details.
