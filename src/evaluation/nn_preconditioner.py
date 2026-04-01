from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.solvers.preconditioners import Preconditioner
import scipy.sparse as sp


def make_nn_preconditioner(
    model: nn.Module,
    N: int,
    device: torch.device | None = None,
    dim: int = 2,
) -> Preconditioner:
    """Wrap a trained U-Net as a Callable preconditioner for FCG.

    Works for both 2D (dim=2) and 3D (dim=3).
    Uses unit-norm scaling: r_unit = r / ||r||, output scaled back by ||r||.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    grid_shape = (N,) * dim
    padded_shape = (N + 2,) * dim

    def apply(r: np.ndarray) -> np.ndarray:
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-30:
            return np.zeros_like(r)

        r_unit = r / r_norm
        r_grid = r_unit.reshape(grid_shape)

        padded = np.zeros(padded_shape, dtype=np.float32)
        interior = tuple(slice(1, -1) for _ in range(dim))
        padded[interior] = r_grid

        x_in = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_in)

        pred_np = pred.squeeze().cpu().numpy()
        return pred_np.ravel() * r_norm

    return apply


def make_composite_preconditioner(
    model: nn.Module,
    A: sp.spmatrix,
    N: int,
    device: torch.device | None = None,
    jacobi_sweeps: int = 2,
    dim: int = 2,
) -> Preconditioner:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    diag = A.diagonal().copy()
    diag[diag == 0] = 1.0
    inv_diag = 1.0 / diag
    omega = 2.0 / 3.0
    grid_shape = (N,) * dim
    padded_shape = (N + 2,) * dim

    def jacobi_smooth(x: np.ndarray, r: np.ndarray) -> np.ndarray:
        for _ in range(jacobi_sweeps):
            x = x + omega * inv_diag * (r - A @ x)
        return x

    def nn_correction(r: np.ndarray) -> np.ndarray:
        r_grid = r.reshape(grid_shape)
        padded = np.zeros(padded_shape, dtype=np.float32)
        interior = tuple(slice(1, -1) for _ in range(dim))
        padded[interior] = r_grid
        x_in = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_in)
        return pred.squeeze().cpu().numpy().ravel()

    def apply(r: np.ndarray) -> np.ndarray:
        e = np.zeros_like(r)
        e = jacobi_smooth(e, r)
        r_after = r - A @ e
        e_nn = nn_correction(r_after)
        e = e + e_nn
        e = jacobi_smooth(e, r)
        return e

    return apply
