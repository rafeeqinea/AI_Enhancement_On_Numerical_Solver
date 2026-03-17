from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.solvers.preconditioners import Preconditioner


def make_nn_preconditioner(
    model: nn.Module,
    N: int,
    res_mean: float = 0.0,
    res_std: float = 1.0,
    err_mean: float = 0.0,
    err_std: float = 1.0,
    device: torch.device | None = None,
) -> Preconditioner:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    def apply(r: np.ndarray) -> np.ndarray:
        r_grid = r.reshape(N, N)

        normed = (r_grid - res_mean) / res_std

        padded = np.zeros((N + 2, N + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = normed
        x_in = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_in)

        pred_np = pred.squeeze().cpu().numpy()
        denormed = pred_np * err_std + err_mean

        return denormed.ravel()

    return apply
