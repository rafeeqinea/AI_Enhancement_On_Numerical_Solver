from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.direct import solve_direct
from src.utils.metrics import compute_error


def predict_warmstart(
    model: nn.Module,
    source: np.ndarray,
    N: int,
    source_mean: float = 0.0,
    source_std: float = 1.0,
    sol_mean: float = 0.0,
    sol_std: float = 1.0,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

    normed_source = (source - source_mean) / source_std

    padded = np.zeros((N + 2, N + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = normed_source
    x_in = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x_in)

    pred_np = pred.squeeze().cpu().numpy()
    denormed = pred_np * sol_std + sol_mean

    return denormed.ravel()


def evaluate_warmstart(
    model: nn.Module,
    N: int,
    norm_stats: dict | None = None,
    num_samples: int = 50,
    tol: float = 1e-6,
    seed: int = 99,
    device: torch.device | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)

    if norm_stats is None:
        norm_stats = {'source_mean': 0.0, 'source_std': 1.0,
                      'sol_mean': 0.0, 'sol_std': 1.0}

    cold_iters = []
    warm_iters = []
    cold_times = []
    warm_times = []
    errors = []

    for _ in range(num_samples):
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        direct = solve_direct(A, b)

        cold = conjugate_gradient(A, b, tol=tol)
        cold_iters.append(cold.iterations)
        cold_times.append(cold.time_seconds)

        x0 = predict_warmstart(
            model, f, N,
            source_mean=norm_stats['source_mean'],
            source_std=norm_stats['source_std'],
            sol_mean=norm_stats['sol_mean'],
            sol_std=norm_stats['sol_std'],
            device=device,
        )
        warm = conjugate_gradient(A, b, x0=x0, tol=tol)
        warm_iters.append(warm.iterations)
        warm_times.append(warm.time_seconds)

        errors.append(compute_error(warm.solution, direct.solution))

    return {
        'N': N,
        'num_samples': num_samples,
        'cold_iters_mean': float(np.mean(cold_iters)),
        'cold_iters_std': float(np.std(cold_iters)),
        'warm_iters_mean': float(np.mean(warm_iters)),
        'warm_iters_std': float(np.std(warm_iters)),
        'iteration_reduction': float(1 - np.mean(warm_iters) / np.mean(cold_iters)),
        'cold_time_mean': float(np.mean(cold_times)),
        'warm_time_mean': float(np.mean(warm_times)),
        'speedup': float(np.mean(cold_times) / max(np.mean(warm_times), 1e-15)),
        'mean_error': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
    }
