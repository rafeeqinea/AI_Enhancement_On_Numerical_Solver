from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.direct import solve_direct
from src.solvers.preconditioners import Preconditioner
from src.utils.metrics import compute_error


def evaluate_preconditioner(
    precond_name: str,
    precond_fn: Preconditioner,
    N: int,
    num_samples: int = 50,
    tol: float = 1e-6,
    seed: int = 99,
    use_fcg: bool = False,
    m_max: int = 20,
    x0_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)

    cold_iters: list[int] = []
    precond_iters: list[int] = []
    cold_times: list[float] = []
    precond_times: list[float] = []
    errors: list[float] = []

    solver = flexible_cg if use_fcg else preconditioned_cg

    for _ in range(num_samples):
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)
        direct = solve_direct(A, b)

        cold = conjugate_gradient(A, b, tol=tol)
        cold_iters.append(cold.iterations)
        cold_times.append(cold.time_seconds)

        x0 = x0_fn(f, N) if x0_fn is not None else None

        kwargs: dict[str, Any] = {'tol': tol}
        if use_fcg:
            kwargs['m_max'] = m_max

        result = solver(A, b, precond_fn, x0=x0, **kwargs)
        precond_iters.append(result.iterations)
        precond_times.append(result.time_seconds)
        errors.append(compute_error(result.solution, direct.solution))

    return {
        'precond_name': precond_name,
        'N': N,
        'num_samples': num_samples,
        'cold_iters_mean': float(np.mean(cold_iters)),
        'cold_iters_std': float(np.std(cold_iters)),
        'precond_iters_mean': float(np.mean(precond_iters)),
        'precond_iters_std': float(np.std(precond_iters)),
        'iteration_reduction': float(1 - np.mean(precond_iters) / np.mean(cold_iters)),
        'cold_time_mean': float(np.mean(cold_times)),
        'precond_time_mean': float(np.mean(precond_times)),
        'speedup': float(np.mean(cold_times) / max(np.mean(precond_times), 1e-15)),
        'mean_error': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
    }
