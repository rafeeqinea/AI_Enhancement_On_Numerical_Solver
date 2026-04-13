from __future__ import annotations

import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

from src.solvers.cg import CGResult
from src.solvers.fcg import flexible_cg
from src.solvers.preconditioners import Preconditioner


def recast_solve(
    A_ref: sp.spmatrix,
    A_new: sp.spmatrix,
    b: np.ndarray,
    precond: Preconditioner,
    tol: float = 1e-6,
    max_outer: int = 50,
    max_inner: int = 200,
    m_max: int = 20,
    omega: float | None = None,
) -> CGResult:
    start = time.perf_counter()
    n = len(b)
    norm_b = np.linalg.norm(b)

    if norm_b < 1e-15:
        return CGResult(
            solution=np.zeros(n), iterations=0, converged=True,
            residual_history=[0.0],
            time_seconds=time.perf_counter() - start,
        )

    delta_A = A_ref - A_new

    if omega is None:
        delta_norm = sp.linalg.norm(delta_A, 'fro')
        ref_norm = sp.linalg.norm(A_ref, 'fro')
        ratio = delta_norm / ref_norm
        omega = min(1.0, 0.8 / max(ratio, 0.1))

    u = np.zeros(n)
    residual_history = []
    total_iters = 0

    for outer in range(max_outer):
        rhs = b + delta_A @ u

        result = flexible_cg(
            A_ref, rhs, precond,
            x0=u, tol=tol * 0.1, max_iter=max_inner, m_max=m_max,
        )
        u_new = result.solution
        total_iters += result.iterations

        u = (1.0 - omega) * u + omega * u_new

        true_residual = b - A_new @ u
        rel_res = np.linalg.norm(true_residual) / norm_b
        residual_history.append(rel_res)

        if rel_res < tol:
            return CGResult(
                solution=u, iterations=total_iters, converged=True,
                residual_history=residual_history,
                time_seconds=time.perf_counter() - start,
            )

    rel_res = np.linalg.norm(b - A_new @ u) / norm_b
    residual_history.append(rel_res)
    return CGResult(
        solution=u, iterations=total_iters,
        converged=(rel_res < tol),
        residual_history=residual_history,
        time_seconds=time.perf_counter() - start,
    )
