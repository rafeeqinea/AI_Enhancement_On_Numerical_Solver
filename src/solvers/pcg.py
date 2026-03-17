import time

import numpy as np

from src.solvers.cg import CGResult
from src.solvers.preconditioners import Preconditioner


def preconditioned_cg(
    A,
    b: np.ndarray,
    precond: Preconditioner,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> CGResult:
    start = time.perf_counter()
    n = len(b)

    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A @ x
    norm_b = np.linalg.norm(b)

    if norm_b < 1e-15:
        return CGResult(
            solution=x, iterations=0, converged=True,
            residual_history=[0.0],
            time_seconds=time.perf_counter() - start,
        )

    z = precond(r)
    p = z.copy()
    rz = r @ z
    residual_history = []

    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b
        residual_history.append(rel_res)

        if rel_res < tol:
            return CGResult(
                solution=x, iterations=k, converged=True,
                residual_history=residual_history,
                time_seconds=time.perf_counter() - start,
            )

        q = A @ p
        alpha = rz / (p @ q)
        x = x + alpha * p
        r = r - alpha * q
        z = precond(r)
        rz_new = r @ z
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    rel_res = np.linalg.norm(r) / norm_b
    residual_history.append(rel_res)
    return CGResult(
        solution=x, iterations=max_iter, converged=False,
        residual_history=residual_history,
        time_seconds=time.perf_counter() - start,
    )
