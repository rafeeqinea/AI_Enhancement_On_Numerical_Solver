import time

import numpy as np
import scipy.sparse as sp

from src.solvers.cg import CGResult
from src.solvers.preconditioners import Preconditioner


def flexible_cg(
    A: sp.spmatrix,
    b: np.ndarray,
    precond: Preconditioner,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 10000,
    m_max: int = 20,
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

    P: list[np.ndarray] = []
    S: list[np.ndarray] = []
    residual_history = []

    for i in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b
        residual_history.append(rel_res)

        if rel_res < tol:
            return CGResult(
                solution=x, iterations=i, converged=True,
                residual_history=residual_history,
                time_seconds=time.perf_counter() - start,
            )

        w = precond(r)

        # Re-orthogonalise against only the recent search directions to keep the
        # method flexible without storing the full Krylov history.
        m_i = min(i, max(1, i % (m_max + 1)))
        p = w.copy()
        start_idx = max(0, len(P) - m_i)
        for k in range(start_idx, len(P)):
            ps = P[k] @ S[k]
            if abs(ps) > 1e-30:
                coeff = (w @ S[k]) / ps
                p = p - coeff * P[k]

        s = A @ p
        ps_cur = p @ s
        if abs(ps_cur) < 1e-30:
            break

        alpha = (p @ r) / ps_cur
        x = x + alpha * p
        r = r - alpha * s

        P.append(p)
        S.append(s)

        # Keep a sliding window of directions; this is the restart/limited-memory part.
        if len(P) > m_max + 1:
            P.pop(0)
            S.pop(0)

    rel_res = np.linalg.norm(r) / norm_b
    residual_history.append(rel_res)
    return CGResult(
        solution=x, iterations=len(residual_history) - 1,
        converged=(rel_res < tol),
        residual_history=residual_history,
        time_seconds=time.perf_counter() - start,
    )
