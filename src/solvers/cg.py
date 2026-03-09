import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CGResult:
    solution: np.ndarray
    iterations: int
    converged: bool
    residual_history: list = field(default_factory=list)
    time_seconds: float = 0.0


def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=10000):
    start = time.perf_counter()
    n = len(b)

    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A @ x
    d = r.copy()
    delta_new = r @ r
    norm_b = np.linalg.norm(b)

    if norm_b < 1e-15:
        return CGResult(
            solution=x, iterations=0, converged=True,
            residual_history=[0.0],
            time_seconds=time.perf_counter() - start
        )

    residual_history = []

    for k in range(max_iter):
        rel_res = math.sqrt(delta_new) / norm_b
        residual_history.append(rel_res)

        if rel_res < tol:
            return CGResult(
                solution=x, iterations=k, converged=True,
                residual_history=residual_history,
                time_seconds=time.perf_counter() - start
            )

        q = A @ d
        alpha = delta_new / (d @ q)
        x = x + alpha * d
        r = r - alpha * q
        delta_old = delta_new
        delta_new = r @ r
        beta = delta_new / delta_old
        d = r + beta * d

    rel_res = math.sqrt(delta_new) / norm_b
    residual_history.append(rel_res)
    return CGResult(
        solution=x, iterations=max_iter, converged=False,
        residual_history=residual_history,
        time_seconds=time.perf_counter() - start
    )
