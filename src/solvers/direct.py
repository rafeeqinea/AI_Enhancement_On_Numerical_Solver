import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve


@dataclass
class DirectResult:
    solution: np.ndarray
    time_seconds: float


def solve_direct(A, b):
    start = time.perf_counter()
    x = spsolve(A, b)
    elapsed = time.perf_counter() - start
    return DirectResult(solution=x, time_seconds=elapsed)
