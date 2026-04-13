from __future__ import annotations

import numpy as np

from src.solvers.cg import CGResult
from src.solvers.direct import DirectResult


def compute_error(approximate: np.ndarray, reference: np.ndarray) -> float:
    ref_norm = np.linalg.norm(reference)
    if ref_norm < 1e-15:
        return 0.0
    return float(np.linalg.norm(approximate - reference) / ref_norm)


def compute_speedup(baseline_time: float, test_time: float) -> float:
    if test_time < 1e-15:
        return float('inf')
    return baseline_time / test_time


def summarize_run(
    N: int,
    cg_result: CGResult,
    direct_result: DirectResult,
) -> dict:
    return {
        'N': N,
        'dof': N * N,
        'cg_iterations': cg_result.iterations,
        'cg_converged': cg_result.converged,
        'cg_time': cg_result.time_seconds,
        'direct_time': direct_result.time_seconds,
        'relative_error': compute_error(cg_result.solution, direct_result.solution),
        'speedup': compute_speedup(direct_result.time_seconds, cg_result.time_seconds),
        'final_residual': cg_result.residual_history[-1] if cg_result.residual_history else None,
    }


def summarize_experiment(runs: list[dict]) -> dict:
    return {
        'num_runs': len(runs),
        'grid_sizes': [r['N'] for r in runs],
        'total_cg_time': sum(r['cg_time'] for r in runs),
        'total_direct_time': sum(r['direct_time'] for r in runs),
        'max_iterations': max(r['cg_iterations'] for r in runs),
        'all_converged': all(r['cg_converged'] for r in runs),
        'runs': runs,
    }
