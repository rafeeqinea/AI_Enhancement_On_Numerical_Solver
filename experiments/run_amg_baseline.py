"""AMG baseline evaluation using PyAMG at N=16, 32, 64."""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pyamg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.direct import solve_direct
from src.utils.metrics import compute_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
EVAL_SIZES = [16, 32, 64]
NUM_SAMPLES = 50
SEED = 99
TOL = 1e-6


def run():
    print('=' * 60)
    print('AMG Baseline Evaluation (PyAMG)')
    print('=' * 60)

    all_results = {}

    for N in EVAL_SIZES:
        print(f'\nN={N} (DOF={N*N})')
        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(SEED)

        # Build AMG solver
        ml = pyamg.ruge_stuben_solver(A)

        iters_list = []
        times_list = []
        errors_list = []

        for s in range(NUM_SAMPLES):
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)
            x_true = solve_direct(A, b).solution

            residuals = []
            t0 = time.perf_counter()
            x = ml.solve(b, tol=TOL, maxiter=500,
                        residuals=residuals)
            t1 = time.perf_counter()

            n_iters = len(residuals) - 1
            error = compute_error(x, x_true)

            iters_list.append(n_iters)
            times_list.append(t1 - t0)
            errors_list.append(error)

        result = {
            'N': N,
            'dof': N * N,
            'num_samples': NUM_SAMPLES,
            'method': 'AMG (PyAMG ruge_stuben)',
            'mean_iters': float(np.mean(iters_list)),
            'std_iters': float(np.std(iters_list)),
            'mean_time': float(np.mean(times_list)),
            'mean_error': float(np.mean(errors_list)),
            'max_error': float(np.max(errors_list)),
        }

        all_results[f'N{N}'] = result
        print(f'  AMG: {result["mean_iters"]:.1f} iters '
              f'(std {result["std_iters"]:.2f}), '
              f'time {result["mean_time"]*1000:.1f}ms, '
              f'error {result["mean_error"]:.2e}')

    # Save
    out_path = os.path.join(RESULTS_DIR, 'baseline', 'amg_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    run()
