from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.direct import solve_direct
from src.utils.metrics import compute_error, compute_speedup, summarize_run, summarize_experiment
from src.utils.visualize import plot_convergence, plot_solution, plot_scaling, plot_comparison_bar


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline')
GRID_SIZES = [8, 16, 32, 64, 128]
NUM_SOURCES = 5
SEED = 42
TOL = 1e-6


def run_single(N: int, rng: np.random.Generator) -> dict:
    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)

    cg_iters_list = []
    cg_times_list = []
    direct_times_list = []
    errors_list = []

    last_cg = None
    last_direct = None

    for _ in range(NUM_SOURCES):
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        direct_result = solve_direct(A, b)
        cg_result = conjugate_gradient(A, b, tol=TOL)

        cg_iters_list.append(cg_result.iterations)
        cg_times_list.append(cg_result.time_seconds)
        direct_times_list.append(direct_result.time_seconds)
        errors_list.append(compute_error(cg_result.solution, direct_result.solution))

        last_cg = cg_result
        last_direct = direct_result

    return {
        'N': N,
        'dof': N * N,
        'num_sources': NUM_SOURCES,
        'mean_cg_iterations': float(np.mean(cg_iters_list)),
        'std_cg_iterations': float(np.std(cg_iters_list)),
        'mean_cg_time': float(np.mean(cg_times_list)),
        'mean_direct_time': float(np.mean(direct_times_list)),
        'mean_relative_error': float(np.mean(errors_list)),
        'max_relative_error': float(np.max(errors_list)),
        'speedup': float(np.mean(direct_times_list)) / max(float(np.mean(cg_times_list)), 1e-15),
        'last_cg_result': last_cg,
        'last_direct_result': last_direct,
    }


def run_experiment() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f'Running baseline experiment: N in {GRID_SIZES}, {NUM_SOURCES} sources each')
    print(f'Tolerance: {TOL}, Seed: {SEED}')
    print('=' * 60)

    runs = []
    grid_sizes = []
    mean_iterations = []

    for N in GRID_SIZES:
        print(f'\nN = {N:>4d}  (dof = {N*N:>6d}) ...', end=' ', flush=True)
        result = run_single(N, rng)

        print(f'CG: {result["mean_cg_iterations"]:.0f} iters (±{result["std_cg_iterations"]:.1f}), '
              f'error: {result["mean_relative_error"]:.2e}, '
              f'speedup: {result["speedup"]:.2f}x')

        grid_sizes.append(N)
        mean_iterations.append(result['mean_cg_iterations'])

        plot_convergence(
            result['last_cg_result'],
            title=f'CG Convergence (N={N})',
            save_path=os.path.join(RESULTS_DIR, f'convergence_N{N}.png'),
        )

        plot_solution(
            result['last_cg_result'].solution, N,
            title=f'CG Solution (N={N})',
            save_path=os.path.join(RESULTS_DIR, f'solution_N{N}.png'),
        )

        serialisable = {k: v for k, v in result.items()
                        if k not in ('last_cg_result', 'last_direct_result')}
        runs.append(serialisable)

    print('\n' + '=' * 60)

    plot_scaling(
        grid_sizes,
        [int(round(m)) for m in mean_iterations],
        title='CG Iteration Scaling (Random Sources)',
        save_path=os.path.join(RESULTS_DIR, 'scaling.png'),
    )

    cg_times = [r['mean_cg_time'] for r in runs]
    direct_times = [r['mean_direct_time'] for r in runs]
    labels = [f'N={r["N"]}' for r in runs]
    plot_comparison_bar(
        labels, cg_times,
        title='CG Solve Time by Grid Size',
        save_path=os.path.join(RESULTS_DIR, 'timing_cg.png'),
    )
    plot_comparison_bar(
        labels, direct_times,
        title='Direct Solve Time by Grid Size',
        save_path=os.path.join(RESULTS_DIR, 'timing_direct.png'),
    )

    summary = {
        'experiment': 'baseline_v0',
        'tolerance': TOL,
        'seed': SEED,
        'num_sources_per_N': NUM_SOURCES,
        'runs': runs,
    }

    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as fp:
        json.dump(summary, fp, indent=2)

    print(f'\nResults saved to {results_path}')
    print(f'Figures saved to {RESULTS_DIR}/')

    return summary


if __name__ == '__main__':
    run_experiment()
