from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import assemble_poisson_2d
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.evaluation.evaluate_precond import evaluate_preconditioner
from src.utils.visualize import plot_convergence, plot_comparison_bar

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'preconditioned')
EVAL_SIZES = [16, 32, 64]
NUM_SAMPLES = 50
SEED = 99
TOL = 1e-6


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for N in EVAL_SIZES:
        print(f'\n{"="*50}')
        print(f'Grid N={N} (DOF={N*N})')
        print(f'{"="*50}')

        A = assemble_poisson_2d(N)

        jac_precond = jacobi_preconditioner(A)
        ic0_precond = ic0_preconditioner(A)

        print(f'\nCase 3: Jacobi + PCG ...')
        jac_result = evaluate_preconditioner(
            'Jacobi', jac_precond, N,
            num_samples=NUM_SAMPLES, tol=TOL, seed=SEED,
        )
        print(f'  CG baseline: {jac_result["cold_iters_mean"]:.1f} iters')
        print(f'  Jacobi+PCG:  {jac_result["precond_iters_mean"]:.1f} iters')
        print(f'  Reduction:   {jac_result["iteration_reduction"]*100:.1f}%')

        print(f'\nCase 4: IC(0) + PCG ...')
        ic0_result = evaluate_preconditioner(
            'IC(0)', ic0_precond, N,
            num_samples=NUM_SAMPLES, tol=TOL, seed=SEED,
        )
        print(f'  CG baseline: {ic0_result["cold_iters_mean"]:.1f} iters')
        print(f'  IC(0)+PCG:   {ic0_result["precond_iters_mean"]:.1f} iters')
        print(f'  Reduction:   {ic0_result["iteration_reduction"]*100:.1f}%')

        all_results[f'N{N}'] = {
            'jacobi': jac_result,
            'ic0': ic0_result,
        }

        labels = ['CG (Case 1)', 'Jacobi+PCG (Case 3)', 'IC(0)+PCG (Case 4)']
        iters = [
            jac_result['cold_iters_mean'],
            jac_result['precond_iters_mean'],
            ic0_result['precond_iters_mean'],
        ]
        fig_path = os.path.join(RESULTS_DIR, f'comparison_N{N}.png')
        plot_comparison_bar(labels, iters, title=f'Iteration Comparison N={N}', save_path=fig_path)
        print(f'  Saved: {fig_path}')

    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nAll results saved to {results_path}')

    print('\n' + '='*60)
    print('SUMMARY: Cases 3-4 Iteration Counts')
    print('='*60)
    print(f'{"N":>5} {"CG":>8} {"Jacobi":>10} {"IC(0)":>10} {"Jac %":>8} {"IC0 %":>8}')
    print('-'*60)
    for N in EVAL_SIZES:
        r = all_results[f'N{N}']
        cg = r['jacobi']['cold_iters_mean']
        jac = r['jacobi']['precond_iters_mean']
        ic0 = r['ic0']['precond_iters_mean']
        jac_red = r['jacobi']['iteration_reduction'] * 100
        ic0_red = r['ic0']['iteration_reduction'] * 100
        print(f'{N:>5} {cg:>8.1f} {jac:>10.1f} {ic0:>10.1f} {jac_red:>7.1f}% {ic0_red:>7.1f}%')

    return all_results


if __name__ == '__main__':
    run_experiment()
