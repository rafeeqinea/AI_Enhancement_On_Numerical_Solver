from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.data.dataset import PoissonDataset
from src.data.precond_dataset import PrecondDataset
from src.models.unet import UNet
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.direct import solve_direct
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.evaluation.evaluate import predict_warmstart
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.utils.metrics import compute_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'factorial')
EVAL_SIZES = [16, 32, 64]
NUM_SAMPLES = 50
SEED = 99
TOL = 1e-6


def load_warmstart_model(device):
    model = UNet(base_features=16, levels=2)
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'results', 'warmstart', 'unet_checkpoints', 'best_model.pt')
    if not os.path.exists(ckpt):
        return None, None
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)

    data_dirs = []
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    for N in EVAL_SIZES:
        d = os.path.join(base, f'N{N}_3000samples_seed42')
        if os.path.exists(d):
            data_dirs.append(d)
    if not data_dirs:
        return model, {'source_mean': 0.0, 'source_std': 1.0, 'sol_mean': 0.0, 'sol_std': 1.0}

    ds = PoissonDataset(data_dirs[0], normalise=True)
    stats = {
        'source_mean': ds.source_mean, 'source_std': ds.source_std,
        'sol_mean': ds.sol_mean, 'sol_std': ds.sol_std,
    }
    return model, stats


def load_condition_model(N, device):
    model = UNet(base_features=16, levels=3)
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'results', 'nn_precond', f'condition_checkpoints_N{N}', 'best_model.pt')
    if not os.path.exists(ckpt):
        return None
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    return model


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ws_model, ws_stats = load_warmstart_model(device)
    if ws_model is None:
        print('WARNING: No warm-start model found. Cases 2, 5, 8 will be skipped.')

    all_results = {}

    for N in EVAL_SIZES:
        print(f'\n{"="*70}')
        print(f'FACTORIAL EXPERIMENT — N={N} (DOF={N*N})')
        print(f'{"="*70}')

        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(SEED)

        jac_precond = jacobi_preconditioner(A)
        ic0_precond = ic0_preconditioner(A)

        cond_model = load_condition_model(N, device)
        nn_precond = None
        if cond_model is not None:
            nn_precond = make_nn_preconditioner(cond_model, N, device=device)

        cases = {
            'Case 1': {'iters': [], 'times': [], 'errors': []},
            'Case 2': {'iters': [], 'times': [], 'errors': []},
            'Case 3': {'iters': [], 'times': [], 'errors': []},
            'Case 4': {'iters': [], 'times': [], 'errors': []},
            'Case 5': {'iters': [], 'times': [], 'errors': []},
            'Case 6': {'iters': [], 'times': [], 'errors': []},
            'Case 7': {'iters': [], 'times': [], 'errors': []},
            'Case 8': {'iters': [], 'times': [], 'errors': []},
        }

        for s in range(NUM_SAMPLES):
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)
            direct = solve_direct(A, b)
            x_true = direct.solution

            r1 = conjugate_gradient(A, b, tol=TOL)
            cases['Case 1']['iters'].append(r1.iterations)
            cases['Case 1']['times'].append(r1.time_seconds)
            cases['Case 1']['errors'].append(compute_error(r1.solution, x_true))

            if ws_model is not None:
                x0 = predict_warmstart(ws_model, f, N, **ws_stats, device=device)
                r2 = conjugate_gradient(A, b, x0=x0, tol=TOL)
                cases['Case 2']['iters'].append(r2.iterations)
                cases['Case 2']['times'].append(r2.time_seconds)
                cases['Case 2']['errors'].append(compute_error(r2.solution, x_true))

            r3 = preconditioned_cg(A, b, jac_precond, tol=TOL)
            cases['Case 3']['iters'].append(r3.iterations)
            cases['Case 3']['times'].append(r3.time_seconds)
            cases['Case 3']['errors'].append(compute_error(r3.solution, x_true))

            r4 = preconditioned_cg(A, b, ic0_precond, tol=TOL)
            cases['Case 4']['iters'].append(r4.iterations)
            cases['Case 4']['times'].append(r4.time_seconds)
            cases['Case 4']['errors'].append(compute_error(r4.solution, x_true))

            if ws_model is not None:
                x0 = predict_warmstart(ws_model, f, N, **ws_stats, device=device)
                r5 = preconditioned_cg(A, b, ic0_precond, x0=x0, tol=TOL)
                cases['Case 5']['iters'].append(r5.iterations)
                cases['Case 5']['times'].append(r5.time_seconds)
                cases['Case 5']['errors'].append(compute_error(r5.solution, x_true))

            if nn_precond is not None:
                r7 = flexible_cg(A, b, nn_precond, tol=TOL, max_iter=1000, m_max=20)
                cases['Case 7']['iters'].append(r7.iterations)
                cases['Case 7']['times'].append(r7.time_seconds)
                cases['Case 7']['errors'].append(compute_error(r7.solution, x_true))

            if ws_model is not None and nn_precond is not None:
                x0 = predict_warmstart(ws_model, f, N, **ws_stats, device=device)
                r8 = flexible_cg(A, b, nn_precond, x0=x0, tol=TOL, max_iter=1000, m_max=20)
                cases['Case 8']['iters'].append(r8.iterations)
                cases['Case 8']['times'].append(r8.time_seconds)
                cases['Case 8']['errors'].append(compute_error(r8.solution, x_true))

        n_results = {}
        baseline_iters = np.mean(cases['Case 1']['iters'])

        for case_name, data in cases.items():
            if not data['iters']:
                continue
            mean_iters = float(np.mean(data['iters']))
            n_results[case_name] = {
                'mean_iters': mean_iters,
                'std_iters': float(np.std(data['iters'])),
                'mean_time': float(np.mean(data['times'])),
                'mean_error': float(np.mean(data['errors'])),
                'reduction_pct': float((1 - mean_iters / baseline_iters) * 100),
            }

        all_results[f'N{N}'] = n_results

        print(f'\n{"Case":<10} {"Iters":>8} {"Reduction":>12} {"Error":>12}')
        print('-' * 45)
        for case_name in ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 7', 'Case 8']:
            if case_name in n_results:
                r = n_results[case_name]
                print(f'{case_name:<10} {r["mean_iters"]:>8.1f} {r["reduction_pct"]:>11.1f}% {r["mean_error"]:>12.2e}')

    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n' + '=' * 70)
    print('FULL FACTORIAL RESULTS')
    print('=' * 70)
    header = f'{"Case":<10}'
    for N in EVAL_SIZES:
        header += f' {"N="+str(N):>10}'
    print(header)
    print('-' * (10 + 11 * len(EVAL_SIZES)))

    for case_name in ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 7', 'Case 8']:
        row = f'{case_name:<10}'
        for N in EVAL_SIZES:
            key = f'N{N}'
            if key in all_results and case_name in all_results[key]:
                row += f' {all_results[key][case_name]["mean_iters"]:>10.1f}'
            else:
                row += f' {"—":>10}'
        print(row)

    return all_results


if __name__ == '__main__':
    run_experiment()
