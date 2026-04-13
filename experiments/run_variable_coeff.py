from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import (
    assemble_variable_poisson_2d, get_grid_points,
    generate_diffusion_coefficient,
)
from src.data.generate import generate_source_term
from src.models.unet import UNet
from src.training.losses import ConditionLoss
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.direct import solve_direct
from src.solvers.preconditioners import ic0_preconditioner
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.utils.metrics import compute_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'variable_coeff')
N = 32
NUM_TRAIN_COEFFICIENTS = 20
NUM_TEST = 20
EPOCHS = 200
STEPS_PER_EPOCH = 100
PATIENCE = 30
LR = 1e-3
NUM_PROBES = 32
TOL = 1e-6
PATTERNS = ['smooth', 'discontinuous', 'layered']


def train_for_coefficient(A, N, save_dir, device):
    model = UNet(base_features=16, levels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = ConditionLoss(A, N, num_probes=NUM_PROBES).to(device)

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for _ in range(STEPS_PER_EPOCH):
            optimizer.zero_grad()
            loss = loss_fn(model, device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        loss_val = float(np.mean(losses))
        scheduler.step(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f'    Epoch {epoch:4d} | loss={loss_val:.4f} best={best_loss:.4f} p={no_improve}/{PATIENCE}')

        if no_improve >= PATIENCE:
            print(f'    Early stopping at epoch {epoch}')
            break

    return best_loss


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Grid: N={N}, DOF={N*N}')

    X, Y = get_grid_points(N)
    all_results = {}

    for pattern in PATTERNS:
        print(f'\n{"="*60}')
        print(f'Pattern: {pattern}')
        print(f'{"="*60}')

        rng_train = np.random.default_rng(42)
        rng_test = np.random.default_rng(99)

        print(f'\n  Training preconditioner on {NUM_TRAIN_COEFFICIENTS} {pattern} coefficients...')
        D_ref = generate_diffusion_coefficient(X, Y, rng_train, pattern=pattern)
        A_ref = assemble_variable_poisson_2d(N, D_ref)

        save_dir = os.path.join(RESULTS_DIR, f'checkpoints_{pattern}')
        model_path = os.path.join(save_dir, 'best_model.pt')

        if os.path.exists(model_path):
            print(f'  Loading existing checkpoint')
        else:
            best = train_for_coefficient(A_ref, N, save_dir, device)
            print(f'  Best condition loss: {best:.4f}')

        model = UNet(base_features=16, levels=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        print(f'\n  Evaluating on {NUM_TEST} test problems...')
        cg_iters, ic0_iters, nn_iters = [], [], []
        cg_errors, ic0_errors, nn_errors = [], [], []

        for t in range(NUM_TEST):
            D_test = generate_diffusion_coefficient(X, Y, rng_test, pattern=pattern)
            A_test = assemble_variable_poisson_2d(N, D_test)
            f = generate_source_term(X, Y, rng_test)
            b = f.ravel()
            direct = solve_direct(A_test, b)

            r_cg = conjugate_gradient(A_test, b, tol=TOL)
            cg_iters.append(r_cg.iterations)
            cg_errors.append(compute_error(r_cg.solution, direct.solution))

            ic0 = ic0_preconditioner(A_test)
            r_ic0 = preconditioned_cg(A_test, b, ic0, tol=TOL)
            ic0_iters.append(r_ic0.iterations)
            ic0_errors.append(compute_error(r_ic0.solution, direct.solution))

            nn_precond = make_nn_preconditioner(model, N, device=device)
            r_nn = flexible_cg(A_test, b, nn_precond, tol=TOL, max_iter=1000, m_max=20)
            nn_iters.append(r_nn.iterations)
            nn_errors.append(compute_error(r_nn.solution, direct.solution))

        results = {
            'pattern': pattern,
            'N': N,
            'D_ref_range': [float(D_ref.min()), float(D_ref.max())],
            'cg_mean': float(np.mean(cg_iters)),
            'ic0_mean': float(np.mean(ic0_iters)),
            'nn_mean': float(np.mean(nn_iters)),
            'ic0_reduction': float(1 - np.mean(ic0_iters) / np.mean(cg_iters)),
            'nn_reduction': float(1 - np.mean(nn_iters) / np.mean(cg_iters)),
        }
        all_results[pattern] = results

        print(f'\n  {"Method":<20} {"Iters":>8} {"Reduction":>12} {"Error":>12}')
        print(f'  {"-"*55}')
        print(f'  {"CG":<20} {np.mean(cg_iters):>8.1f} {"—":>12} {np.mean(cg_errors):>12.2e}')
        print(f'  {"IC(0)+PCG":<20} {np.mean(ic0_iters):>8.1f} {results["ic0_reduction"]*100:>11.1f}% {np.mean(ic0_errors):>12.2e}')
        print(f'  {"NN(cond)+FCG":<20} {np.mean(nn_iters):>8.1f} {results["nn_reduction"]*100:>11.1f}% {np.mean(nn_errors):>12.2e}')

    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n' + '=' * 60)
    print('VARIABLE-COEFFICIENT SUMMARY (N=32)')
    print('=' * 60)
    print(f'{"Pattern":<15} {"CG":>6} {"IC(0)":>8} {"NN":>8} {"IC0%":>8} {"NN%":>8}')
    print('-' * 55)
    for p in PATTERNS:
        r = all_results[p]
        print(f'{p:<15} {r["cg_mean"]:>6.0f} {r["ic0_mean"]:>8.0f} {r["nn_mean"]:>8.0f} '
              f'{r["ic0_reduction"]*100:>7.0f}% {r["nn_reduction"]*100:>7.0f}%')


if __name__ == '__main__':
    run_experiment()
