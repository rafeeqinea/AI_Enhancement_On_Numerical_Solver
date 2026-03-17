from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.precond_dataset import generate_precond_data, PrecondDataset
from src.models.unet import UNet
from src.training.train_precond import train_preconditioner
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.evaluation.evaluate_precond import evaluate_preconditioner
from src.data.poisson import assemble_poisson_2d

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'nn_precond')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
EVAL_SIZES = [16, 32, 64]
NUM_SYSTEMS = 100
CG_ITERS = 100
SEED = 42
EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    all_results = {}

    for N in EVAL_SIZES:
        print(f'\n{"="*60}')
        print(f'Case 6: NN Preconditioner (MSE) — N={N}')
        print(f'{"="*60}')

        data_dir = os.path.join(DATA_DIR, f'precond_N{N}_{NUM_SYSTEMS}sys_{CG_ITERS}iters_seed{SEED}')
        if not os.path.exists(data_dir):
            print(f'  Generating preconditioner training data ...')
            data_dir = str(generate_precond_data(
                N=N, num_systems=NUM_SYSTEMS, cg_iters=CG_ITERS,
                seed=SEED, base_dir=DATA_DIR,
            ))
            print(f'  Data saved to {data_dir}')
        else:
            print(f'  Using existing data: {data_dir}')

        checkpoint_dir = os.path.join(RESULTS_DIR, f'mse_checkpoints_N{N}')
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')

        model = UNet(base_features=16, levels=3)

        if os.path.exists(model_path):
            print(f'  Loading existing checkpoint: {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model = model.to(device)
        else:
            print(f'  Training U-Net preconditioner (MSE loss) ...')
            print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
            result = train_preconditioner(
                model, [data_dir],
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                save_dir=checkpoint_dir, device=device,
            )
            print(f'  Training complete: {result["epochs_trained"]} epochs, '
                  f'best_val={result["best_val_loss"]:.6f}')
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model = model.to(device)

        ds = PrecondDataset(data_dir, normalise=True)
        nn_precond = make_nn_preconditioner(
            model, N,
            res_mean=ds.res_mean, res_std=ds.res_std,
            err_mean=ds.err_mean, err_std=ds.err_std,
            device=device,
        )

        print(f'  Evaluating Case 6: FCG with NN preconditioner ...')
        eval_result = evaluate_preconditioner(
            'NN-MSE', nn_precond, N,
            num_samples=50, tol=1e-6, seed=99,
            use_fcg=True, m_max=20,
        )

        print(f'  CG baseline:  {eval_result["cold_iters_mean"]:.1f} iters')
        print(f'  NN+FCG:       {eval_result["precond_iters_mean"]:.1f} iters')
        print(f'  Reduction:    {eval_result["iteration_reduction"]*100:.1f}%')

        all_results[f'N{N}'] = eval_result

    results_path = os.path.join(RESULTS_DIR, 'mse_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n' + '='*60)
    print('SUMMARY: Case 6 — NN Preconditioner (MSE Loss)')
    print('='*60)
    print(f'{"N":>5} {"CG":>8} {"NN+FCG":>10} {"Reduction":>12}')
    print('-'*40)
    for N in EVAL_SIZES:
        r = all_results[f'N{N}']
        print(f'{N:>5} {r["cold_iters_mean"]:>8.1f} {r["precond_iters_mean"]:>10.1f} '
              f'{r["iteration_reduction"]*100:>11.1f}%')

    return all_results


if __name__ == '__main__':
    run_experiment()
