from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import assemble_poisson_2d
from src.data.precond_dataset import PrecondDataset
from src.models.unet import UNet
from src.training.losses import ConditionLoss
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.evaluation.evaluate_precond import evaluate_preconditioner

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'nn_precond')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
EVAL_SIZES = [16, 32, 64]
EPOCHS = 300
PATIENCE = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_PROBES = 32


def train_with_condition_loss(
    model: nn.Module,
    A,
    N: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    num_probes: int,
    save_dir: str,
    device: torch.device,
) -> dict:

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = ConditionLoss(A, N, num_probes=num_probes).to(device)

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    epochs_no_improve = 0
    history: dict[str, list[float]] = {'train_loss': [], 'lr': []}
    start_time = time.perf_counter()

    steps_per_epoch = 100

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = loss_fn(model, device)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        loss_val = float(np.mean(epoch_losses))
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(loss_val)
        history['lr'].append(current_lr)

        scheduler.step(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.perf_counter() - start_time
            print(f'  Epoch {epoch:4d} | loss={loss_val:.6f} best={best_loss:.6f} '
                  f'lr={current_lr:.2e} patience={epochs_no_improve}/{patience} [{elapsed:.0f}s]')

        if epochs_no_improve >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break

    total_time = time.perf_counter() - start_time

    with open(os.path.join(save_dir, 'training_log.json'), 'w') as f:
        json.dump(history, f)

    return {
        'epochs_trained': epoch,
        'best_loss': best_loss,
        'training_time_seconds': total_time,
    }


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    all_results = {}

    for N in EVAL_SIZES:
        print(f'\n{"="*60}')
        print(f'Case 7: NN Preconditioner (Condition Loss) — N={N}')
        print(f'{"="*60}')

        A = assemble_poisson_2d(N)
        checkpoint_dir = os.path.join(RESULTS_DIR, f'condition_checkpoints_N{N}')
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')

        model = UNet(base_features=16, levels=3)

        if os.path.exists(model_path):
            print(f'  Loading existing checkpoint: {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model = model.to(device)
        else:
            print(f'  Training U-Net preconditioner (condition loss) ...')
            print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
            result = train_with_condition_loss(
                model, A, N,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, num_probes=NUM_PROBES,
                save_dir=checkpoint_dir, device=device,
            )
            print(f'  Training complete: {result["epochs_trained"]} epochs, '
                  f'best_loss={result["best_loss"]:.6f}')
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model = model.to(device)

        nn_precond = make_nn_preconditioner(model, N, device=device)

        print(f'  Evaluating Case 7: FCG with condition-trained preconditioner ...')
        eval_result = evaluate_preconditioner(
            'NN-Condition', nn_precond, N,
            num_samples=50, tol=1e-6, seed=99,
            use_fcg=True, m_max=20,
        )

        print(f'  CG baseline:  {eval_result["cold_iters_mean"]:.1f} iters')
        print(f'  NN+FCG:       {eval_result["precond_iters_mean"]:.1f} iters')
        print(f'  Reduction:    {eval_result["iteration_reduction"]*100:.1f}%')

        all_results[f'N{N}'] = eval_result

    results_path = os.path.join(RESULTS_DIR, 'condition_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\n' + '='*60)
    print('SUMMARY: Case 7 — NN Preconditioner (Condition Loss)')
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
