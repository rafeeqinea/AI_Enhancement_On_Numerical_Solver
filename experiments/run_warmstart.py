from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from src.models.cnn import BaselineCNN
from src.models.unet import UNet
from src.data.dataset import PoissonDataset
from src.training.train import train
from src.evaluation.evaluate import evaluate_warmstart
from src.utils.visualize import (
    plot_scaling, plot_comparison_bar,
    animate_training_curve, animate_predictions,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'warmstart')
DATA_DIRS = [
    'data/processed/N16_3000samples_seed42',
    'data/processed/N32_3000samples_seed42',
    'data/processed/N64_3000samples_seed42',
]
EVAL_SIZES = [16, 32, 64]
EVAL_SAMPLES = 50
TOL = 1e-6


def train_model(model: torch.nn.Module, name: str, epochs: int = 5000,
                weight_decay: float = 1e-4, save_snapshots: bool = False) -> dict | None:
    ckpt_path = os.path.join(RESULTS_DIR, f'{name}_checkpoints', 'best_model.pt')
    if os.path.exists(ckpt_path) and '--retrain' not in sys.argv:
        print(f'  Found existing checkpoint at {ckpt_path}, skipping training.')
        print(f'  (use --retrain to force retraining)')
        return None

    save_dir = os.path.join(RESULTS_DIR, f'{name}_checkpoints')
    result = train(
        model,
        data_dirs=DATA_DIRS,
        epochs=epochs,
        batch_size=32,
        lr=1e-3,
        weight_decay=weight_decay,
        patience=100,
        save_dir=save_dir,
        save_snapshots=save_snapshots,
        snapshot_every=5,
    )
    print(f'\n{name} training: {result["epochs_trained"]} epochs, '
          f'best val loss = {result["best_val_loss"]:.6f}, '
          f'time = {result["training_time_seconds"]:.1f}s')
    return result


def get_norm_stats_per_n(data_dirs: list[str]) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for d in data_dirs:
        ds = PoissonDataset(d, normalise=True)
        stats[ds.N] = {
            'source_mean': ds.source_mean,
            'source_std': ds.source_std,
            'sol_mean': ds.sol_mean,
            'sol_std': ds.sol_std,
        }
    return stats


def evaluate_model(model: torch.nn.Module, name: str,
                   norm_stats_per_n: dict[int, dict]) -> list[dict]:
    device = next(model.parameters()).device
    results = []
    for N in EVAL_SIZES:
        ns = norm_stats_per_n.get(N)
        print(f'  Evaluating {name} on N={N}...', end=' ', flush=True)
        r = evaluate_warmstart(model, N, norm_stats=ns,
                               num_samples=EVAL_SAMPLES, tol=TOL, device=device)
        print(f'cold={r["cold_iters_mean"]:.0f}, warm={r["warm_iters_mean"]:.0f}, '
              f'reduction={r["iteration_reduction"]:.1%}')
        results.append(r)
    return results


def load_model(model: torch.nn.Module, name: str, device: torch.device) -> None:
    ckpt_path = os.path.join(RESULTS_DIR, f'{name}_checkpoints', 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)


def run_experiment() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('=' * 60)

    print('\n--- BaselineCNN ---')
    cnn = BaselineCNN(hidden_channels=32, num_layers=7)
    cnn_train = train_model(cnn, 'cnn')
    load_model(cnn, 'cnn', device)

    print('\n--- UNet ---')
    unet = UNet(base_features=16, levels=2)
    unet_train = train_model(unet, 'unet', save_snapshots=True)
    load_model(unet, 'unet', device)

    print('\n--- Computing normalisation stats per N ---')
    norm_stats_per_n = get_norm_stats_per_n(DATA_DIRS)
    for N, ns in sorted(norm_stats_per_n.items()):
        print(f'  N={N}: source(mean={ns["source_mean"]:.4f}, std={ns["source_std"]:.4f}), '
              f'sol(mean={ns["sol_mean"]:.4f}, std={ns["sol_std"]:.4f})')

    print('\n--- Evaluating CNN warm-start ---')
    cnn_eval = evaluate_model(cnn, 'CNN', norm_stats_per_n)

    print('\n--- Evaluating UNet warm-start ---')
    unet_eval = evaluate_model(unet, 'UNet', norm_stats_per_n)

    cnn_params = sum(p.numel() for p in cnn.parameters())
    unet_params = sum(p.numel() for p in unet.parameters())

    cold_iters = [r['cold_iters_mean'] for r in cnn_eval]
    cnn_iters = [r['warm_iters_mean'] for r in cnn_eval]
    unet_iters = [r['warm_iters_mean'] for r in unet_eval]

    plot_scaling(
        EVAL_SIZES, [int(round(c)) for c in cold_iters],
        title='Cold CG Iterations',
        save_path=os.path.join(RESULTS_DIR, 'cold_scaling.png'),
    )

    for label, iters in [('CNN', cnn_iters), ('UNet', unet_iters)]:
        plot_scaling(
            EVAL_SIZES, [int(round(i)) for i in iters],
            title=f'{label} Warm-Start Iterations',
            save_path=os.path.join(RESULTS_DIR, f'{label.lower()}_warmstart_scaling.png'),
        )

    for i, N in enumerate(EVAL_SIZES):
        plot_comparison_bar(
            ['Cold CG', 'CNN Warm', 'UNet Warm'],
            [cold_iters[i], cnn_iters[i], unet_iters[i]],
            title=f'Iterations Comparison (N={N})',
            save_path=os.path.join(RESULTS_DIR, f'comparison_N{N}.png'),
        )

    summary = {
        'models': {
            'cnn': {'params': cnn_params, 'training': cnn_train},
            'unet': {'params': unet_params, 'training': unet_train},
        },
        'evaluation': {
            'cnn': cnn_eval,
            'unet': unet_eval,
        },
    }

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n--- Generating training animations ---')
    for name in ['cnn', 'unet']:
        log_path = os.path.join(RESULTS_DIR, f'{name}_checkpoints', 'training_log.json')
        if os.path.exists(log_path):
            animate_training_curve(
                log_path,
                os.path.join(RESULTS_DIR, f'{name}_training.mp4'),
                title=f'{name.upper()} Training Progress',
            )

    snap_dir = os.path.join(RESULTS_DIR, 'unet_checkpoints', 'snapshots')
    if os.path.isdir(snap_dir):
        animate_predictions(
            snap_dir,
            os.path.join(RESULTS_DIR, 'unet_prediction_evolution.mp4'),
            title='UNet Prediction Evolution',
        )

    print('\n' + '=' * 60)
    print(f'CNN params: {cnn_params:,}')
    print(f'UNet params: {unet_params:,}')
    print(f'\nResults saved to {RESULTS_DIR}/')

    return summary


if __name__ == '__main__':
    run_experiment()
