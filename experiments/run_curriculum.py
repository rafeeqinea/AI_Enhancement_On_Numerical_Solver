"""Curriculum training: train on smallest grid, fine-tune upward.

2D: N=16 → N=32 → N=64 → N=128
3D: N=16 → N=32 → N=64

Usage:
    python -m experiments.run_curriculum --dim 2
    python -m experiments.run_curriculum --dim 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.data.poisson import assemble_poisson_2d, assemble_poisson_3d
from src.models.unet import UNet
from src.training.losses import ConditionLoss
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.preconditioners import ic0_preconditioner
from src.solvers.direct import solve_direct
from src.evaluation.nn_preconditioner import make_nn_preconditioner


def train_step(
    model: torch.nn.Module,
    loss_fn: ConditionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    dim: int,
    probe_batch: int,
) -> float:
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(
        model, device,
        use_amp=True,
        use_checkpointing=(dim == 3),
        probe_batch_size=probe_batch,
    )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def evaluate(model, N, dim, device, num_samples=50) -> dict:
    """Evaluate CG, IC(0), NN+FCG on num_samples problems."""
    if dim == 2:
        A = assemble_poisson_2d(N)
    else:
        A = assemble_poisson_3d(N)

    n = N ** dim
    rng = np.random.default_rng(99)

    grid_N = N if dim == 3 else None
    ic0 = ic0_preconditioner(A, grid_N=grid_N, dim=dim)
    nn_precond = make_nn_preconditioner(model, N, device=device, dim=dim)

    cg_iters, ic0_iters, nn_iters = [], [], []
    cg_times, ic0_times, nn_times = [], [], []
    cg_accs, ic0_accs, nn_accs = [], [], []

    for _ in range(num_samples):
        b = rng.standard_normal(n)
        exact = solve_direct(A, b).solution

        t0 = time.perf_counter()
        res_cg = conjugate_gradient(A, b, tol=1e-6)
        cg_times.append(time.perf_counter() - t0)
        cg_iters.append(res_cg.iterations)
        cg_accs.append((1 - np.linalg.norm(res_cg.solution - exact) / np.linalg.norm(exact)) * 100)

        t0 = time.perf_counter()
        res_ic0 = preconditioned_cg(A, b, ic0, tol=1e-6)
        ic0_times.append(time.perf_counter() - t0)
        ic0_iters.append(res_ic0.iterations)
        ic0_accs.append((1 - np.linalg.norm(res_ic0.solution - exact) / np.linalg.norm(exact)) * 100)

        t0 = time.perf_counter()
        res_nn = flexible_cg(A, b, nn_precond, tol=1e-6, max_iter=1000, m_max=20)
        nn_times.append(time.perf_counter() - t0)
        nn_iters.append(res_nn.iterations)
        nn_accs.append((1 - np.linalg.norm(res_nn.solution - exact) / np.linalg.norm(exact)) * 100)

    return {
        'N': N, 'dim': dim, 'n_dof': n, 'num_samples': num_samples,
        'cg': {'iters_mean': float(np.mean(cg_iters)), 'iters_std': float(np.std(cg_iters)),
               'time_ms': float(np.mean(cg_times) * 1000), 'accuracy': float(np.mean(cg_accs))},
        'ic0': {'iters_mean': float(np.mean(ic0_iters)), 'iters_std': float(np.std(ic0_iters)),
                'time_ms': float(np.mean(ic0_times) * 1000), 'accuracy': float(np.mean(ic0_accs))},
        'nn': {'iters_mean': float(np.mean(nn_iters)), 'iters_std': float(np.std(nn_iters)),
               'time_ms': float(np.mean(nn_times) * 1000), 'accuracy': float(np.mean(nn_accs)),
               'converged_all': all(i < 1000 for i in nn_iters)},
    }


def run_curriculum(dim: int, grid_sizes: list[int], epochs_scratch: int, epochs_finetune: int,
                   steps: int, probes: int, probe_batch: int, base_features: int, levels: int,
                   lr_scratch: float, lr_finetune: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_base = Path(f'results/curriculum/{dim}d')
    save_base.mkdir(parents=True, exist_ok=True)

    model = UNet(base_features=base_features, levels=levels, dim=dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'{"="*70}')
    print(f'CURRICULUM TRAINING ({dim}D)')
    print(f'Grid sizes: {grid_sizes}')
    print(f'Model: {params:,} params, base_features={base_features}, levels={levels}')
    print(f'Scratch: {epochs_scratch} epochs, Fine-tune: {epochs_finetune} epochs')
    print(f'{"="*70}')

    all_results = {}

    for i, N in enumerate(grid_sizes):
        is_scratch = (i == 0)
        epochs = epochs_scratch if is_scratch else epochs_finetune
        lr = lr_scratch if is_scratch else lr_finetune

        print(f'\n--- {"TRAINING FROM SCRATCH" if is_scratch else "FINE-TUNING"}: N={N} ({N**dim} DOFs), {epochs} epochs ---')

        # Build A and loss for this grid size
        A = assemble_poisson_2d(N) if dim == 2 else assemble_poisson_3d(N)
        loss_fn = ConditionLoss(A, N, num_probes=probes, dim=dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler('cuda')

        save_dir = save_base / f'N{N}'
        save_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')
        loss_history = []
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for step in range(steps):
                epoch_loss += train_step(model, loss_fn, optimizer, scaler, device, dim, probe_batch)
            scheduler.step()
            avg_loss = epoch_loss / steps
            loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_dir / 'best.pt')

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t_start
                print(f'  Epoch {epoch:4d}/{epochs}: loss={avg_loss:.2f}, best={best_loss:.2f}, time={elapsed:.0f}s')

        torch.save(model.state_dict(), save_dir / 'final.pt')
        np.save(save_dir / 'loss_history.npy', np.array(loss_history))
        train_time = time.time() - t_start

        # Save config
        config = {
            'N': N, 'dim': dim, 'epochs': epochs, 'steps': steps,
            'lr': lr, 'is_scratch': is_scratch, 'probes': probes,
            'best_loss': best_loss, 'training_time_s': train_time,
            'prev_N': grid_sizes[i - 1] if i > 0 else None,
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f'  Training done in {train_time:.0f}s. Best loss: {best_loss:.2f}')

        # Evaluate at this grid size
        print(f'  Evaluating N={N}...')
        model.load_state_dict(torch.load(save_dir / 'best.pt', map_location=device, weights_only=True))
        results = evaluate(model, N, dim, device, num_samples=50)
        all_results[f'N{N}'] = results
        all_results[f'N{N}']['train_time_s'] = train_time
        all_results[f'N{N}']['train_type'] = 'scratch' if is_scratch else 'finetune'
        all_results[f'N{N}']['epochs'] = epochs

        red_ic0 = 1 - results['ic0']['iters_mean'] / results['cg']['iters_mean']
        red_nn = 1 - results['nn']['iters_mean'] / results['cg']['iters_mean']
        print(f'  CG:     {results["cg"]["iters_mean"]:.1f} iters, {results["cg"]["time_ms"]:.2f}ms, {results["cg"]["accuracy"]:.7f}%')
        print(f'  IC(0):  {results["ic0"]["iters_mean"]:.1f} iters, {results["ic0"]["time_ms"]:.2f}ms, {results["ic0"]["accuracy"]:.7f}% ({red_ic0:.1%})')
        print(f'  NN:     {results["nn"]["iters_mean"]:.1f} iters, {results["nn"]["time_ms"]:.2f}ms, {results["nn"]["accuracy"]:.7f}% ({red_nn:.1%})')

    # Also test transfer: evaluate the final model (trained@largest N) on ALL grid sizes
    print(f'\n--- TRANSFER TEST: final model tested on all sizes ---')
    for N in grid_sizes:
        results = evaluate(model, N, dim, device, num_samples=50)
        red_nn = 1 - results['nn']['iters_mean'] / results['cg']['iters_mean']
        print(f'  N={N:4d}: CG={results["cg"]["iters_mean"]:.1f}, NN={results["nn"]["iters_mean"]:.1f} ({red_nn:.1%}), acc={results["nn"]["accuracy"]:.7f}%')
        all_results[f'transfer_N{N}'] = results

    # Save all results
    with open(save_base / 'curriculum_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nAll results saved to {save_base / "curriculum_results.json"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, required=True, choices=[2, 3])
    parser.add_argument('--epochs-scratch', type=int, default=None)
    parser.add_argument('--epochs-finetune', type=int, default=None)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--probes', type=int, default=None)
    parser.add_argument('--probe-batch', type=int, default=None)
    parser.add_argument('--base-features', type=int, default=None)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--lr-scratch', type=float, default=1e-3)
    parser.add_argument('--lr-finetune', type=float, default=3e-4)
    args = parser.parse_args()

    if args.dim == 2:
        grid_sizes = [16, 32, 64, 128]
        defaults = {'epochs_scratch': 200, 'epochs_finetune': 50, 'probes': 64,
                     'probe_batch': 64, 'base_features': 16}
    else:
        grid_sizes = [16, 32, 64]
        defaults = {'epochs_scratch': 200, 'epochs_finetune': 100, 'probes': 128,
                     'probe_batch': 128, 'base_features': 32}

    run_curriculum(
        dim=args.dim,
        grid_sizes=grid_sizes,
        epochs_scratch=args.epochs_scratch or defaults['epochs_scratch'],
        epochs_finetune=args.epochs_finetune or defaults['epochs_finetune'],
        steps=args.steps,
        probes=args.probes or defaults['probes'],
        probe_batch=args.probe_batch or defaults['probe_batch'],
        base_features=args.base_features or defaults['base_features'],
        levels=args.levels,
        lr_scratch=args.lr_scratch,
        lr_finetune=args.lr_finetune,
    )


if __name__ == '__main__':
    main()
