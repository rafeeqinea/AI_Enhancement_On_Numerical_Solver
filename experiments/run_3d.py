"""v4 3D Poisson experiments: Cases 1 (CG), 4 (IC(0)+PCG), 7 (condition loss U-Net+FCG).

Usage:
    python -m experiments.run_3d --train --N 32
    python -m experiments.run_3d --evaluate --N 32
    python -m experiments.run_3d --all --N 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.data.poisson import assemble_poisson_3d, assemble_rhs_3d, get_grid_points_3d
from src.data.generate import generate_source_term_3d
from src.models.unet import UNet
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.training.losses import ConditionLoss


def train_condition_loss_3d(
    N: int,
    epochs: int = 500,
    steps_per_epoch: int = 100,
    lr: float = 1e-3,
    num_probes: int = 32,
    probe_batch_size: int = 4,
    base_features: int = 16,
    levels: int = 3,
    save_dir: str = 'results/checkpoints/3d',
    checkpoint_every: int = 50,
) -> Path:
    """Train U-Net preconditioner with condition loss on 3D Poisson."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training 3D condition loss preconditioner: N={N}, device={device}')
    print(f'Config: epochs={epochs}, steps/epoch={steps_per_epoch}, probes={num_probes}, '
          f'probe_batch={probe_batch_size}, lr={lr}')

    A = assemble_poisson_3d(N)
    model = UNet(base_features=base_features, levels=levels, dim=3).to(device)
    loss_fn = ConditionLoss(A, N, num_probes=num_probes, dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')  # Prevents float16 overflow in AMP

    params = sum(p.numel() for p in model.parameters())
    print(f'Model: {params:,} parameters')

    save_path = Path(save_dir) / f'condition_3d_N{N}'
    save_path.mkdir(parents=True, exist_ok=True)

    loss_history = []
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            loss = loss_fn(
                model, device,
                use_amp=True,
                use_checkpointing=True,
                probe_batch_size=probe_batch_size,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path / 'best.pt')

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f'  Epoch {epoch:4d}/{epochs}: loss={avg_loss:.2f}, '
                  f'best={best_loss:.2f}, lr={scheduler.get_last_lr()[0]:.2e}, '
                  f'time={elapsed:.0f}s')

        if epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), save_path / f'epoch_{epoch:04d}.pt')

    # Save final
    torch.save(model.state_dict(), save_path / 'final.pt')
    np.save(save_path / 'loss_history.npy', np.array(loss_history))

    # Save config
    config = {
        'N': N, 'dim': 3, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch,
        'lr': lr, 'num_probes': num_probes, 'probe_batch_size': probe_batch_size,
        'base_features': base_features, 'levels': levels, 'params': params,
        'best_loss': best_loss, 'training_time_s': time.time() - t_start,
    }
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'Training complete. Best loss: {best_loss:.2f}. '
          f'Time: {time.time() - t_start:.0f}s')
    print(f'Saved to {save_path}')
    return save_path


def evaluate_3d(
    N: int,
    model_path: str | Path | None = None,
    num_samples: int = 50,
    seed: int = 99,
    base_features: int = 16,
    levels: int = 3,
    tol: float = 1e-6,
    max_iter: int = 5000,
) -> dict:
    """Evaluate Cases 1, 4, 7 on 3D Poisson at grid size N."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n=== Evaluating 3D Poisson N={N} ({N**3} DOFs) ===')

    A = assemble_poisson_3d(N)
    X, Y, Z = get_grid_points_3d(N)
    rng = np.random.default_rng(seed)

    # Setup preconditioners
    print('Setting up IC(0)...')
    t0 = time.time()
    ic0 = ic0_preconditioner(A, grid_N=N, dim=3)
    ic0_setup = time.time() - t0
    print(f'  IC(0) setup: {ic0_setup:.2f}s')

    # Load NN preconditioner if model available
    nn_precond = None
    if model_path is not None:
        model_path = Path(model_path)
        best_pt = model_path / 'best.pt'
        if best_pt.exists():
            model = UNet(base_features=base_features, levels=levels, dim=3).to(device)
            model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
            nn_precond = make_nn_preconditioner(model, N, device=device, dim=3)
            print(f'  Loaded NN preconditioner from {best_pt}')
        else:
            print(f'  WARNING: {best_pt} not found, skipping Case 7')

    results = {'N': N, 'n_dof': N**3, 'num_samples': num_samples}

    # Collect iteration counts
    cg_iters, ic0_iters, nn_iters = [], [], []

    for i in range(num_samples):
        f = generate_source_term_3d(X, Y, Z, rng)
        b = assemble_rhs_3d(f, N)

        # Case 1: Baseline CG
        res_cg = conjugate_gradient(A, b, tol=tol, max_iter=max_iter)
        cg_iters.append(res_cg.iterations)

        # Case 4: IC(0) + PCG
        res_ic0 = preconditioned_cg(A, b, ic0, tol=tol, max_iter=max_iter)
        ic0_iters.append(res_ic0.iterations)

        # Case 7: Condition loss U-Net + FCG
        if nn_precond is not None:
            res_nn = flexible_cg(A, b, nn_precond, tol=tol, max_iter=max_iter, m_max=20)
            nn_iters.append(res_nn.iterations)

        if (i + 1) % 10 == 0:
            print(f'  Sample {i+1}/{num_samples}: CG={res_cg.iterations}, '
                  f'IC(0)={res_ic0.iterations}'
                  + (f', NN={res_nn.iterations}' if nn_precond else ''))

    # Summarize
    results['case1_cg'] = {
        'mean': float(np.mean(cg_iters)),
        'std': float(np.std(cg_iters)),
        'all': [int(x) for x in cg_iters],
    }
    results['case4_ic0'] = {
        'mean': float(np.mean(ic0_iters)),
        'std': float(np.std(ic0_iters)),
        'setup_time': ic0_setup,
        'all': [int(x) for x in ic0_iters],
    }
    if nn_iters:
        results['case7_nn'] = {
            'mean': float(np.mean(nn_iters)),
            'std': float(np.std(nn_iters)),
            'all': [int(x) for x in nn_iters],
        }

    # Print summary
    print(f'\n--- Results N={N} ({N**3} DOFs) ---')
    print(f'Case 1 (CG):       {np.mean(cg_iters):.1f} +/- {np.std(cg_iters):.1f} iters')
    print(f'Case 4 (IC(0)):    {np.mean(ic0_iters):.1f} +/- {np.std(ic0_iters):.1f} iters '
          f'({1-np.mean(ic0_iters)/np.mean(cg_iters):.1%} reduction)')
    if nn_iters:
        print(f'Case 7 (NN+FCG):   {np.mean(nn_iters):.1f} +/- {np.std(nn_iters):.1f} iters '
              f'({1-np.mean(nn_iters)/np.mean(cg_iters):.1%} reduction)')

    return results


def main():
    parser = argparse.ArgumentParser(description='v4 3D Poisson experiments')
    parser.add_argument('--N', type=int, default=32, help='Grid size per axis')
    parser.add_argument('--train', action='store_true', help='Train condition loss model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate all cases')
    parser.add_argument('--all', action='store_true', help='Train then evaluate')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--probes', type=int, default=32)
    parser.add_argument('--probe-batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--samples', type=int, default=50, help='Eval samples')
    parser.add_argument('--base-features', type=int, default=16)
    parser.add_argument('--levels', type=int, default=3)
    args = parser.parse_args()

    save_dir = f'results/checkpoints/3d'
    model_dir = Path(save_dir) / f'condition_3d_N{args.N}'

    if args.train or args.all:
        train_condition_loss_3d(
            N=args.N, epochs=args.epochs, steps_per_epoch=args.steps,
            lr=args.lr, num_probes=args.probes, probe_batch_size=args.probe_batch,
            base_features=args.base_features, levels=args.levels,
            save_dir=save_dir,
        )

    if args.evaluate or args.all:
        results = evaluate_3d(
            N=args.N, model_path=model_dir if model_dir.exists() else None,
            num_samples=args.samples,
            base_features=args.base_features, levels=args.levels,
        )
        # Save results
        results_dir = Path('results/3d')
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f'results_N{args.N}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {results_dir / f"results_N{args.N}.json"}')


if __name__ == '__main__':
    main()
