"""Generate verified Chapter 5 figures from authoritative JSON artefacts.

Produces:
  1. fig3_case_comparison_regen.png  — 8-case factorial at N=32
  2. fig4_curriculum_regen.png       — curriculum N=16–128 (no N=256)
  3. fig_residual_trajectory.png     — Case 6 vs Case 7 residual at N=32
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'report_figures')

# ─── Figure 1: 8-case factorial comparison at N=32 ───────────────────────

def generate_factorial_figure():
    """Read from factorial/results.json + mse_results.json. Plot 8 cases at N=32."""
    with open(os.path.join(RESULTS_DIR, 'factorial', 'results.json')) as f:
        factorial = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'nn_precond', 'mse_results.json')) as f:
        mse = json.load(f)

    n32 = factorial['N32']
    case6_iters = mse['N32']['precond_iters_mean']  # 1000.0

    labels = [
        'Case 1\nCG',
        'Case 2\nWS+CG',
        'Case 3\nJacobi',
        'Case 4\nIC(0)',
        'Case 5\nWS+IC(0)',
        'Case 6\nMSE',
        'Case 7\nCond. Loss',
        'Case 8\nWS+Cond.',
    ]
    iters = [
        n32['Case 1']['mean_iters'],
        n32['Case 2']['mean_iters'],
        n32['Case 3']['mean_iters'],
        n32['Case 4']['mean_iters'],
        n32['Case 5']['mean_iters'],
        case6_iters,
        n32['Case 7']['mean_iters'],
        n32['Case 8']['mean_iters'],
    ]

    # Verify before plotting (Correction #5)
    print('Factorial figure — values from JSON:')
    for label, val in zip(labels, iters):
        print(f'  {label.replace(chr(10), " "):20s}  {val:.2f}')

    colors = [
        '#5B9BD5', '#5B9BD5', '#5B9BD5',   # blue: baseline group
        '#70AD47', '#70AD47',                # green: classical group
        '#FF4444',                            # red: MSE failure
        '#ED7D31', '#ED7D31',                # orange: condition loss group
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Truncate Case 6 for display; actual value annotated
    display_iters = list(iters)
    y_max = 110
    case6_truncated = False
    if display_iters[5] > y_max:
        display_iters[5] = y_max
        case6_truncated = True

    bars = ax.bar(range(8), display_iters, color=colors,
                  edgecolor='#333333', linewidth=0.6, width=0.7)

    # Add hatching to the truncated Case 6 bar to signal it's off-scale
    if case6_truncated:
        bars[5].set_hatch('///')

    # Value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, iters)):
        if i == 5 and case6_truncated:
            ax.text(bar.get_x() + bar.get_width() / 2, y_max + 1,
                    f'{val:.0f}\nFAILS',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#CC0000')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(range(8))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Mean Iterations', fontsize=11)
    ax.set_title('8-Case Factorial Comparison (N=32, 2D Poisson, 50 samples)',
                 fontsize=12, pad=12)
    ax.set_ylim(0, y_max + 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    legend_handles = [
        mpatches.Patch(color='#5B9BD5', label='Baseline (no effective preconditioner)'),
        mpatches.Patch(color='#70AD47', label='Classical (IC(0))'),
        mpatches.Patch(color='#FF4444', label='Learned — MSE (FAILS)'),
        mpatches.Patch(color='#ED7D31', label='Learned — Condition Loss'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_case_comparison_regen.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return out_path


# ─── Figure 2: Curriculum training N=16–128 (no N=256) ───────────────────

def generate_curriculum_figure():
    """Read from curriculum_results.json. Plot N=16 to N=128 only."""
    with open(os.path.join(RESULTS_DIR, 'curriculum', '2d',
                           'curriculum_results.json')) as f:
        curriculum = json.load(f)

    grid_sizes = [16, 32, 64, 128]
    nn_iters = []
    train_times = []
    train_types = []

    for N in grid_sizes:
        key = f'N{N}'
        entry = curriculum[key]
        nn_iters.append(entry['nn']['iters_mean'])
        train_times.append(entry['train_time_s'])
        train_types.append(entry.get('train_type', 'scratch'))

    cumulative_time_min = np.cumsum(train_times) / 60.0

    # Verify before plotting
    print('\nCurriculum figure — values from JSON:')
    for N, it, tt, tp in zip(grid_sizes, nn_iters, train_times, train_types):
        print(f'  N={N:3d}  iters={it:.2f}  train_time={tt:.0f}s  type={tp}')

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(grid_sizes))
    bar_colors = ['#ED7D31'] * len(grid_sizes)
    bars = ax1.bar(x, nn_iters, color=bar_colors, edgecolor='#333333',
                   linewidth=0.6, width=0.55)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, nn_iters)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')

    # X-axis labels with train type
    xlabels = []
    for N, tp in zip(grid_sizes, train_types):
        epochs = curriculum[f'N{N}'].get('epochs', '?')
        xlabels.append(f'N={N}\n{tp} {epochs}ep')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=8.5)
    ax1.set_ylabel('NN+FCG Iterations', fontsize=11, color='#ED7D31')
    ax1.set_title('Curriculum Training: Iteration Counts and Training Time',
                  fontsize=12, pad=12)
    ax1.set_ylim(0, max(nn_iters) * 1.25)
    ax1.spines['top'].set_visible(False)

    # Secondary axis: cumulative training time
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_time_min, 'k-o', linewidth=1.5, markersize=5,
             label='Cumulative training time')
    ax2.set_ylabel('Cumulative Training Time (min)', fontsize=10)
    ax2.spines['top'].set_visible(False)

    # Combined legend
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color='#ED7D31', label='NN+FCG iterations'),
        Line2D([0], [0], color='k', marker='o', linewidth=1.5,
               markersize=5, label='Cumulative training time'),
    ]
    ax1.legend(handles=handles, loc='upper left', fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig4_curriculum_regen.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return out_path


# ─── Figure 3: Residual trajectory — Case 6 vs Case 7 at N=32 ───────────

def generate_residual_trajectory():
    """Run one problem at N=32 with seed 99. Compare CG, MSE+FCG, Cond+FCG."""
    from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
    from src.data.generate import generate_source_term
    from src.models.unet import UNet
    from src.solvers.cg import conjugate_gradient
    from src.solvers.fcg import flexible_cg
    from src.evaluation.nn_preconditioner import make_nn_preconditioner

    N = 32
    A = assemble_poisson_2d(N)
    rng = np.random.default_rng(99)
    X, Y = get_grid_points(N)
    f = generate_source_term(X, Y, rng)
    b = assemble_rhs(f, N)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Plain CG
    cg_result = conjugate_gradient(A, b, tol=1e-6, max_iter=1000)
    print(f'\nResidual trajectory — CG: {cg_result.iterations} iters')

    # Case 6: MSE-trained U-Net in FCG
    mse_model = UNet(base_features=16, levels=3)
    mse_ckpt = os.path.join(RESULTS_DIR, 'nn_precond',
                            'mse_checkpoints_N32', 'best_model.pt')
    mse_model.load_state_dict(
        torch.load(mse_ckpt, map_location=device, weights_only=True))
    mse_model = mse_model.to(device)
    mse_precond = make_nn_preconditioner(mse_model, N, device=device)
    mse_result = flexible_cg(A, b, mse_precond, tol=1e-6, max_iter=1000,
                             m_max=20)
    print(f'Residual trajectory — MSE+FCG: {mse_result.iterations} iters, '
          f'converged={mse_result.converged}')

    # Case 7: Condition-loss-trained U-Net in FCG
    cond_model = UNet(base_features=16, levels=3)
    cond_ckpt = os.path.join(RESULTS_DIR, 'nn_precond',
                             'condition_checkpoints_N32', 'best_model.pt')
    cond_model.load_state_dict(
        torch.load(cond_ckpt, map_location=device, weights_only=True))
    cond_model = cond_model.to(device)
    cond_precond = make_nn_preconditioner(cond_model, N, device=device)
    cond_result = flexible_cg(A, b, cond_precond, tol=1e-6, max_iter=1000,
                              m_max=20)
    print(f'Residual trajectory — Cond+FCG: {cond_result.iterations} iters, '
          f'converged={cond_result.converged}')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.semilogy(range(len(cg_result.residual_history)),
                cg_result.residual_history,
                'b-', linewidth=1.2, alpha=0.7, label='CG (no preconditioner)')
    ax.semilogy(range(len(mse_result.residual_history)),
                mse_result.residual_history,
                'r-', linewidth=1.5, label='Case 6: MSE + FCG')
    ax.semilogy(range(len(cond_result.residual_history)),
                cond_result.residual_history,
                '#ED7D31', linewidth=1.5,
                label='Case 7: Condition Loss + FCG')

    # Tolerance line
    ax.axhline(y=1e-6, color='gray', linestyle='--', linewidth=0.8,
               label='Tolerance (1e-6)')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Relative Residual Norm', fontsize=11)
    ax.set_title('Residual Convergence: MSE vs Condition Loss (N=32, 2D Poisson)',
                 fontsize=12, pad=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, max(len(cg_result.residual_history),
                       len(mse_result.residual_history)) + 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig_residual_trajectory.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('='*60)
    print('Chapter 5 Figure Generation')
    print('='*60)

    p1 = generate_factorial_figure()
    p2 = generate_curriculum_figure()
    p3 = generate_residual_trajectory()

    print('\n' + '='*60)
    print('DONE — Chapter 5 figures generated:')
    print(f'  1. {p1}')
    print(f'  2. {p2}')
    print(f'  3. {p3}')
    print('='*60)
