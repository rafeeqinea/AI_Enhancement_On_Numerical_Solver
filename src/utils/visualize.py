from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.solvers.cg import CGResult


def plot_convergence(
    cg_result: CGResult,
    title: str = 'CG Convergence',
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(cg_result.residual_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_solution(
    u: np.ndarray,
    N: int,
    title: str = 'Solution',
    save_path: str | Path | None = None,
) -> plt.Figure:
    grid = u.reshape(N, N) if u.ndim == 1 else u

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    fig.colorbar(im, ax=ax, label='u(x, y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_scaling(
    grid_sizes: list[int],
    iterations: list[int],
    title: str = 'CG Iteration Scaling',
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid_sizes, iterations, 'ro-', linewidth=2, markersize=8, label='CG iterations')

    ns = np.array(grid_sizes, dtype=float)
    ax.plot(ns, iterations[0] * (ns / ns[0]) ** 2, 'k--', alpha=0.5, label='O(N²) reference')

    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Iterations to Converge')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_comparison_bar(
    labels: list[str],
    times: list[float],
    title: str = 'Solver Timing Comparison',
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.bar(labels, times, color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
