from __future__ import annotations

import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def animate_training_curve(
    log_path: str | Path,
    save_path: str | Path,
    fps: int = 30,
    epochs_per_frame: int = 2,
    title: str = 'Training Progress',
) -> None:
    with open(log_path) as f:
        log = json.load(f)

    train_loss = log['history']['train_loss']
    val_loss = log['history']['val_loss']
    lr_hist = log['history']['lr']
    total = len(train_loss)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1.set_xlim(0, total)
    y_max = max(max(train_loss[:min(50, total)]), max(val_loss[:min(50, total)]))
    y_min = min(min(train_loss), min(val_loss)) * 0.9
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, alpha=0.3)

    line_train, = ax1.plot([], [], 'b-', linewidth=1.2, label='Train')
    line_val, = ax1.plot([], [], 'r-', linewidth=1.2, label='Val')
    best_dot, = ax1.plot([], [], 'g*', markersize=12)
    epoch_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=11,
                          verticalalignment='top', fontfamily='monospace')
    ax1.legend(loc='upper right')

    ax2.set_xlim(0, total)
    ax2.set_ylim(min(lr_hist) * 0.5, max(lr_hist) * 2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, alpha=0.3)
    line_lr, = ax2.plot([], [], 'g-', linewidth=1.2)

    fig.tight_layout()

    frame_indices = list(range(0, total, epochs_per_frame))
    if frame_indices[-1] != total - 1:
        frame_indices.append(total - 1)

    def update(frame_idx):
        end = frame_indices[frame_idx] + 1
        xs = list(range(end))
        line_train.set_data(xs, train_loss[:end])
        line_val.set_data(xs, val_loss[:end])
        line_lr.set_data(xs, lr_hist[:end])

        best_idx = int(np.argmin(val_loss[:end]))
        best_dot.set_data([best_idx], [val_loss[best_idx]])

        gap = val_loss[end - 1] / max(train_loss[end - 1], 1e-12)
        epoch_text.set_text(
            f'Epoch {end}/{total}\n'
            f'Train: {train_loss[end-1]:.6f}\n'
            f'Val:   {val_loss[end-1]:.6f}\n'
            f'Best:  {val_loss[best_idx]:.6f} (ep {best_idx+1})\n'
            f'Gap:   {gap:.1f}x'
        )
        return line_train, line_val, line_lr, best_dot, epoch_text

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                   interval=1000 // fps, blit=True)

    save_path = Path(save_path)
    if save_path.suffix == '.gif':
        anim.save(str(save_path), writer='pillow', fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(save_path), writer=writer)

    plt.close(fig)
    print(f'Saved animation: {save_path} ({len(frame_indices)} frames, {len(frame_indices)/fps:.1f}s)')


def animate_predictions(
    snapshot_dir: str | Path,
    save_path: str | Path,
    fps: int = 15,
    title: str = 'Prediction Evolution',
) -> None:
    snapshot_dir = Path(snapshot_dir)
    files = sorted(snapshot_dir.glob('epoch_*.npz'), key=lambda f: int(f.stem.split('_')[1]))
    if not files:
        print(f'No snapshots found in {snapshot_dir}')
        return

    first = np.load(files[0])
    source = first['source']
    truth = first['truth']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].set_title('Source Term')
    axes[0].imshow(source, origin='lower', cmap='RdBu_r')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_title('True Solution')
    axes[1].imshow(truth, origin='lower', cmap='viridis')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    vmin, vmax = truth.min(), truth.max()
    pred_data = first['prediction']
    im_pred = axes[2].imshow(pred_data, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title('Prediction (Epoch 0)')
    fig.colorbar(im_pred, ax=axes[2], fraction=0.046)
    fig.tight_layout()

    def update(frame_idx):
        data = np.load(files[frame_idx])
        pred = data['prediction']
        epoch = int(files[frame_idx].stem.split('_')[1])
        mse = float(np.mean((pred - truth) ** 2))
        im_pred.set_data(pred)
        axes[2].set_title(f'Prediction (Epoch {epoch}, MSE={mse:.4f})')
        return [im_pred]

    anim = animation.FuncAnimation(fig, update, frames=len(files),
                                   interval=1000 // fps, blit=True)

    save_path = Path(save_path)
    if save_path.suffix == '.gif':
        anim.save(str(save_path), writer='pillow', fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(save_path), writer=writer)

    plt.close(fig)
    print(f'Saved animation: {save_path} ({len(files)} frames, {len(files)/fps:.1f}s)')
