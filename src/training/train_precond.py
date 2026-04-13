from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.precond_dataset import PrecondDataset


def make_precond_loaders(
    data_dirs: list[str | Path],
    batch_size: int = 32,
    val_fraction: float = 0.2,
) -> tuple[list[DataLoader], list[DataLoader]]:
    train_loaders = []
    val_loaders = []

    for d in data_dirs:
        ds = PrecondDataset(d, normalise=True)
        n_val = int(len(ds) * val_fraction)
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    return train_loaders, val_loaders


def train_preconditioner(
    model: nn.Module,
    data_dirs: list[str | Path],
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    save_dir: str | Path = 'results/checkpoints',
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.MSELoss()

    train_loaders, val_loaders = make_precond_loaders(data_dirs, batch_size)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history: dict[str, list[float]] = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for loader in train_loaders:
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for loader in val_loaders:
                for x_batch, y_batch in loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)
                    val_losses.append(criterion(pred, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.perf_counter() - start_time
            gap = val_loss / max(train_loss, 1e-15)
            print(f'  Epoch {epoch:4d} | train={train_loss:.6f} val={val_loss:.6f} '
                  f'gap={gap:.1f}x lr={current_lr:.2e} patience={epochs_no_improve}/{patience} '
                  f'[{elapsed:.0f}s]')

        if epochs_no_improve >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break

    total_time = time.perf_counter() - start_time

    with open(save_dir / 'training_log.json', 'w') as f:
        json.dump(history, f)

    norm_stats = {}
    for d in data_dirs:
        ds = PrecondDataset(d, normalise=True)
        norm_stats[str(d)] = {
            'res_mean': ds.res_mean, 'res_std': ds.res_std,
            'err_mean': ds.err_mean, 'err_std': ds.err_std,
        }
    with open(save_dir / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f)

    return {
        'epochs_trained': epoch,
        'best_val_loss': best_val_loss,
        'training_time_seconds': total_time,
        'history': history,
    }
