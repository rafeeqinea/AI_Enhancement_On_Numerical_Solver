from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import PoissonDataset


def make_loaders(
    data_dirs: list[str | Path],
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DataLoader], list[DataLoader]]:
    gen = torch.Generator().manual_seed(seed)
    train_loaders: list[DataLoader] = []
    val_loaders: list[DataLoader] = []

    for d in data_dirs:
        ds = PoissonDataset(d)
        n_val = int(len(ds) * val_fraction)
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    return train_loaders, val_loaders


def train_one_epoch(
    model: nn.Module,
    train_loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for loader in train_loaders:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

    return total_loss / total_samples


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loaders: list[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for loader in val_loaders:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

    return total_loss / total_samples


def train(
    model: nn.Module,
    data_dirs: list[str | Path],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 15,
    save_dir: str | Path = 'results/checkpoints',
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    train_loaders, val_loaders = make_loaders(data_dirs, batch_size)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    start = time.perf_counter()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loaders, optimizer, criterion, device)
        val_loss = validate(model, val_loaders, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1

        elapsed_so_far = time.perf_counter() - start
        if (epoch + 1) % 25 == 0 or epoch == 0:
            gap = val_loss / max(train_loss, 1e-12)
            print(f'[{elapsed_so_far:6.0f}s] Epoch {epoch+1:>4d}/{epochs}  '
                  f'train={train_loss:.6f}  val={val_loss:.6f}  '
                  f'gap={gap:.1f}x  lr={current_lr:.2e}  '
                  f'best={best_val_loss:.6f}  wait={epochs_no_improve}/{patience}')

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    elapsed = time.perf_counter() - start

    result = {
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'training_time_seconds': elapsed,
        'device': str(device),
        'history': history,
    }

    with open(save_dir / 'training_log.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result
