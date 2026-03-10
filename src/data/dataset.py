from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PoissonDataset(Dataset):
    def __init__(self, data_dir: str | Path, normalise: bool = True) -> None:
        data_dir = Path(data_dir)
        self.sources = np.load(data_dir / 'sources.npy')
        self.solutions = np.load(data_dir / 'solutions.npy')
        self.N = self.sources.shape[1]
        self.normalise = normalise

        if normalise:
            self.source_mean = float(self.sources.mean())
            self.source_std = float(self.sources.std()) + 1e-8
            self.sol_mean = float(self.solutions.mean())
            self.sol_std = float(self.solutions.std()) + 1e-8
        else:
            self.source_mean = 0.0
            self.source_std = 1.0
            self.sol_mean = 0.0
            self.sol_std = 1.0

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.sources[idx]
        solution = self.solutions[idx]

        if self.normalise:
            source = (source - self.source_mean) / self.source_std
            solution = (solution - self.sol_mean) / self.sol_std

        padded = np.zeros((self.N + 2, self.N + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = source

        x = torch.from_numpy(padded).unsqueeze(0)
        y = torch.from_numpy(solution.astype(np.float32)).unsqueeze(0)

        return x, y
