from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.direct import solve_direct


def generate_precond_data(
    N: int,
    num_systems: int = 100,
    cg_iters: int = 100,
    seed: int = 42,
    base_dir: str | Path = 'data/processed',
) -> Path:
    rng = np.random.default_rng(seed)
    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)

    all_residuals = []
    all_errors = []

    for i in range(num_systems):
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)
        direct = solve_direct(A, b)
        u_exact = direct.solution

        x = np.zeros(N * N)
        r = b - A @ x
        d = r.copy()
        delta_new = r @ r

        for k in range(cg_iters):
            e = u_exact - x
            all_residuals.append(r.reshape(N, N).copy())
            all_errors.append(e.reshape(N, N).copy())

            q = A @ d
            alpha = delta_new / (d @ q)
            x = x + alpha * d
            r = r - alpha * q
            delta_old = delta_new
            delta_new = r @ r
            beta = delta_new / delta_old
            d = r + beta * d

    residuals = np.stack(all_residuals).astype(np.float32)
    errors = np.stack(all_errors).astype(np.float32)

    save_dir = Path(base_dir) / f'precond_N{N}_{num_systems}sys_{cg_iters}iters_seed{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / 'residuals.npy', residuals)
    np.save(save_dir / 'errors.npy', errors)

    return save_dir


class PrecondDataset(Dataset):
    def __init__(self, data_dir: str | Path, normalise: bool = True) -> None:
        data_dir = Path(data_dir)
        self.residuals = np.load(data_dir / 'residuals.npy')
        self.errors = np.load(data_dir / 'errors.npy')
        self.N = self.residuals.shape[1]
        self.normalise = normalise

        if normalise:
            self.res_mean = float(self.residuals.mean())
            self.res_std = float(self.residuals.std()) + 1e-8
            self.err_mean = float(self.errors.mean())
            self.err_std = float(self.errors.std()) + 1e-8
        else:
            self.res_mean = 0.0
            self.res_std = 1.0
            self.err_mean = 0.0
            self.err_std = 1.0

    def __len__(self) -> int:
        return len(self.residuals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        res = self.residuals[idx]
        err = self.errors[idx]

        if self.normalise:
            res = (res - self.res_mean) / self.res_std
            err = (err - self.err_mean) / self.err_std

        padded = np.zeros((self.N + 2, self.N + 2), dtype=np.float32)
        padded[1:-1, 1:-1] = res

        x = torch.from_numpy(padded).unsqueeze(0)
        y = torch.from_numpy(err.astype(np.float32)).unsqueeze(0)

        return x, y
