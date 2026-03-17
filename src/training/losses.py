from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class ConditionLoss(nn.Module):
    def __init__(self, A: sp.spmatrix, N: int, num_probes: int = 32) -> None:
        super().__init__()
        self.N = N
        self.num_probes = num_probes

        A_coo = A.tocoo().astype(np.float32)
        indices = torch.tensor(
            np.vstack([A_coo.row, A_coo.col]), dtype=torch.long,
        )
        values = torch.tensor(A_coo.data, dtype=torch.float32)
        A_torch = torch.sparse_coo_tensor(
            indices, values, (A.shape[0], A.shape[1]),
        ).coalesce()
        self.register_buffer('A_torch', A_torch)

    def forward(self, model: nn.Module, device: torch.device) -> torch.Tensor:
        N = self.N
        K = self.num_probes

        w = torch.randn(K, 1, N + 2, N + 2, device=device)
        w[:, :, 0, :] = 0
        w[:, :, -1, :] = 0
        w[:, :, :, 0] = 0
        w[:, :, :, -1] = 0

        z = model(w)

        z_flat = z.reshape(K, N * N)

        Az = torch.sparse.mm(self.A_torch, z_flat.t()).t()

        w_interior = w[:, 0, 1:-1, 1:-1].reshape(K, N * N)

        diff = w_interior - Az
        loss = (diff * diff).sum(dim=1).mean()

        return loss
