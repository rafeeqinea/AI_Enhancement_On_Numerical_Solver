from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class ConditionLoss(nn.Module):
    """Hutchinson estimator of ||I - AM||^2_F.

    Works for both 2D (dim=2) and 3D (dim=3) grids.
    Supports AMP (autocast), gradient checkpointing, and probe accumulation
    for large grids.
    """

    def __init__(
        self,
        A: sp.spmatrix,
        N: int,
        num_probes: int = 32,
        dim: int = 2,
    ) -> None:
        super().__init__()
        self.N = N
        self.num_probes = num_probes
        self.dim = dim
        self.n_dof = N ** dim

        A_coo = A.tocoo().astype(np.float32)
        indices = torch.tensor(
            np.vstack([A_coo.row, A_coo.col]), dtype=torch.long,
        )
        values = torch.tensor(A_coo.data, dtype=torch.float32)
        A_torch = torch.sparse_coo_tensor(
            indices, values, (A.shape[0], A.shape[1]),
        ).coalesce()
        self.register_buffer('A_torch', A_torch)

    def _probe_shape(self) -> tuple[int, ...]:
        """Shape of a single probe: (1, N+2, N+2) for 2D, (1, N+2, N+2, N+2) for 3D."""
        return (1,) + (self.N + 2,) * self.dim

    def _boundary_mask(self, w: torch.Tensor) -> torch.Tensor:
        """Zero out boundary of probes in-place."""
        if self.dim == 2:
            w[:, :, 0, :] = 0
            w[:, :, -1, :] = 0
            w[:, :, :, 0] = 0
            w[:, :, :, -1] = 0
        else:
            w[:, :, 0, :, :] = 0
            w[:, :, -1, :, :] = 0
            w[:, :, :, 0, :] = 0
            w[:, :, :, -1, :] = 0
            w[:, :, :, :, 0] = 0
            w[:, :, :, :, -1] = 0
        return w

    def _extract_interior(self, w: torch.Tensor) -> torch.Tensor:
        """Extract interior DOFs from padded grid probe."""
        if self.dim == 2:
            return w[:, 0, 1:-1, 1:-1].reshape(w.size(0), self.n_dof)
        else:
            return w[:, 0, 1:-1, 1:-1, 1:-1].reshape(w.size(0), self.n_dof)

    def forward(
        self,
        model: nn.Module,
        device: torch.device,
        use_amp: bool = False,
        use_checkpointing: bool = False,
        probe_batch_size: int | None = None,
    ) -> torch.Tensor:
        N = self.N
        K = self.num_probes
        k_batch = probe_batch_size or K

        total_loss = torch.tensor(0.0, device=device)
        n_batches = 0

        for start in range(0, K, k_batch):
            k = min(k_batch, K - start)
            shape = (k,) + self._probe_shape()
            w = torch.randn(shape, device=device)
            w = self._boundary_mask(w)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    z = self._forward_model(model, w, use_checkpointing)
            else:
                z = self._forward_model(model, w, use_checkpointing)

            z_flat = z.reshape(k, self.n_dof).float()
            Az = torch.sparse.mm(self.A_torch, z_flat.t()).t()
            w_interior = self._extract_interior(w)
            diff = w_interior - Az
            batch_loss = (diff * diff).sum(dim=1).mean()
            total_loss = total_loss + batch_loss
            n_batches += 1

        return total_loss / n_batches

    def _forward_model(
        self, model: nn.Module, w: torch.Tensor, use_checkpointing: bool,
    ) -> torch.Tensor:
        if use_checkpointing:
            return grad_checkpoint(model, w, use_reentrant=False)
        return model(w)
