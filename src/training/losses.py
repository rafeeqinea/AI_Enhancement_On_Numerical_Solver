from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def _build_poisson_kernel(dim: int) -> torch.Tensor:
    """Build the Poisson stencil as a convolution kernel.

    2D: 5-point stencil [[0,-1,0],[-1,4,-1],[0,-1,0]]
    3D: 7-point stencil (center=6, face neighbors=-1)
    """
    if dim == 2:
        kernel = torch.tensor([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]], dtype=torch.float32)
        return kernel.reshape(1, 1, 3, 3)
    else:
        kernel = torch.zeros(3, 3, 3, dtype=torch.float32)
        kernel[1, 1, 1] = 6    # center
        kernel[0, 1, 1] = -1   # front
        kernel[2, 1, 1] = -1   # back
        kernel[1, 0, 1] = -1   # bottom
        kernel[1, 2, 1] = -1   # top
        kernel[1, 1, 0] = -1   # left
        kernel[1, 1, 2] = -1   # right
        return kernel.reshape(1, 1, 3, 3, 3)


class ConditionLoss(nn.Module):
    """Hutchinson estimator of ||I - AM||^2_F.

    Works for both 2D (dim=2) and 3D (dim=3) grids.
    Supports two modes:
      - 'sparse': uses torch.sparse.mm (original, memory-bound)
      - 'conv': uses F.conv2d/conv3d with Poisson stencil (fast, compute-bound)
    """

    def __init__(
        self,
        A: sp.spmatrix,
        N: int,
        num_probes: int = 32,
        dim: int = 2,
        mode: str = 'conv',
    ) -> None:
        super().__init__()
        self.N = N
        self.num_probes = num_probes
        self.dim = dim
        self.n_dof = N ** dim
        self.mode = mode

        if mode == 'sparse':
            A_coo = A.tocoo().astype(np.float32)
            indices = torch.tensor(
                np.vstack([A_coo.row, A_coo.col]), dtype=torch.long,
            )
            values = torch.tensor(A_coo.data, dtype=torch.float32)
            A_torch = torch.sparse_coo_tensor(
                indices, values, (A.shape[0], A.shape[1]),
            ).coalesce()
            self.register_buffer('A_torch', A_torch)
        else:
            # Conv mode: register the stencil kernel as a buffer
            kernel = _build_poisson_kernel(dim)
            self.register_buffer('stencil_kernel', kernel)

    def _probe_shape(self) -> tuple[int, ...]:
        return (1,) + (self.N + 2,) * self.dim

    def _boundary_mask(self, w: torch.Tensor) -> torch.Tensor:
        # The model works on zero-padded grids, so boundary probes must stay zero too.
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
        if self.dim == 2:
            return w[:, 0, 1:-1, 1:-1].reshape(w.size(0), self.n_dof)
        else:
            return w[:, 0, 1:-1, 1:-1, 1:-1].reshape(w.size(0), self.n_dof)

    def _apply_A_conv(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Poisson operator via convolution on the padded grid output.

        z comes from the model: shape (K, 1, N, N) for 2D or (K, 1, N, N, N) for 3D.
        We pad with zeros (Dirichlet BC), convolve with the stencil, extract interior.
        Result: A*z as a flat vector, same as sparse.mm but using dense conv.
        """
        N = self.N

        # Pad z with zeros (boundary conditions)
        if self.dim == 2:
            z_padded = F.pad(z, (1, 1, 1, 1), mode='constant', value=0)
            # Apply stencil: conv2d with padding=0 on already-padded input gives N×N output
            Az = F.conv2d(z_padded, self.stencil_kernel, padding=0)
        else:
            z_padded = F.pad(z, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
            Az = F.conv3d(z_padded, self.stencil_kernel, padding=0)

        return Az.reshape(z.size(0), self.n_dof)

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
            # Batch the Hutchinson probes so the loss fits on smaller GPUs.
            k = min(k_batch, K - start)
            shape = (k,) + self._probe_shape()
            w = torch.randn(shape, device=device)
            w = self._boundary_mask(w)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    z = self._forward_model(model, w, use_checkpointing)
            else:
                z = self._forward_model(model, w, use_checkpointing)

            if self.mode == 'conv':
                # z shape: (K, 1, N, N) or (K, 1, N, N, N) — direct from model
                Az = self._apply_A_conv(z.float())
            else:
                z_flat = z.reshape(k, self.n_dof).float()
                Az = torch.sparse.mm(self.A_torch, z_flat.t()).t()

            w_interior = self._extract_interior(w)
            diff = w_interior - Az
            # Hutchinson estimate of ||I - AM||^2_F from the sampled probes.
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
