import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.poisson import assemble_poisson_2d
from src.training.losses import ConditionLoss
from src.models.unet import UNet


@pytest.fixture
def small_system():
    N = 4
    A = assemble_poisson_2d(N)
    return A, N


class TestConditionLoss:
    def test_positive_output(self, small_system):
        A, N = small_system
        model = UNet(base_features=4, levels=1)
        loss_fn = ConditionLoss(A, N, num_probes=8)
        device = torch.device('cpu')
        loss = loss_fn(model, device)
        assert loss.item() > 0

    def test_gradient_flows(self, small_system):
        A, N = small_system
        model = UNet(base_features=4, levels=1)
        loss_fn = ConditionLoss(A, N, num_probes=8)
        device = torch.device('cpu')
        loss = loss_fn(model, device)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad

    def test_probe_count_affects_variance(self, small_system):
        A, N = small_system
        model = UNet(base_features=4, levels=1)
        device = torch.device('cpu')

        loss_fn_low = ConditionLoss(A, N, num_probes=2)
        loss_fn_high = ConditionLoss(A, N, num_probes=64)

        torch.manual_seed(0)
        losses_low = [loss_fn_low(model, device).item() for _ in range(10)]
        torch.manual_seed(0)
        losses_high = [loss_fn_high(model, device).item() for _ in range(10)]

        std_low = np.std(losses_low)
        std_high = np.std(losses_high)
        assert std_high <= std_low * 2

    def test_sparse_matrix_shape(self, small_system):
        A, N = small_system
        loss_fn = ConditionLoss(A, N, num_probes=4)
        assert loss_fn.A_torch.shape == (N * N, N * N)

    def test_zero_model_gives_large_loss(self, small_system):
        A, N = small_system

        class ZeroModel(nn.Module):
            def forward(self, x):
                return x[:, :, 1:-1, 1:-1] * 0

        model = ZeroModel()
        loss_fn = ConditionLoss(A, N, num_probes=32)
        device = torch.device('cpu')
        loss = loss_fn(model, device)
        assert loss.item() > 1.0
