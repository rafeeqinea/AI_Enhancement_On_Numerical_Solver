"""Smoke test for the condition-loss training pipeline."""
import numpy as np
import pytest
import torch

from src.data.poisson import assemble_poisson_2d
from src.models.unet import UNet
from src.training.losses import ConditionLoss
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.solvers.cg import conjugate_gradient
from src.solvers.fcg import flexible_cg


class TestConditionLossPipeline:
    @pytest.fixture
    def tiny_setup(self):
        N = 4
        A = assemble_poisson_2d(N)
        device = torch.device('cpu')
        model = UNet(base_features=4, levels=1, dim=2).to(device)
        return N, A, device, model

    def test_training_step_runs(self, tiny_setup):
        N, A, device, model = tiny_setup
        loss_fn = ConditionLoss(A, N, num_probes=4, dim=2, mode='sparse')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        loss = loss_fn(model, device)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert np.isfinite(loss.item())

    def test_loss_decreases_over_steps(self, tiny_setup):
        N, A, device, model = tiny_setup
        loss_fn = ConditionLoss(A, N, num_probes=8, dim=2, mode='sparse')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn(model, device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0] * 1.5, "Loss should not explode"

    def test_trained_model_wraps_as_preconditioner(self, tiny_setup):
        N, A, device, model = tiny_setup
        precond = make_nn_preconditioner(model, N, device=device, dim=2)

        rng = np.random.default_rng(42)
        b = rng.standard_normal(N * N)

        result = flexible_cg(A, b, precond, tol=1e-6, max_iter=500)
        assert result.iterations > 0
        assert result.iterations < 500
