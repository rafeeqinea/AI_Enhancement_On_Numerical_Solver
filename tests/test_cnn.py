import pytest
import torch
from src.models.cnn import BaselineCNN


class TestBaselineCNN:
    def test_output_shape(self):
        model = BaselineCNN()
        x = torch.randn(4, 1, 18, 18)
        y = model(x)
        assert y.shape == (4, 1, 16, 16)

    def test_output_shape_various_N(self):
        model = BaselineCNN()
        for N in [8, 16, 32, 64]:
            x = torch.randn(2, 1, N + 2, N + 2)
            y = model(x)
            assert y.shape == (2, 1, N, N)

    def test_parameter_count(self):
        model = BaselineCNN(hidden_channels=32, num_layers=7)
        total = sum(p.numel() for p in model.parameters())
        assert 10_000 < total < 200_000

    def test_gradients_flow(self):
        model = BaselineCNN()
        x = torch.randn(2, 1, 10, 10, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_deterministic(self):
        model = BaselineCNN()
        model.eval()
        x = torch.randn(1, 1, 18, 18)
        y1 = model(x)
        y2 = model(x)
        assert torch.allclose(y1, y2)

    def test_custom_channels(self):
        model = BaselineCNN(hidden_channels=64, num_layers=5)
        x = torch.randn(2, 1, 34, 34)
        y = model(x)
        assert y.shape == (2, 1, 32, 32)
