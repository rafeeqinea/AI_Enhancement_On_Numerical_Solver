import pytest
import torch
from src.models.unet import UNet


class TestUNet:
    def test_output_shape(self):
        model = UNet(base_features=16, levels=3)
        x = torch.randn(2, 1, 18, 18)
        y = model(x)
        assert y.shape == (2, 1, 16, 16)

    def test_output_shape_various_N(self):
        model = UNet(base_features=16, levels=3)
        for N in [16, 32, 64]:
            x = torch.randn(2, 1, N + 2, N + 2)
            y = model(x)
            assert y.shape == (2, 1, N, N), f'Failed for N={N}'

    def test_larger_model(self):
        model = UNet(base_features=32, levels=4)
        x = torch.randn(1, 1, 66, 66)
        y = model(x)
        assert y.shape == (1, 1, 64, 64)

    def test_parameter_count_larger_than_cnn(self):
        from src.models.cnn import BaselineCNN
        cnn = BaselineCNN()
        unet = UNet(base_features=32, levels=4)
        cnn_params = sum(p.numel() for p in cnn.parameters())
        unet_params = sum(p.numel() for p in unet.parameters())
        assert unet_params > cnn_params * 5

    def test_skip_connections_matter(self):
        model = UNet(base_features=16, levels=3)
        model.eval()
        x1 = torch.randn(1, 1, 10, 10)
        x2 = x1.clone()
        x2[0, 0, 1, 1] += 0.1
        y1 = model(x1)
        y2 = model(x2)
        assert not torch.allclose(y1, y2)

    def test_gradients_flow(self):
        model = UNet(base_features=16, levels=3)
        x = torch.randn(2, 1, 18, 18, requires_grad=True)
        y = model(x)
        y.sum().backward()
        assert x.grad is not None

    def test_batch_norm_present(self):
        model = UNet(base_features=16, levels=3)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_layers) > 0
