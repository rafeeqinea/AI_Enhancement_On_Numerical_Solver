from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, hidden_channels: int = 32, num_layers: int = 7) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        layers.append(nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out[:, :, 1:-1, 1:-1]
