from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base_features: int = 32, levels: int = 4) -> None:
        super().__init__()
        self.levels = levels

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        ch = base_features
        in_ch = 1

        for _ in range(levels):
            self.encoders.append(ConvBlock(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch
            ch *= 2

        self.bottleneck = ConvBlock(in_ch, ch)

        for _ in range(levels):
            self.upconvs.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.decoders.append(ConvBlock(ch, ch // 2))
            ch //= 2

        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            dh = skip.size(2) - x.size(2)
            dw = skip.size(3) - x.size(3)
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.head(x)
        return x[:, :, 1:-1, 1:-1]
