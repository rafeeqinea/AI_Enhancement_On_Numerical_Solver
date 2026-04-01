from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_layers(dim: int):
    """Return (Conv, BN, Pool, UpConv) classes for the given spatial dimension."""
    if dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.ConvTranspose2d
    elif dim == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, nn.ConvTranspose3d
    raise ValueError(f"dim must be 2 or 3, got {dim}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int = 2) -> None:
        super().__init__()
        Conv, BN, _, _ = _conv_layers(dim)
        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1),
            BN(out_ch),
            nn.ReLU(inplace=True),
            Conv(out_ch, out_ch, 3, padding=1),
            BN(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base_features: int = 32, levels: int = 4, dim: int = 2) -> None:
        super().__init__()
        self.levels = levels
        self.dim = dim

        _, _, Pool, UpConv = _conv_layers(dim)
        Conv, _, _, _ = _conv_layers(dim)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        ch = base_features
        in_ch = 1

        for _ in range(levels):
            self.encoders.append(ConvBlock(in_ch, ch, dim=dim))
            self.pools.append(Pool(2))
            in_ch = ch
            ch *= 2

        self.bottleneck = ConvBlock(in_ch, ch, dim=dim)

        for _ in range(levels):
            self.upconvs.append(UpConv(ch, ch // 2, 2, stride=2))
            self.decoders.append(ConvBlock(ch, ch // 2, dim=dim))
            ch //= 2

        self.head = Conv(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Pad to match skip connection size — works for any dimension
            pad_sizes = []
            for d in range(self.dim - 1, -1, -1):
                diff = skip.size(d + 2) - x.size(d + 2)
                pad_sizes.extend([diff // 2, diff - diff // 2])
            x = F.pad(x, pad_sizes)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.head(x)

        # Strip boundary padding: x[:, :, 1:-1, 1:-1] for 2D, x[:, :, 1:-1, 1:-1, 1:-1] for 3D
        slices = [slice(None), slice(None)] + [slice(1, -1)] * self.dim
        return x[tuple(slices)]
