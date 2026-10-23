"""Image decoders."""

import torch
import torch.nn as nn

# Modified for SGLang; see this directory's README.md for upstream source.


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.block[2].weight)
        nn.init.zeros_(self.block[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PostConvSmoother(nn.Module):
    def __init__(
        self, in_channels: int = 3, hidden_channels: int = 64, num_blocks: int = 3
    ) -> None:
        super().__init__()
        self.in_proj: nn.Conv2d = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.blocks: nn.Sequential = nn.Sequential(
            *[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.out_proj: nn.Conv2d = nn.Conv2d(
            hidden_channels, in_channels, kernel_size=1
        )

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        return x + self.out_proj(h)


class ProgressiveConvDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 4096, out_channels: int = 3) -> None:
        super().__init__()

        self.up_blocks: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(hidden_dim, 512, kernel_size=3, padding=1),
                    nn.GroupNorm(32, 512),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.GroupNorm(32, 256),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(256, 64, kernel_size=3, padding=1),
                    nn.GroupNorm(32, 64),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(16, 32),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
            ]
        )

        self.out_conv: nn.Conv2d = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        x = x_2d
        for block in self.up_blocks:
            x = block(x)

        out = self.out_conv(x)
        return out


class PatchDecoderPostPixelShuffle(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(4096, 4096, kernel_size=3, padding=1)
        self.ps1: nn.PixelShuffle = nn.PixelShuffle(4)
        self.act1: nn.GELU = nn.GELU()

        self.conv2: nn.Conv2d = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.ps2: nn.PixelShuffle = nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ps1(self.act1(self.conv1(x)))
        x = self.ps2(self.conv2(x))
        return x


class PatchDecoderPrePixelShuffle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ps1: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv1: nn.Conv2d = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.act1: nn.GELU = nn.GELU()

        self.ps2: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv2: nn.Conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act2: nn.GELU = nn.GELU()

        self.ps3: nn.PixelShuffle = nn.PixelShuffle(8)
        self.conv3: nn.Conv2d = nn.Conv2d(4, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(self.ps1((x))))
        x = self.act2(self.conv2(self.ps2((x))))
        x = self.conv3(self.ps3((x)))
        return x


class PatchDecoderPrePixelShuffleSingleProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ps1: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv1: nn.Conv2d = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.act1: nn.GELU = nn.GELU()

        self.ps2: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv2: nn.Conv2d = nn.Conv2d(256, 192, kernel_size=3, padding=1)

        self.ps3: nn.PixelShuffle = nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(self.ps1((x))))
        x = self.ps3(self.conv2(self.ps2((x))))
        return x


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.ps1: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv1: nn.Conv2d = nn.Conv2d(
            input_dim // 4, hidden_dim, kernel_size=3, padding=1
        )
        self.act1: nn.GELU = nn.GELU()

        self.ps2: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv2: nn.Conv2d = nn.Conv2d(
            hidden_dim // 4, 192, kernel_size=3, padding=1
        )

        self.ps3: nn.PixelShuffle = nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(self.ps1((x))))
        x = self.ps3(self.conv2(self.ps2((x))))
        return x


# note (Wes): Old imports and pickles still refer to decoders by these legacy names.
PatchDecoder_postps = PatchDecoderPostPixelShuffle
PatchDecoder_preps = PatchDecoderPrePixelShuffle
PatchDecoder_preps1 = PatchDecoderPrePixelShuffleSingleProjection
