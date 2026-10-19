"""Position embeddings."""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


def precompute_freqs_cis_2d(
    dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0
) -> torch.Tensor:
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


class NerfEmbedder(nn.Module):
    def __init__(
        self, in_channels: int, hidden_size_input: int, max_freqs: int
    ) -> None:
        super().__init__()
        self.max_freqs: int = max_freqs
        self.hidden_size_input: int = hidden_size_input
        self.embedder: nn.Sequential = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(
        self, patch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = precompute_freqs_cis_2d(
            self.max_freqs**2 * 2, patch_size, patch_size
        ).real
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, P2, C = inputs.shape
        patch_size = int(P2**0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
    pe_interpolation: float = 1.0,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """grid_size: int of the grid height and width"""
    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    else:
        pass
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """embed_dim: output dimension for each position"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def interpolate_pos_embed(
    model_path: str | Path, pe_key: str = "gen_pos_embed", new_len: int = 4096
) -> None:
    state_dict = torch.load(model_path, map_location="cpu")

    pos_embed_1d = state_dict[pe_key]
    _, ori_len, embed_dim = pos_embed_1d.shape

    ori_size = int(ori_len**0.5)
    new_size = int(new_len**0.5)

    if ori_size != new_size:
        logger.info(
            f"Position interpolate from {ori_size:d}x{ori_size:d} to {new_size:d}x{new_size:d}"
        )
        pos_embed_2d = pos_embed_1d.reshape(-1, ori_size, ori_size, embed_dim).permute(
            0, 3, 1, 2
        )
        pos_embed_2d = torch.nn.functional.interpolate(
            pos_embed_2d, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_embed_1d = pos_embed_2d.permute(0, 2, 3, 1).flatten(1, 2)
        state_dict[pe_key] = pos_embed_1d
    else:
        pass

    torch.save(state_dict, model_path)


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None:
        super().__init__()
        self.max_num_patch_per_side: int = max_num_patch_per_side
        self.hidden_size: int = hidden_size
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size), requires_grad=False
        )
        self._init_weights()

    def _init_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, self.max_num_patch_per_side
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]
