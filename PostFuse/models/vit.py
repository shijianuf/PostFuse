"""ViT on log-mel for CASIA SER."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.casia_vit_config import (
    DROPOUT,
    MAX_TIME_FRAMES,
    N_MEL_PATCHES,
    N_MELS,
    N_TIME_PATCHES,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_PATCHES,
    PATCH_DIM,
    PATCH_MEL,
    PATCH_TIME,
    PROJECTION_DIM,
    VIT_HEAD_UNITS,
    VIT_MLP_HIDDEN,
)


def fix_spec_length(
    x: torch.Tensor, mask: torch.Tensor, max_frames: int = MAX_TIME_FRAMES
) -> tuple[torch.Tensor, torch.Tensor]:
    t = x.size(1)
    if t > max_frames:
        x = x[:, :max_frames]
        mask = mask[:, :max_frames]
    elif t < max_frames:
        pad = max_frames - t
        x = F.pad(x, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad), value=False)
    return x, mask


class ViTBlock(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        num_heads: int,
        mlp_hidden: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            projection_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, projection_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FixedSpectrogramPatches(nn.Module):
    """Fixed 8×16 patch grid on 64×160 log-mel."""

    def __init__(
        self,
        patch_mel: int = PATCH_MEL,
        patch_time: int = PATCH_TIME,
        max_frames: int = MAX_TIME_FRAMES,
    ):
        super().__init__()
        self.patch_mel = patch_mel
        self.patch_time = patch_time
        self.max_frames = max_frames
        self.n_time_patches = max_frames // patch_time

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = fix_spec_length(x, mask, self.max_frames)
        b = x.size(0)
        spec = x.transpose(1, 2)
        patch_dim = self.patch_mel * self.patch_time

        patches = spec.unfold(1, self.patch_mel, self.patch_mel).unfold(
            2, self.patch_time, self.patch_time
        )
        patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(b, NUM_PATCHES, patch_dim)

        frame_mask = mask.view(b, self.n_time_patches, self.patch_time)
        time_valid = frame_mask.all(dim=2)
        patch_valid = (
            time_valid.unsqueeze(1)
            .expand(b, N_MELS // self.patch_mel, self.n_time_patches)
            .reshape(b, NUM_PATCHES)
            .bool()
        )
        return patches, patch_valid


class PatchEncoder2D(nn.Module):
    def __init__(self, projection_dim: int, patch_dim: int = PATCH_DIM):
        super().__init__()
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.mel_pos = nn.Embedding(N_MEL_PATCHES, projection_dim)
        self.time_pos = nn.Embedding(N_TIME_PATCHES, projection_dim)
        mel_idx = torch.arange(NUM_PATCHES) // N_TIME_PATCHES
        time_idx = torch.arange(NUM_PATCHES) % N_TIME_PATCHES
        self.register_buffer("_mel_idx", mel_idx, persistent=False)
        self.register_buffer("_time_idx", time_idx, persistent=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        pos = self.mel_pos(self._mel_idx) + self.time_pos(self._time_idx)
        return self.projection(patches) + pos.unsqueeze(0)


class LogMelViT(nn.Module):
    """
    ViT on log-mel with flatten classification head.

    Patch grid: 8×10 on 64×160 → 128 tokens (dim 64).
    Block MLP: 64→256→64; head: flatten → 256 → 6.
    """

    def __init__(
        self,
        num_classes: int = 6,
        projection_dim: int = PROJECTION_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        mlp_hidden: int = VIT_MLP_HIDDEN,
        mlp_head_units: tuple[int, ...] = VIT_HEAD_UNITS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.patches = FixedSpectrogramPatches()
        self.patch_encoder = PatchEncoder2D(projection_dim)
        self.blocks = nn.ModuleList(
            [ViTBlock(projection_dim, num_heads, mlp_hidden, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)

        flat_dim = NUM_PATCHES * projection_dim
        head: list[nn.Module] = [nn.Flatten()]
        in_dim = flat_dim
        for units in mlp_head_units:
            head.extend([nn.Linear(in_dim, units), nn.GELU()])
            in_dim = units
        head.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        patches, patch_valid = self.patches(x, mask)
        tokens = self.patch_encoder(patches)

        pad_mask: torch.Tensor | None = None
        if not patch_valid.all():
            pad_mask = ~patch_valid

        h = tokens
        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)
        h = self.norm(h)

        if not patch_valid.all():
            h = h * patch_valid.unsqueeze(-1).float()
        return self.head(h)
