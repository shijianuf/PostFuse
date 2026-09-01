"""Deterministic spatial pooling (avoids AdaptiveAvgPool2d CUDA backward).

torch.nn.AdaptiveAvgPool2d has no deterministic CUDA backward, which triggers
warnings under use_deterministic_algorithms(True, warn_only=True) and can make
same-seed CNN runs diverge via early-stopping checkpoint choice.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Adaptive average pool to a fixed (H, W) with mean reductions only."""

    def __init__(self, output_size: int | tuple[int, int]):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = (int(output_size[0]), int(output_size[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        oh, ow = self.output_size
        if oh <= 0 or ow <= 0:
            raise ValueError(f"Invalid output_size={self.output_size}")

        # Fast path when dims divide evenly (common for mel height after MaxPool).
        if h % oh == 0 and w % ow == 0:
            return (
                x.reshape(b, c, oh, h // oh, ow, w // ow)
                .mean(dim=5)
                .mean(dim=3)
            )

        # Match PyTorch adaptive region edges: floor(i*H/OH) .. ceil((i+1)*H/OH).
        outs = []
        for i in range(oh):
            h0 = int(math.floor(i * h / oh))
            h1 = int(math.ceil((i + 1) * h / oh))
            row = []
            for j in range(ow):
                w0 = int(math.floor(j * w / ow))
                w1 = int(math.ceil((j + 1) * w / ow))
                row.append(x[:, :, h0:h1, w0:w1].mean(dim=(-2, -1)))
            outs.append(torch.stack(row, dim=-1))
        return torch.stack(outs, dim=2)
