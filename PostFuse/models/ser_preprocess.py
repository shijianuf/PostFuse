"""Shared log-mel preprocessing for sequence SER models (LSTM, CLDNN, …)."""
from __future__ import annotations

import torch
import torch.nn as nn


def utterance_normalize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-utterance z-score over valid frames (SER-style feature scaling)."""
    m = mask.unsqueeze(-1).float()
    count = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * m).sum(dim=1, keepdim=True) / count
    var = ((x - mean).pow(2) * m).sum(dim=1, keepdim=True) / count
    return (x - mean) / var.sqrt().clamp(min=1e-5)


class AttentionPool(nn.Module):
    """Masked soft attention over frame outputs (bc-LSTM-style pooling)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D), mask: (B, T) True = valid
        logits = self.score(torch.tanh(h)).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=1)
        return (h * alpha.unsqueeze(-1)).sum(dim=1)


def pack_lstm_outputs(
    outputs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align LSTM pad_packed outputs with the input mask length."""
    if outputs.size(1) < mask.size(1):
        pad = mask.size(1) - outputs.size(1)
        outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad))
    elif outputs.size(1) > mask.size(1):
        outputs = outputs[:, : mask.size(1)]
        mask = mask[:, : outputs.size(1)]
    return outputs, mask


def spec_frame_mask(x: torch.Tensor) -> torch.Tensor:
    """True for valid time frames in log-mel (B, 1, F, T) or (B, F, T)."""
    if x.dim() == 4:
        spec = x.squeeze(1)
    else:
        spec = x
    return spec.abs().sum(dim=1) > 0


def utterance_normalize_spec(x: torch.Tensor) -> torch.Tensor:
    """Per-utterance z-score for CNN-style log-mel (B, 1, F, T) or (B, F, T)."""
    squeeze = x.dim() == 4
    spec = x.squeeze(1) if squeeze else x
    mask = spec_frame_mask(spec)
    normed = utterance_normalize(spec.transpose(1, 2), mask).transpose(1, 2)
    return normed.unsqueeze(1) if squeeze else normed


def lengths_from_spec(x: torch.Tensor) -> torch.Tensor:
    """Valid frame counts for padded log-mel spectrograms."""
    return spec_frame_mask(x).sum(dim=1).clamp(min=1)
