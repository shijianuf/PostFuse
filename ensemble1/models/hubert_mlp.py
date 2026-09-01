"""Frozen-HuBERT utterance embedding + MLP head (exploratory SSL reference).

Not part of the main 11-model log-mel benchmark. Pretraining is frozen;
only this head is trained under the same split/metrics protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HubertMLP(nn.Module):
    """MLP classifier on fixed-dim SSL utterance features (e.g. 768-d HuBERT)."""

    def __init__(
        self,
        feat_dim: int = 768,
        num_classes: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feat_dim)
        return self.net(x)
