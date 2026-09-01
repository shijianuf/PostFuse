"""2-layer BiLSTM on frame-level log-mel (sequence SER baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.casia_lstm_config import DROPOUT, HIDDEN, N_MELS, PROJ_DIM
from models.ser_preprocess import AttentionPool, pack_lstm_outputs, utterance_normalize


class AudioBiLSTM(nn.Module):
    """
    Pipeline: utterance z-score → LayerNorm → linear proj → 2-layer BiLSTM → attention pool → classifier
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden: int = HIDDEN,
        num_layers: int = 2,
        proj_dim: int = PROJ_DIM,
        num_classes: int = 6,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_mels)
        self.proj = nn.Sequential(
            nn.Linear(n_mels, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            proj_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = utterance_normalize(x, mask)
        x = self.input_norm(x)
        x = self.proj(x)

        lengths = mask.sum(dim=1).clamp(min=1).cpu().long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs, mask = pack_lstm_outputs(outputs, mask)

        pooled = self.pool(outputs, mask)
        return self.classifier(pooled)
