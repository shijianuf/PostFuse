"""2-layer unidirectional RNN / GRU / LSTM on frame-level log-mel."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from models.casia_lstm_config import DROPOUT, HIDDEN, N_MELS, NUM_LAYERS, PROJ_DIM
from models.ser_preprocess import AttentionPool, pack_lstm_outputs, utterance_normalize

RNNType = Literal["rnn", "gru", "lstm"]
RNN_CELLS = {
    "rnn": nn.RNN,
    "gru": nn.GRU,
    "lstm": nn.LSTM,
}


class AudioSequenceRNN(nn.Module):
    """
    Shared pipeline for uni-directional sequence baselines:
    utterance z-score → LayerNorm → proj → 2-layer RNN/GRU/LSTM → attention pool → FC head
    """

    def __init__(
        self,
        rnn_type: RNNType,
        n_mels: int = N_MELS,
        hidden: int = HIDDEN,
        num_layers: int = NUM_LAYERS,
        proj_dim: int = PROJ_DIM,
        num_classes: int = 6,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if rnn_type not in RNN_CELLS:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn_type = rnn_type
        self.input_norm = nn.LayerNorm(n_mels)
        self.proj = nn.Sequential(
            nn.Linear(n_mels, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.rnn = RNN_CELLS[rnn_type](
            proj_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, proj_dim),
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
        outputs, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs, mask = pack_lstm_outputs(outputs, mask)

        pooled = self.pool(outputs, mask)
        return self.classifier(pooled)


class AudioLSTM(AudioSequenceRNN):
    def __init__(self, **kwargs):
        super().__init__(rnn_type="lstm", **kwargs)


class AudioGRU(AudioSequenceRNN):
    def __init__(self, **kwargs):
        super().__init__(rnn_type="gru", **kwargs)


class AudioRNN(AudioSequenceRNN):
    def __init__(self, **kwargs):
        super().__init__(rnn_type="rnn", **kwargs)
