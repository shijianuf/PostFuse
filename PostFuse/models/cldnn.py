"""CLDNN: Sainath et al. ICASSP 2015 topology on CNN-style log-mel input."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.casia_lstm_config import DROPOUT, N_MELS, PROJ_DIM
from models.ser_preprocess import (
    AttentionPool,
    lengths_from_spec,
    pack_lstm_outputs,
    spec_frame_mask,
    utterance_normalize_spec,
)


class CLDNN(nn.Module):
    """
    Sainath et al., ICASSP 2015 Figure 1:
      CNN (×2) → dim-red linear → LSTM (×2) → DNN (×2)
      Skip (1): concat raw frames with dim-red features before LSTM
      Skip (2): concat dim-red features with LSTM outputs before DNN

    Input pipeline matches SER_CNN (`CASIAEmotionDataset`, (1, n_mels, T)).
    """

    def __init__(self, n_mels: int = N_MELS, num_classes: int = 6, lstm_hidden: int = 128):
        super().__init__()
        self.n_mels = n_mels
        self.input_norm = nn.LayerNorm(n_mels)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))

        freq_bins = n_mels // 4
        cnn_flat = 64 * freq_bins
        self.cnn_proj = nn.Linear(cnn_flat, lstm_hidden)

        # Skip (1): raw log-mel ∥ dim-red CNN features
        self.lstm = nn.LSTM(
            n_mels + lstm_hidden,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=DROPOUT,
        )

        self.pool = AttentionPool(lstm_hidden * 2)

        # Skip (2): dim-red ∥ LSTM frame outputs → 2-layer DNN
        self.dnn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, PROJ_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(PROJ_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = utterance_normalize_spec(x)
        raw_seq = self.input_norm(x.squeeze(1).transpose(1, 2))
        mask = spec_frame_mask(x)
        lengths = lengths_from_spec(x).cpu().long()

        h = self.pool1(F.relu(self.bn1(self.conv1(x))))
        h = self.pool2(F.relu(self.bn2(self.conv2(h))))

        b, c, f, t = h.shape
        cnn_seq = self.cnn_proj(h.permute(0, 3, 1, 2).reshape(b, t, c * f))

        # Skip (1)
        lstm_in = torch.cat([raw_seq, cnn_seq], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_in, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out, mask = pack_lstm_outputs(lstm_out, mask)

        # Skip (2)
        merged = torch.cat([cnn_seq, lstm_out], dim=-1)
        if merged.size(1) < mask.size(1):
            pad = mask.size(1) - merged.size(1)
            merged = F.pad(merged, (0, 0, 0, pad))
        elif merged.size(1) > mask.size(1):
            merged = merged[:, : mask.size(1)]

        pooled = self.pool(merged, mask)
        return self.dnn(pooled)
