"""Frame-level GCN on log-mel (utterance-level SER baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_utils import normalize_adj, sequential_edges


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(adj @ x))


class AudioGCN(nn.Module):
    """Each mel frame is a graph node; edges connect adjacent frames."""

    def __init__(self, n_mels: int = 64, hidden: int = 128, num_classes: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, hidden)
        self.gcn1 = GCNLayer(hidden, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        h = self.input_proj(x)
        edge_index = sequential_edges(n)
        adj = normalize_adj(edge_index.to(x.device), n, device=x.device)
        h = self.gcn1(h, adj)
        h = self.gcn2(h, adj)
        return h.mean(dim=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = []
        for i in range(x.size(0)):
            valid = x[i][mask[i]]
            logits.append(self.classifier(self._encode(valid)))
        return torch.stack(logits, dim=0)
