"""DenseNet-style 2D-CNN on log-mel (Huang et al., CVPR 2017 topology, CASIA-scaled)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deterministic_pool import DeterministicAdaptiveAvgPool2d


class _DenseLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.relu(self.norm(x), inplace=True))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_features: int, growth_rate: int):
        super().__init__()
        layers = []
        features = in_features
        for _ in range(num_layers):
            layers.append(_DenseLayer(features, growth_rate))
            features += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, out_features, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x), inplace=True))
        return self.pool(x)


class SER_DenseNet(nn.Module):
    """Compact DenseNet on single-channel log-mel; same input pipeline as SER_CNN."""

    def __init__(
        self,
        num_classes: int = 6,
        growth_rate: int = 24,
        block_layers: tuple[int, ...] = (4, 4, 4),
        num_init_features: int = 32,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features
        blocks = []
        for i, n_layers in enumerate(block_layers):
            blocks.append(_DenseBlock(n_layers, num_features, growth_rate))
            num_features += n_layers * growth_rate
            if i != len(block_layers) - 1:
                out_features = num_features // 2
                blocks.append(_Transition(num_features, out_features))
                num_features = out_features
        self.dense = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(num_features)
        self.pool = DeterministicAdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dense(x)
        x = F.relu(self.norm(x), inplace=True)
        x = self.pool(x)
        return self.classifier(x)
