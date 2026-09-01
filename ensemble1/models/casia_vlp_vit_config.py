"""Vertical Patch ViT student (128×128 log-mel, patch 128×1)."""
from __future__ import annotations

N_MELS = 128
N_FRAMES = 128
PATCH_MEL = 128
PATCH_TIME = 1
NUM_PATCHES = N_FRAMES // PATCH_TIME  # 128 vertical strips
PATCH_DIM = PATCH_MEL * PATCH_TIME
PROJECTION_DIM = 256
NUM_HEADS = 5
NUM_LAYERS = 3
MLP_DIM = 512
DROPOUT = 0.1
