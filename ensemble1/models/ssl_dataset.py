"""Dataset helpers for frozen SSL feature-cache baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from erm.datasets.casia import CASIASample
from models.feature_cache import features_for_split, load_feature_cache, sample_cache_key


class CachedFeatureDataset(Dataset):
    """Utterance-level fixed vectors from a feature cache (HuBERT, etc.)."""

    def __init__(
        self,
        samples: list[CASIASample],
        features: np.ndarray,
        *,
        data_dir: Path | None = None,
    ):
        if features.ndim != 2 or len(features) != len(samples):
            raise ValueError(
                f"Expected features shape (N, D) aligned with samples; "
                f"got {getattr(features, 'shape', None)} vs N={len(samples)}"
            )
        self.samples = samples
        self.features = np.asarray(features, dtype=np.float32)
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y, s.emotion, s.speaker


def collate_feature_batch(batch):
    xs, ys, emotions, speakers = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(emotions), list(speakers)


def loaders_from_feat_cache(
    cache_path: Path,
    train: list[CASIASample],
    val: list[CASIASample],
    test: list[CASIASample],
    *,
    data_dir: Path,
    batch_size: int,
):
    cache_samples, features, _ = load_feature_cache(cache_path)
    train_x = features_for_split(train, cache_samples, features, data_dir=data_dir)
    val_x = features_for_split(val, cache_samples, features, data_dir=data_dir)
    test_x = features_for_split(test, cache_samples, features, data_dir=data_dir)

    train_ds = CachedFeatureDataset(train, train_x, data_dir=data_dir)
    val_ds = CachedFeatureDataset(val, val_x, data_dir=data_dir)
    test_ds = CachedFeatureDataset(test, test_x, data_dir=data_dir)
    return train_ds, val_ds, test_ds, int(features.shape[1])


def cache_key_preview(sample: CASIASample, data_dir: Path) -> str:
    return sample_cache_key(sample.path, data_dir=data_dir, emotion=sample.emotion)
