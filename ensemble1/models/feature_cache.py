"""Portable feature-cache helpers for SSL downstream baselines."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from erm.datasets.casia import CASIASample


def sample_cache_key(
    path: Path | str,
    *,
    data_dir: Path | None = None,
    emotion: str | None = None,
) -> str:
    p = Path(path)
    if data_dir is not None:
        data_root = Path(data_dir).resolve()
        for candidate in (p, p.resolve()):
            try:
                return candidate.resolve().relative_to(data_root).as_posix()
            except ValueError:
                pass
        parts = p.as_posix().split("/")
        for i, part in enumerate(parts):
            if part.lower() == "casia" and i + 1 < len(parts):
                return "/".join(parts[i + 1 :])
    if p.is_file():
        return p.resolve().as_posix()
    if emotion is not None:
        return f"{emotion}/{p.name}"
    return p.as_posix()


def sample_to_cache_record(sample: CASIASample, data_dir: Path) -> dict:
    return {
        "path": sample_cache_key(sample.path, data_dir=data_dir, emotion=sample.emotion),
        "label": sample.label,
        "emotion": sample.emotion,
        "speaker": sample.speaker,
    }


def cache_keys_for_samples(
    samples: list[CASIASample],
    *,
    data_dir: Path | None = None,
) -> set[str]:
    return {
        sample_cache_key(s.path, data_dir=data_dir, emotion=s.emotion) for s in samples
    }


def cache_covers_samples(
    cache_samples: list[CASIASample],
    required_samples: list[CASIASample],
    *,
    data_dir: Path | None = None,
) -> bool:
    cached = cache_keys_for_samples(cache_samples, data_dir=data_dir)
    required = cache_keys_for_samples(required_samples, data_dir=data_dir)
    return required.issubset(cached)


def load_feature_cache(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    samples = [
        CASIASample(
            path=Path(s["path"]),
            label=s["label"],
            emotion=s["emotion"],
            speaker=s["speaker"],
        )
        for s in data["samples"]
    ]
    features = data["features"]
    if isinstance(features, np.ndarray) and features.ndim == 2:
        features = np.asarray(features, dtype=np.float32)
    return samples, features, data.get("lengths")


def features_for_split(
    split_samples: list[CASIASample],
    cache_samples,
    features,
    *,
    data_dir: Path | None = None,
    lengths: np.ndarray | None = None,
):
    path_to_row: dict[str, int] = {}
    for i, s in enumerate(cache_samples):
        key = sample_cache_key(s.path, data_dir=data_dir, emotion=s.emotion)
        path_to_row.setdefault(key, i)

    rows: list[int] = []
    missing: list[str] = []
    for s in split_samples:
        key = sample_cache_key(s.path, data_dir=data_dir, emotion=s.emotion)
        if key not in path_to_row:
            missing.append(key)
            continue
        rows.append(path_to_row[key])

    if missing:
        preview = "\n  ".join(missing[:5])
        extra = f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise RuntimeError(
            f"Feature cache is missing {len(missing)} sample(s). "
            f"Rebuild with --rebuild-feat-cache or run the feature extraction script.\n"
            f"Missing keys:\n  {preview}{extra}"
        )

    if isinstance(features, np.ndarray) and features.ndim == 2:
        return features[rows]

    split_feats = [features[i] for i in rows]
    split_lengths = None
    if lengths is not None:
        split_lengths = lengths[rows]
    return split_feats, split_lengths
