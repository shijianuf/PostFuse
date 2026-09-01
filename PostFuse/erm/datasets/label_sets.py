"""Label-set helpers for cross-corpus probes (shared emotion intersection)."""
from __future__ import annotations

from dataclasses import replace

# Intersection of CASIA (6) and EmoDB (7), excluding surprise / boredom / disgust.
SHARED5_LABELS = ["angry", "fear", "happy", "neutral", "sad"]

LABEL_SETS: dict[str, list[str] | None] = {
    "full": None,  # use dataset-native labels
    "shared5": SHARED5_LABELS,
}


def resolve_emotion_labels(label_set: str, dataset_labels: list[str]) -> list[str]:
    key = (label_set or "full").lower().strip()
    if key not in LABEL_SETS:
        raise ValueError(f"Unknown label_set {label_set!r}; choose from {sorted(LABEL_SETS)}")
    custom = LABEL_SETS[key]
    return list(dataset_labels) if custom is None else list(custom)


def filter_and_remap_samples(samples: list, emotion_labels: list[str]) -> list:
    """Keep samples whose emotion is in emotion_labels; remap .label to 0..C-1."""
    name2id = {name: i for i, name in enumerate(emotion_labels)}
    out = []
    for s in samples:
        emo = getattr(s, "emotion", None)
        if emo not in name2id:
            continue
        out.append(replace(s, label=name2id[emo]))
    if not out:
        raise ValueError(f"No samples left after filtering to {emotion_labels}")
    return out
