"""CASIA Chinese emotional speech corpus loader."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID = {name: i for i, name in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(EMOTION_LABELS)}


@dataclass
class CASIASample:
    path: Path
    label: int
    emotion: str
    speaker: str


def discover_casia_samples(data_dir: Path) -> list[CASIASample]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"CASIA directory not found: {data_dir}")

    samples: list[CASIASample] = []
    for emotion_dir in sorted(data_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name.lower()
        if emotion not in LABEL2ID:
            continue
        for wav in sorted(emotion_dir.glob("*.wav")):
            speaker = wav.stem.split("_")[0]
            samples.append(
                CASIASample(
                    path=wav,
                    label=LABEL2ID[emotion],
                    emotion=emotion,
                    speaker=speaker,
                )
            )

    if not samples:
        raise FileNotFoundError(
            f"No wav files under {data_dir}; expected CASIA/{{angry,fear,...}}/*.wav"
        )
    return samples


RANDOM_SPLIT_TRAIN_RATIO = 0.7
RANDOM_SPLIT_VAL_RATIO = 0.1
SPEAKER_INDEP_TRAIN_RATIO = 0.8
SPEAKER_INDEP_VAL_RATIO = 0.1


def split_samples(
    samples: list[CASIASample],
    *,
    seed: int = 42,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    speaker_independent: bool = False,
    test_speaker: str | None = None,
) -> tuple[list[CASIASample], list[CASIASample], list[CASIASample]]:
    if speaker_independent:
        tr = SPEAKER_INDEP_TRAIN_RATIO if train_ratio is None else train_ratio
        vr = SPEAKER_INDEP_VAL_RATIO if val_ratio is None else val_ratio
        speakers = sorted({s.speaker for s in samples})
        if test_speaker is None:
            test_speaker = speakers[-1]
        test_set = [s for s in samples if s.speaker == test_speaker]
        rest = [s for s in samples if s.speaker != test_speaker]
        random.Random(seed).shuffle(rest)
        n_val = max(1, int(len(rest) * vr / (tr + vr)))
        val_set = rest[:n_val]
        train_set = rest[n_val:]
        return train_set, val_set, test_set

    tr = RANDOM_SPLIT_TRAIN_RATIO if train_ratio is None else train_ratio
    vr = RANDOM_SPLIT_VAL_RATIO if val_ratio is None else val_ratio

    rng = random.Random(seed)
    by_label: dict[int, list[CASIASample]] = {i: [] for i in range(len(EMOTION_LABELS))}
    for s in samples:
        by_label[s.label].append(s)

    train_set, val_set, test_set = [], [], []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_train = int(n * tr)
        n_val = max(1, int(n * vr))
        train_set.extend(label_samples[:n_train])
        val_set.extend(label_samples[n_train : n_train + n_val])
        test_set.extend(label_samples[n_train + n_val :])
    return train_set, val_set, test_set
