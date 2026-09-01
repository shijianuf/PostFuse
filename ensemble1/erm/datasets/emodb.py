"""Berlin EmoDB (EMO-DB) loader — second-corpus check (German acted SER).

Filename (official): SStteeVv.wav
  SS = speaker id (03, 08, 09, 10, 11, 12, 13, 14, 15, 16)
  tt = text code (a01, a02, …)
  e  = emotion letter:
       W anger (Wut), L boredom (Langeweile), E disgust (Ekel),
       A fear/anxiety (Angst), F happiness (Freude), T sadness (Trauer),
       N neutral
  V  = version letter (a, b, c, …)

Default layout after upload:
  database/EmoDB/EmoDB Dataset_wav_datasets/*.wav
Also accepts: database/EmoDB/**/*.wav and emotion-folder layouts.

Split: stratified random 7:1:2 (same ratios as CASIA). Seeds {10,20,30,40,50}.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

# Native 7-class EmoDB set (German labels mapped to English keys).
EMOTION_LABELS = [
    "angry",
    "boredom",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]
LABEL2ID = {name: i for i, name in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(EMOTION_LABELS)}

EMODB_LETTER2EMOTION = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "N": "neutral",
    "T": "sad",
}

_FOLDER_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "wut": "angry",
    "boredom": "boredom",
    "bored": "boredom",
    "langeweile": "boredom",
    "disgust": "disgust",
    "ekel": "disgust",
    "fear": "fear",
    "anxiety": "fear",
    "angst": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "freude": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "trauer": "sad",
}

RANDOM_SPLIT_TRAIN_RATIO = 0.7
RANDOM_SPLIT_VAL_RATIO = 0.1
SPEAKER_INDEP_TRAIN_RATIO = 0.8
SPEAKER_INDEP_VAL_RATIO = 0.1

_EMODB_NAME_RE = re.compile(
    r"^(?P<speaker>\d{2})(?P<text>[a-z]\d{2})(?P<emot>[WLEAFTN])(?P<ver>[a-z])$",
    re.IGNORECASE,
)


@dataclass
class EmoDBSample:
    path: Path
    label: int
    emotion: str
    speaker: str


def parse_emodb_stem(stem: str) -> tuple[str, str] | None:
    """Return (speaker, emotion_name) or None."""
    m = _EMODB_NAME_RE.match(stem.strip())
    if not m:
        return None
    letter = m.group("emot").upper()
    emotion = EMODB_LETTER2EMOTION.get(letter)
    if emotion is None:
        return None
    return m.group("speaker"), emotion


def _wav_roots(data_dir: Path) -> list[Path]:
    candidates = [
        data_dir / "EmoDB Dataset_wav_datasets",
        data_dir / "wav",
        data_dir / "AudioWAV",
        data_dir,
    ]
    return [p for p in candidates if p.is_dir()]


def discover_emodb_samples(data_dir: Path) -> list[EmoDBSample]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"EmoDB directory not found: {data_dir}\n"
            "Expected database/EmoDB/EmoDB Dataset_wav_datasets/*.wav"
        )

    samples: list[EmoDBSample] = []
    seen: set[str] = set()

    for root in _wav_roots(data_dir):
        for wav in sorted(root.rglob("*.wav")):
            parsed = parse_emodb_stem(wav.stem)
            if parsed is None:
                continue
            speaker, emotion = parsed
            key = str(wav.resolve())
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                EmoDBSample(
                    path=wav,
                    label=LABEL2ID[emotion],
                    emotion=emotion,
                    speaker=speaker,
                )
            )

    if not samples:
        # Emotion-folder fallback
        for emotion_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            emotion = _FOLDER_ALIASES.get(emotion_dir.name.lower())
            if emotion is None:
                continue
            for wav in sorted(emotion_dir.glob("*.wav")):
                key = str(wav.resolve())
                if key in seen:
                    continue
                seen.add(key)
                speaker = wav.stem[:2] if len(wav.stem) >= 2 else wav.stem.split("_")[0]
                samples.append(
                    EmoDBSample(
                        path=wav,
                        label=LABEL2ID[emotion],
                        emotion=emotion,
                        speaker=speaker,
                    )
                )

    if not samples:
        raise FileNotFoundError(
            f"No EmoDB wav files under {data_dir}. "
            "Expected names like 03a01Fa.wav under EmoDB Dataset_wav_datasets/."
        )
    return samples


def split_samples(
    samples: list[EmoDBSample],
    *,
    seed: int = 42,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    speaker_independent: bool = False,
    test_speaker: str | None = None,
) -> tuple[list[EmoDBSample], list[EmoDBSample], list[EmoDBSample]]:
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
    by_label: dict[int, list[EmoDBSample]] = {i: [] for i in range(len(EMOTION_LABELS))}
    for s in samples:
        by_label[s.label].append(s)

    train_set, val_set, test_set = [], [], []
    for label_samples in by_label.values():
        if not label_samples:
            continue
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_train = int(n * tr)
        n_val = max(1, int(n * vr)) if n > 1 else 0
        if n_train + n_val >= n and n > 2:
            n_val = max(1, n - n_train - 1)
        train_set.extend(label_samples[:n_train])
        val_set.extend(label_samples[n_train : n_train + n_val])
        test_set.extend(label_samples[n_train + n_val :])
    if not train_set or not val_set or not test_set:
        raise ValueError(
            f"EmoDB stratified split empty; n_samples={len(samples)} seed={seed}"
        )
    return train_set, val_set, test_set
