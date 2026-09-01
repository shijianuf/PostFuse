"""Log-mel feature extraction and datasets for CASIA baselines."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from erm.datasets.casia import CASIASample, ID2LABEL


def load_log_mel(
    wav_path: Path,
    *,
    sr: int = 16000,
    n_mels: int = 64,
    max_seconds: float | None = None,
) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    if max_seconds is not None and max_seconds > 0:
        max_len = int(sr * max_seconds)
        if len(y) > max_len:
            y = y[:max_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


SPEC128_N_MELS = 128
SPEC128_N_FRAMES = 128


def fix_spec_128x128(feat: np.ndarray) -> np.ndarray:
    """Crop/pad log-mel to (128, 128) for VLP-ViT / SepTr."""
    n_mels, n_frames = feat.shape
    out = np.zeros((SPEC128_N_MELS, SPEC128_N_FRAMES), dtype=np.float32)
    h = min(n_mels, SPEC128_N_MELS)
    w = min(n_frames, SPEC128_N_FRAMES)
    out[:h, :w] = feat[:h, :w]
    return out


def load_log_mel_128(
    wav_path: Path,
    *,
    sr: int = 16000,
    max_seconds: float = 8.0,
) -> np.ndarray:
    mel = load_log_mel(wav_path, sr=sr, n_mels=SPEC128_N_MELS, max_seconds=max_seconds)
    return fix_spec_128x128(mel)


class CASIAEmotionDataset(Dataset):
    def __init__(
        self,
        samples: list[CASIASample],
        *,
        sr: int = 16000,
        n_mels: int = 64,
        max_seconds: float | None = None,
    ):
        self.samples = samples
        self.sr = sr
        self.n_mels = n_mels
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        feat = load_log_mel(
            s.path, sr=self.sr, n_mels=self.n_mels, max_seconds=self.max_seconds
        )
        x = torch.from_numpy(feat).unsqueeze(0)
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y, s.emotion, s.speaker


class CASIAEmotionDataset128(Dataset):
    """Fixed 128×128 log-mel for VLP-ViT and SepTr."""

    def __init__(
        self,
        samples: list[CASIASample],
        *,
        sr: int = 16000,
        max_seconds: float = 8.0,
    ):
        self.samples = samples
        self.sr = sr
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        feat = load_log_mel_128(s.path, sr=self.sr, max_seconds=self.max_seconds)
        x = torch.from_numpy(feat).unsqueeze(0)
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y, s.emotion, s.speaker


def collate_batch(batch):
    xs, ys, emotions, speakers = zip(*batch)
    max_t = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        if x.shape[-1] < max_t:
            pad = torch.zeros(1, x.shape[1], max_t - x.shape[2], dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        padded.append(x)
    return torch.stack(padded), torch.stack(ys), list(emotions), list(speakers)


def collate_batch_fixed128(batch):
    xs, ys, emotions, speakers = zip(*batch)
    stacked = torch.stack(xs)
    if stacked.shape[-2] != SPEC128_N_MELS or stacked.shape[-1] != SPEC128_N_FRAMES:
        fixed = []
        for x in xs:
            arr = x.squeeze(0).numpy()
            fixed.append(torch.from_numpy(fix_spec_128x128(arr)).unsqueeze(0))
        stacked = torch.stack(fixed)
    return stacked, torch.stack(ys), list(emotions), list(speakers)


class CASIASequenceDataset(Dataset):
    """Frame-level log-mel (T, n_mels) for BiLSTM / ViT / GCN."""

    def __init__(
        self,
        samples: list[CASIASample],
        *,
        sr: int = 16000,
        n_mels: int = 64,
        max_seconds: float | None = 8.0,
    ):
        self.samples = samples
        self.sr = sr
        self.n_mels = n_mels
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        mel = load_log_mel(
            s.path, sr=self.sr, n_mels=self.n_mels, max_seconds=self.max_seconds
        )
        x = torch.from_numpy(mel.T.copy())
        y = torch.tensor(s.label, dtype=torch.long)
        return x, y, s.emotion, s.speaker


def collate_sequence_batch(batch):
    xs, ys, emotions, speakers = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    padded, masks = [], []
    for x in xs:
        t, d = x.shape
        pad_len = max_t - t
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, d, dtype=x.dtype)], dim=0)
        mask = torch.zeros(max_t, dtype=torch.bool)
        mask[:t] = True
        padded.append(x)
        masks.append(mask)
    return (
        torch.stack(padded),
        torch.stack(masks),
        torch.stack(ys),
        list(emotions),
        list(speakers),
    )


def evaluate(model, loader, device, *, emotion_labels: list[str] | None = None, forward_kwargs=None):
    model.eval()
    all_y, all_p = [], []
    forward_kwargs = forward_kwargs or {}
    if emotion_labels is None:
        emotion_labels = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, _, _ = batch
                x, y = x.to(device), y.to(device)
                logits = model(x, **forward_kwargs)
            else:
                x, mask, y, _, _ = batch
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits = model(x, mask, **forward_kwargs)
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_p.extend(pred.tolist())
            all_y.extend(y.cpu().numpy().tolist())

    from erm.training.metrics import compute_ser_metrics

    m = compute_ser_metrics(
        all_y,
        all_p,
        labels=emotion_labels,
    )
    return (
        m["test_accuracy"],
        m["test_weighted_f1"],
        m["test_macro_f1"],
        m["test_unweighted_accuracy"],
        m["report"],
    )
