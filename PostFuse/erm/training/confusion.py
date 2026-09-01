"""ensemble baselines Confusion matrix"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


def plot_confusion_matrix(
    cm,
    *,
    title: str = "Confusion matrix",
    normalize: bool = False,
    cmap=plt.cm.Oranges,
    labels: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    labels = labels or [str(i) for i in range(cm.shape[0])]
    cm = np.asarray(cm)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] confusion matrix -> {save_path}")
    plt.close()


def collect_cnn_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=-1).cpu().numpy()
            all_p.extend(pred.tolist())
            all_y.extend(y.numpy().tolist())
    return all_y, all_p


def collect_sequence_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, mask, y, _, _ in loader:
            x, mask = x.to(device), mask.to(device)
            pred = model(x, mask).argmax(dim=-1).cpu().numpy()
            all_p.extend(pred.tolist())
            all_y.extend(y.numpy().tolist())
    return all_y, all_p


def collect_dialogue_predictions(model, dialogues, features, device):
    model.eval()
    logits_sum: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    labels: dict[int, int] = {}

    with torch.no_grad():
        for d in dialogues:
            idxs = [u.sample_idx for u in d.utterances]
            x = torch.from_numpy(features[idxs].copy()).float().to(device)
            logits = model(x)
            for i, u in enumerate(d.utterances):
                if u.sample_idx not in logits_sum:
                    logits_sum[u.sample_idx] = logits[i].cpu()
                    counts[u.sample_idx] = 1
                    labels[u.sample_idx] = u.label
                else:
                    logits_sum[u.sample_idx] += logits[i].cpu()
                    counts[u.sample_idx] += 1

    all_y, all_p = [], []
    for idx in sorted(labels.keys()):
        pred = (logits_sum[idx] / counts[idx]).argmax().item()
        all_y.append(labels[idx])
        all_p.append(pred)
    return all_y, all_p


def save_confusion_matrices(
    y_true,
    y_pred,
    labels: list[str],
    *,
    run_dir: Path,
    prefix: str = "test",
    method: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    data_dir = run_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    title_base = f"{method} — {prefix}" if method else prefix

    # Short names for transfer probes (prefix="cm"); keep long names for baselines.
    if prefix == "cm":
        csv_path = data_dir / "cm.csv"
        json_path = data_dir / "cm.json"
        png_path = figures_dir / "cm.png"
        png_norm_path = figures_dir / "cm_n.png"
    else:
        csv_path = data_dir / f"{prefix}_confusion_matrix.csv"
        json_path = data_dir / f"{prefix}_confusion_matrix.json"
        png_path = figures_dir / f"{prefix}_confusion_matrix.png"
        png_norm_path = figures_dir / f"{prefix}_confusion_matrix_normalized.png"

    np.savetxt(csv_path, cm, fmt="%d", delimiter=",")
    json_path.write_text(
        json.dumps(
            {
                "prefix": prefix,
                "method": method,
                "labels": labels,
                "matrix": cm.tolist(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] confusion matrix csv -> {csv_path}")
    print(f"[OK] confusion matrix json -> {json_path}")

    plot_confusion_matrix(
        cm,
        title=f"{title_base} confusion matrix",
        normalize=False,
        labels=labels,
        save_path=png_path,
    )
    plot_confusion_matrix(
        cm,
        title=f"{title_base} confusion matrix (normalized)",
        normalize=True,
        labels=labels,
        save_path=png_norm_path,
    )

    meta = {
        "prefix": prefix,
        "method": method,
        "labels": labels,
        "csv": str(csv_path.relative_to(run_dir)),
        "json": str(json_path.relative_to(run_dir)),
        "png": str(png_path.relative_to(run_dir)),
        "png_normalized": str(png_norm_path.relative_to(run_dir)),
    }
    return cm, meta
