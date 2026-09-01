"""Plot training metrics vs epoch from epochs.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_epoch_records(epochs_path: Path) -> list[dict[str, Any]]:
    path = Path(epochs_path)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _best_epoch(records: list[dict[str, Any]]) -> int:
    epochs = [int(r["epoch"]) for r in records]
    val_f1 = [float(r["val_weighted_f1"]) for r in records]
    best_idx = max(range(len(val_f1)), key=lambda i: val_f1[i])
    return epochs[best_idx]


def plot_training_loss_curve(
    records: list[dict[str, Any]],
    *,
    save_path: Path,
    title: str = "Training loss",
    best_epoch: int | None = None,
) -> Path:
    if not records:
        raise ValueError("No epoch records to plot")

    epochs = [int(r["epoch"]) for r in records]
    losses = [float(r["train_loss"]) for r in records]
    if best_epoch is None:
        best_epoch = _best_epoch(records)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        epochs,
        losses,
        color="#2563eb",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="train loss",
    )
    ax.axvline(
        best_epoch,
        color="#94a3b8",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label=f"best @ {best_epoch}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_validation_metrics_curve(
    records: list[dict[str, Any]],
    *,
    save_path: Path,
    title: str = "Validation metrics",
    best_epoch: int | None = None,
) -> Path:
    if not records:
        raise ValueError("No epoch records to plot")

    epochs = [int(r["epoch"]) for r in records]
    val_f1 = [float(r["val_weighted_f1"]) for r in records]
    val_acc = [
        float(r["val_accuracy"]) if r.get("val_accuracy") is not None else None
        for r in records
    ]
    has_acc = any(v is not None for v in val_acc)
    if best_epoch is None:
        best_epoch = _best_epoch(records)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        epochs,
        val_f1,
        color="#16a34a",
        marker="s",
        markersize=4,
        linewidth=1.5,
        label="val weighted F1",
    )
    if has_acc:
        acc_vals = [v if v is not None else float("nan") for v in val_acc]
        ax.plot(
            epochs,
            acc_vals,
            color="#ea580c",
            marker="^",
            markersize=4,
            linewidth=1.5,
            label="val accuracy",
        )
    ax.axvline(
        best_epoch,
        color="#94a3b8",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label=f"best @ {best_epoch}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation metric")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_training_curves(
    records: list[dict[str, Any]],
    *,
    save_dir: Path,
    title_prefix: str = "Training",
    best_epoch: int | None = None,
) -> dict[str, Path]:
    """Save separate loss and validation metric figures."""
    if best_epoch is None and records:
        best_epoch = _best_epoch(records)

    save_dir = Path(save_dir)
    loss_path = plot_training_loss_curve(
        records,
        save_path=save_dir / "training_loss.png",
        title=f"{title_prefix} — train loss",
        best_epoch=best_epoch,
    )
    val_path = plot_validation_metrics_curve(
        records,
        save_path=save_dir / "validation_metrics.png",
        title=f"{title_prefix} — validation metrics",
        best_epoch=best_epoch,
    )
    return {"training_loss": loss_path, "validation_metrics": val_path}
