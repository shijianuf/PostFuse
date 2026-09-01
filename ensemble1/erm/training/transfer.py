"""Lightweight transfer-probe helpers (shared5 Source-only / Head-FT / Full-FT)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from erm.datasets.registry import (
    _canonical_dataset,
    member_search_roots,
)
from erm.training.run_logger import load_torch_checkpoint

# Attribute name of the classification head per model key.
HEAD_ATTR: dict[str, str] = {
    "cnn": "classifier",
    "resnet": "classifier",
    "densenet": "classifier",
    "bilstm": "classifier",
    "lstm": "classifier",
    "gru": "classifier",
    "rnn": "classifier",
    "gcn": "classifier",
    "vlp_vit": "mlp_head",
    "vit": "head",
}


def head_module(model: nn.Module, model_key: str) -> nn.Module:
    attr = HEAD_ATTR.get(model_key)
    if attr is None or not hasattr(model, attr):
        raise ValueError(f"No registered classification head for model '{model_key}'")
    return getattr(model, attr)


def head_parameter_ids(model: nn.Module, model_key: str) -> set[int]:
    return {id(p) for p in head_module(model, model_key).parameters()}


def freeze_backbone(model: nn.Module, model_key: str) -> None:
    head_ids = head_parameter_ids(model, model_key)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def _cfg_dataset_key(cfg: dict) -> str:
    hp = cfg.get("hyperparams") or {}
    data = cfg.get("data") or cfg.get("data_info") or {}
    raw = str(
        hp.get("dataset_key")
        or data.get("dataset_key")
        or hp.get("dataset")
        or data.get("dataset")
        or ""
    ).strip()
    data_dir = str(hp.get("data_dir") or data.get("data_dir") or "").lower()
    if raw:
        try:
            return _canonical_dataset(raw)
        except Exception:
            pass
    if "emodb" in data_dir:
        return "emodb"
    return "casia"


def _cfg_label_set(cfg: dict) -> str:
    hp = cfg.get("hyperparams") or {}
    data = cfg.get("data") or {}
    return str(hp.get("label_set") or data.get("label_set") or "full").lower()


def shared5_scratch_run_name(seed: int, *, speaker_independent: bool = False) -> str:
    if speaker_independent:
        return f"si_s{seed}_e5"
    return f"seed{seed}_rand_e5"


def find_scratch_source_run(
    results_root: Path,
    model: str,
    *,
    source_dataset: str,
    seed: int,
    speaker_independent: bool = False,
    label_set: str = "shared5",
) -> Path | None:
    """Locate an in-domain scratch run for the given label_set (never a probe dir)."""
    want_ds = _canonical_dataset(source_dataset)
    want_ls = (label_set or "full").lower()
    if want_ls == "shared5":
        want_name = shared5_scratch_run_name(
            seed, speaker_independent=speaker_independent
        )
    else:
        from erm.datasets.registry import default_baseline_run_name

        want_name = default_baseline_run_name(
            source_dataset, seed, speaker_independent=speaker_independent
        )

    candidates: list[tuple[float, Path]] = []
    for base in member_search_roots(results_root, want_ds):
        exp_dir = base / model
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            # Skip transfer-probe dirs (legacy *_src* and short c2e5_/e2c5_).
            if "_src" in name or "_c2e" in name or "_e2c" in name:
                continue
            if not name.endswith(want_name):
                continue
            cfg_path = run_dir / "config.json"
            ckpt = run_dir / "model" / "best_model.pt"
            if not cfg_path.is_file() or not ckpt.is_file():
                continue
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if _cfg_dataset_key(cfg) != want_ds:
                continue
            if _cfg_label_set(cfg) != want_ls:
                continue
            hp = cfg.get("hyperparams") or {}
            if hp.get("seed") != seed:
                continue
            if bool(hp.get("speaker_independent")) != speaker_independent:
                continue
            mode = hp.get("train_mode") or (cfg.get("data") or {}).get("train_mode")
            if mode not in (None, "scratch"):
                continue
            summary_path = run_dir / "metrics" / "summary.json"
            val_f1 = 0.0
            if summary_path.is_file():
                val_f1 = float(
                    json.loads(summary_path.read_text(encoding="utf-8")).get(
                        "best_val_weighted_f1", 0.0
                    )
                )
            candidates.append((val_f1, run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_backbone_from_checkpoint(
    model: nn.Module,
    model_key: str,
    ckpt_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> dict:
    """Load source backbone only; skip head tensors (Head-FT)."""
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

    head_prefix = HEAD_ATTR[model_key] + "."
    filtered = {k: v for k, v in state.items() if not k.startswith(head_prefix)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    unexpected_non_head = [k for k in unexpected if not k.startswith(head_prefix)]
    return {
        "mode": "backbone_only",
        "checkpoint": str(ckpt_path),
        "loaded_tensors": len(filtered),
        "missing": list(missing),
        "unexpected_non_head": unexpected_non_head,
    }


def load_full_from_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> dict:
    """Load full source weights including head (Source-only / Full-FT; same num_classes)."""
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    return {
        "mode": "full",
        "checkpoint": str(ckpt_path),
        "loaded_tensors": len(state),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
