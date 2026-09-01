from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ComplexityReport:
    total_params: int
    trainable_params: int
    frozen_params: int
    param_memory_mb: float
    activation_memory_mb: float | None
    macs: int | None
    gflops: float | None
    input_desc: str
    batch_size: int
    train_size: int | None
    time_complexity: str
    space_complexity: str
    train_steps_per_epoch: int | None
    est_train_macs_per_epoch: int | None


class SequenceSummaryWrapper(nn.Module):
    """Expose (frames, mask) tensor inputs for torchinfo."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask)


class DialogueSummaryWrapper(nn.Module):
    """Expose fixed-size dialogue node features for torchinfo."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _param_stats(model: nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def _bytes_to_mb(num_bytes: float) -> float:
    return round(num_bytes / (1024**2), 2)


def _ser_asymptotic_notes(*, frames: int | None = None, nodes: int | None = None) -> tuple[str, str]:
    if nodes is not None:
        return (
            f"O(N²·D); pseudo-dialogue nodes N={nodes}, feature dim D=256",
            f"O(P + N·D); P=params",
        )
    t = frames if frames is not None else "T"
    return (
        f"O(B·T·D); log-mel frames T={t}",
        f"O(P + B·T·D); P=params",
    )


def build_model_summary(
    model: nn.Module,
    *,
    input_size: tuple | None = None,
    input_data: tuple | None = None,
    depth: int = 4,
    col_names: tuple[str, ...] = (
        "input_size",
        "output_size",
        "num_params",
        "mult_adds",
    ),
) -> tuple[str, Any | None]:
    try:
        from torchinfo import summary
    except ImportError:
        total, trainable, frozen = _param_stats(model)
        fallback = (
            str(model)
            + f"\n\nTotal params: {total:,}  trainable: {trainable:,}  frozen: {frozen:,}\n"
            + "Install torchinfo for Keras-style summary: pip install torchinfo\n"
        )
        return fallback, None

    buf = io.StringIO()
    kwargs: dict[str, Any] = {
        "col_names": col_names,
        "depth": depth,
        "verbose": 1,
    }
    if input_data is not None:
        kwargs["input_data"] = input_data
    elif input_size is not None:
        kwargs["input_size"] = input_size
    else:
        raise ValueError("Provide input_size or input_data")

    stats = None
    try:
        with contextlib.redirect_stdout(buf):
            stats = summary(model, **kwargs)
    except RuntimeError as exc:
        total, trainable, frozen = _param_stats(model)
        fallback = (
            str(model)
            + f"\n\ntorchinfo failed ({exc})\n"
            + f"Total params: {total:,}  trainable: {trainable:,}  frozen: {frozen:,}\n"
        )
        return fallback, None
    return buf.getvalue(), stats


def build_complexity_report(
    model: nn.Module,
    *,
    stats: Any | None,
    input_desc: str,
    batch_size: int = 1,
    train_size: int | None = None,
    frames: int | None = None,
    nodes: int | None = None,
) -> ComplexityReport:
    total, trainable, frozen = _param_stats(model)
    param_mb = _bytes_to_mb(total * 4)

    macs = None
    gflops = None
    if stats is not None:
        macs = getattr(stats, "total_mult_adds", None)
        if macs is not None and macs > 0:
            gflops = round(macs / 1e9, 3)

    activation_mb = None
    if stats is not None:
        out_size = getattr(stats, "total_output_size", None)
        if out_size:
            activation_mb = _bytes_to_mb(out_size * batch_size)

    time_note, space_note = _ser_asymptotic_notes(frames=frames, nodes=nodes)

    steps = None
    est_epoch_macs = None
    if train_size and batch_size > 0:
        steps = (train_size + batch_size - 1) // batch_size
        if macs:
            est_epoch_macs = macs * batch_size * steps * 3

    return ComplexityReport(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        param_memory_mb=param_mb,
        activation_memory_mb=activation_mb,
        macs=macs,
        gflops=gflops,
        input_desc=input_desc,
        batch_size=batch_size,
        train_size=train_size,
        time_complexity=time_note,
        space_complexity=space_note,
        train_steps_per_epoch=steps,
        est_train_macs_per_epoch=est_epoch_macs,
    )


def format_complexity_block(report: ComplexityReport) -> str:
    lines = [
        "",
        "=" * 65,
        "Complexity & resource estimate (probe batch=1 unless noted)",
        "=" * 65,
        f"Input: {report.input_desc}",
        "",
        "[Space]",
        f"  Total params     : {report.total_params:,}",
        f"  Trainable params : {report.trainable_params:,}",
        f"  Frozen params    : {report.frozen_params:,}",
        f"  Weight memory    : ~{report.param_memory_mb} MB (fp32)",
    ]
    if report.activation_memory_mb is not None:
        lines.append(
            f"  Activation (est.) : ~{report.activation_memory_mb} MB "
            f"(batch={report.batch_size})"
        )
    lines.extend(
        [
            f"  Asymptotic       : {report.space_complexity}",
            "",
            "[Time — single forward]",
        ]
    )
    if report.macs is not None:
        lines.append(f"  MACs (torchinfo) : {report.macs:,}")
    if report.gflops is not None:
        lines.append(f"  GFLOPs (≈MACs/1e9): {report.gflops}")
    lines.append(f"  Asymptotic       : {report.time_complexity}")

    if report.train_size and report.train_steps_per_epoch:
        lines.extend(
            [
                "",
                "[Time — training estimate]",
                f"  Train samples    : {report.train_size:,}",
                f"  Batch size       : {report.batch_size}",
                f"  Steps / epoch    : {report.train_steps_per_epoch}",
            ]
        )
        if report.est_train_macs_per_epoch:
            lines.append(
                f"  MACs / epoch (≈3× fwd): {report.est_train_macs_per_epoch:,}"
            )
    lines.append("=" * 65)
    return "\n".join(lines)


def format_training_time_block(
    *,
    epoch_times: list[float],
    total_time_sec: float,
    train_size: int | None = None,
) -> str:
    lines = [
        "",
        "=" * 65,
        "Training time statistics",
        "=" * 65,
        f"  Total wall time  : {total_time_sec:.1f} s ({total_time_sec / 60:.2f} min)",
        f"  Epochs completed : {len(epoch_times)}",
    ]
    if epoch_times:
        mean = sum(epoch_times) / len(epoch_times)
        lines.extend(
            [
                f"  Mean epoch time  : {mean:.1f} s",
                f"  Min epoch time   : {min(epoch_times):.1f} s",
                f"  Max epoch time   : {max(epoch_times):.1f} s",
            ]
        )
        if train_size and train_size > 0:
            lines.append(f"  Mean time / sample: {mean / train_size * 1000:.1f} ms")
    lines.append("=" * 65)
    return "\n".join(lines)


def prepare_sequence_summary(
    model: nn.Module,
    batch: tuple,
    device: torch.device,
    *,
    batch_size: int,
    train_size: int,
    depth: int = 4,
) -> str:
    x, mask, *_rest = batch
    n_valid = max(1, int(mask[0].sum().item()))
    x = x[:1, :n_valid].to(device)
    mask = torch.ones(1, n_valid, dtype=torch.bool, device=device)
    wrapper = SequenceSummaryWrapper(model).to(device)
    wrapper.eval()
    table, stats = build_model_summary(wrapper, input_data=(x, mask), depth=depth)
    frames = n_valid
    input_desc = f"frames=(1, {n_valid}, {x.size(-1)}), valid_frames={frames}"
    report = build_complexity_report(
        model,
        stats=stats,
        input_desc=input_desc,
        batch_size=batch_size,
        train_size=train_size,
        frames=frames,
    )
    return table.rstrip() + format_complexity_block(report)


def prepare_dialogue_summary(
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    train_size: int,
    n_nodes: int = 10,
    feat_dim: int = 256,
    depth: int = 4,
) -> str:
    x = torch.randn(n_nodes, feat_dim, device=device)
    wrapper = DialogueSummaryWrapper(model).to(device)
    wrapper.eval()
    table, stats = build_model_summary(wrapper, input_data=(x,), depth=depth)
    input_desc = f"dialogue_nodes=({n_nodes}, {feat_dim})"
    report = build_complexity_report(
        model,
        stats=stats,
        input_desc=input_desc,
        batch_size=batch_size,
        train_size=train_size,
        nodes=n_nodes,
    )
    return table.rstrip() + format_complexity_block(report)
