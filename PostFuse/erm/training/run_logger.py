from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from erm.training.curves import load_epoch_records, plot_training_curves
from erm.training.model_info import (
    build_complexity_report,
    build_model_summary,
    format_complexity_block,
    format_training_time_block,
    prepare_dialogue_summary,
    prepare_sequence_summary,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_torch_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load checkpoint; remap PosixPath↔WindowsPath for cross-OS .pt files."""
    import pathlib

    # Checkpoints saved on Linux pickle PosixPath; Windows cannot instantiate it.
    _posix = getattr(pathlib, "PosixPath", None)
    _win = getattr(pathlib, "WindowsPath", None)
    if sys.platform == "win32" and _posix is not None:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    elif sys.platform != "win32" and _win is not None:
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc, assignment]
    kwargs: dict[str, Any] = {"map_location": map_location}
    try:
        try:
            return torch.load(path, weights_only=False, **kwargs)
        except TypeError:
            return torch.load(path, **kwargs)
    finally:
        if _posix is not None:
            pathlib.PosixPath = _posix  # type: ignore[misc, assignment]
        if _win is not None:
            pathlib.WindowsPath = _win  # type: ignore[misc, assignment]


def collect_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch as _torch

        info["torch_version"] = _torch.__version__
        info["cuda_available"] = _torch.cuda.is_available()
        if _torch.cuda.is_available():
            info["cuda_version"] = _torch.version.cuda
            info["gpu_name"] = _torch.cuda.get_device_name(0)
            info["gpu_count"] = _torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
    return info


class RunLogger:
    """
    Each run -> {results_base}/{experiment}/{YYYYMMDD_HHMMSS}_{run_name}/

    Default results_base is ``{root}/results`` (legacy flat). Prefer
    ``results_base=results/{dataset}`` so CASIA and EmoDB do not share folders.

    logs/     train.log, epochs.jsonl, training_time.txt
    metrics/  summary.json, metrics.json, test_report.txt
    figures/  confusion matrices, training curves
    model/    best_model.pt, model_structure.txt (torchinfo + complexity)
    config.json
    """

    def __init__(
        self,
        root: Path,
        experiment: str,
        method: str,
        run_name: str | None = None,
        *,
        results_base: Path | None = None,
    ):
        self.root = Path(root)
        self.experiment = experiment
        self.method = method
        self.results_base = Path(results_base) if results_base is not None else (self.root / "results")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{run_name}" if run_name else ts
        self.run_dir = self.results_base / experiment / self.run_id

        self.logs_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.figures_dir = self.run_dir / "figures"
        self.model_dir = self.run_dir / "model"
        for d in (self.logs_dir, self.metrics_dir, self.figures_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._log_path = self.logs_dir / "train.log"
        self._epochs_path = self.logs_dir / "epochs.jsonl"
        self._start_time = time.time()
        self._best_metric: float | None = None
        self._epoch_times: list[float] = []
        self._train_size: int | None = None

        latest = self.results_base / experiment / "latest.json"
        try:
            rel_run = str(self.run_dir.relative_to(self.root))
        except ValueError:
            rel_run = str(self.run_dir)
        latest.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "run_dir": rel_run,
                    "method": method,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "best_model.pt"

    def _write_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line)
        print(msg)

    def log_config(
        self,
        hyperparams: dict[str, Any],
        model: nn.Module | None = None,
        data_info: dict[str, Any] | None = None,
    ) -> None:
        config = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "method": self.method,
            "hyperparams": hyperparams,
            "data": data_info or {},
            "environment": collect_env_info(),
            "artifacts": {
                "logs": str(self.logs_dir.relative_to(self.run_dir)),
                "metrics": str(self.metrics_dir.relative_to(self.run_dir)),
                "figures": str(self.figures_dir.relative_to(self.run_dir)),
                "model": str(self.model_dir.relative_to(self.run_dir)),
            },
        }
        if model is not None:
            config["model"] = {
                "class": model.__class__.__name__,
                "num_parameters": count_parameters(model),
            }
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        env = config["environment"]
        self._write_log(f"Run started  run_id={self.run_id}")
        self._write_log(f"Output -> {self.run_dir.relative_to(self.root)}")
        self._write_log(
            f"Env  python={sys.version_info.major}.{sys.version_info.minor}  "
            f"torch={env.get('torch_version')}  gpu={env.get('gpu_name', 'N/A')}"
        )
        hp = "  ".join(f"{k}={v}" for k, v in hyperparams.items())
        self._write_log(f"Config {hp}")
        if model is not None:
            self._write_log(f"Model {model.__class__.__name__}  params={count_parameters(model):,}")

    def log_model_structure(
        self,
        model: nn.Module,
        *,
        input_size: tuple | None = None,
        input_data: tuple | None = None,
        batch_size: int = 1,
        train_size: int | None = None,
        depth: int = 4,
        input_desc: str | None = None,
        frames: int | None = None,
        nodes: int | None = None,
    ) -> Path:
        """Print torchinfo layer table + complexity block before training."""
        if train_size is not None:
            self._train_size = train_size

        table, stats = build_model_summary(
            model,
            input_size=input_size,
            input_data=input_data,
            depth=depth,
        )
        if input_desc is None:
            if input_size is not None:
                input_desc = f"input_size={input_size}"
            elif input_data is not None:
                first = input_data[0]
                input_desc = (
                    f"input_data[0].shape={tuple(first.shape)}"
                    if hasattr(first, "shape")
                    else "input_data=(...)"
                )
            else:
                input_desc = "unknown"

        report = build_complexity_report(
            model,
            stats=stats,
            input_desc=input_desc,
            batch_size=batch_size,
            train_size=train_size,
            frames=frames,
            nodes=nodes,
        )
        text = table.rstrip() + format_complexity_block(report)

        structure_path = self.model_dir / "model_structure.txt"
        structure_path.write_text(text, encoding="utf-8")
        print("\n" + text + "\n")
        self._write_log(f"Model structure -> {structure_path.relative_to(self.run_dir)}")
        return structure_path

    def log_sequence_structure(
        self,
        model: nn.Module,
        batch: tuple,
        device: torch.device,
        *,
        batch_size: int,
        train_size: int,
        depth: int = 4,
    ) -> Path:
        self._train_size = train_size
        text = prepare_sequence_summary(
            model,
            batch,
            device,
            batch_size=batch_size,
            train_size=train_size,
            depth=depth,
        )
        structure_path = self.model_dir / "model_structure.txt"
        structure_path.write_text(text, encoding="utf-8")
        print("\n" + text + "\n")
        self._write_log(f"Model structure -> {structure_path.relative_to(self.run_dir)}")
        return structure_path

    def log_dialogue_structure(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        batch_size: int,
        train_size: int,
        n_nodes: int = 10,
        depth: int = 4,
    ) -> Path:
        self._train_size = train_size
        text = prepare_dialogue_summary(
            model,
            device=device,
            batch_size=batch_size,
            train_size=train_size,
            n_nodes=n_nodes,
            depth=depth,
        )
        structure_path = self.model_dir / "model_structure.txt"
        structure_path.write_text(text, encoding="utf-8")
        print("\n" + text + "\n")
        self._write_log(f"Model structure -> {structure_path.relative_to(self.run_dir)}")
        return structure_path

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        *,
        train_loss: float,
        val_weighted_f1: float,
        val_accuracy: float | None = None,
        epoch_time_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        elapsed = time.time() - self._start_time
        is_best = bool(self._best_metric is None or val_weighted_f1 >= self._best_metric)
        if is_best:
            self._best_metric = val_weighted_f1

        record = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": round(float(train_loss), 6),
            "val_weighted_f1": round(float(val_weighted_f1), 6),
            "val_accuracy": round(float(val_accuracy), 6) if val_accuracy is not None else None,
            "epoch_time_sec": round(epoch_time_sec, 3),
            "elapsed_sec": round(elapsed, 3),
            "is_best": is_best,
        }
        if extra:
            record.update(extra)

        with open(self._epochs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._epoch_times.append(float(epoch_time_sec))

        best_tag = " [best]" if is_best else ""
        acc_str = f"  val_acc={val_accuracy:.4f}" if val_accuracy is not None else ""
        self._write_log(
            f"Epoch {epoch:03d}/{total_epochs:03d}  loss={train_loss:.4f}  "
            f"val_f1={val_weighted_f1:.4f}{acc_str}  time={epoch_time_sec:.1f}s{best_tag}"
        )
        return is_best

    def plot_training_curves(self, *, title: str | None = None) -> dict[str, Path] | None:
        records = load_epoch_records(self._epochs_path)
        if not records:
            return None
        prefix = title or self.method
        paths = plot_training_curves(
            records,
            save_dir=self.figures_dir,
            title_prefix=prefix,
        )
        for name, path in paths.items():
            self._write_log(f"Training curve ({name}) -> {path.relative_to(self.run_dir)}")
        return paths

    def save_checkpoint(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "run_id": self.run_id,
            "method": self.method,
            "epoch": epoch,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, self.checkpoint_path)
        return self.checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        map_location = device if device is not None else "cpu"
        payload = load_torch_checkpoint(self.checkpoint_path, map_location=map_location)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload

    def finish(
        self,
        test_metrics: dict[str, Any],
        test_report: str | None = None,
        test_metrics_detail: dict[str, Any] | None = None,
        confusion_info: dict[str, Any] | None = None,
    ) -> Path:
        total_time = time.time() - self._start_time
        _train_keys = {
            "dataset",
            "model",
            "epochs_trained",
            "early_stopped",
            "patience",
            "speaker_independent",
            "seed",
            "note",
        }
        _test_exclude = _train_keys | {
            "best_val_weighted_f1",
            "test_accuracy",
            "test_weighted_accuracy",
            "test_wa",
            "test_unweighted_accuracy",
            "test_ua",
            "test_weighted_f1",
            "test_macro_f1",
        }
        summary = {
            "run_id": self.run_id,
            "method": self.method,
            "experiment": self.experiment,
            "total_time_sec": round(total_time, 3),
            "validation": {
                "best_weighted_f1": round(float(self._best_metric or 0), 6),
            },
            "test": {
                k: v
                for k, v in test_metrics.items()
                if k not in _test_exclude
                and k
                not in {
                    "test_accuracy",
                    "test_weighted_accuracy",
                    "test_wa",
                    "test_unweighted_accuracy",
                    "test_ua",
                    "test_weighted_f1",
                    "test_macro_f1",
                }
            },
            "training": {k: v for k, v in test_metrics.items() if k in _train_keys},
            **test_metrics,
        }
        if confusion_info:
            summary["confusion"] = confusion_info

        curve_paths = self.plot_training_curves(title=self.method)
        if curve_paths is not None:
            figures = summary.setdefault("figures", {})
            for name, path in curve_paths.items():
                figures[name] = str(path.relative_to(self.run_dir))

        summary_path = self.metrics_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics_path = self.metrics_dir / "metrics.json"
        metrics_payload = dict(summary)
        if test_metrics_detail:
            metrics_payload["per_class"] = test_metrics_detail
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if test_report:
            (self.metrics_dir / "test_report.txt").write_text(test_report, encoding="utf-8")
            print("\n" + test_report)

        acc = test_metrics.get("test_accuracy")
        wa = test_metrics.get("test_wa", acc)
        ua = test_metrics.get("test_ua", test_metrics.get("test_unweighted_accuracy"))
        wf1 = test_metrics.get("test_weighted_f1")
        mf1 = test_metrics.get("test_macro_f1")
        ua_str = f"  UA={ua}" if ua is not None else ""
        mf1_str = f"  macro_f1={mf1}" if mf1 is not None else ""
        time_block = format_training_time_block(
            epoch_times=self._epoch_times,
            total_time_sec=total_time,
            train_size=self._train_size,
        )
        print(time_block)
        with open(self.logs_dir / "training_time.txt", "w", encoding="utf-8") as f:
            f.write(time_block.strip() + "\n")

        self._write_log(
            f"Test  WA={wa}{ua_str}  weighted_f1={wf1}{mf1_str}  total_time={total_time:.1f}s"
        )
        self._write_log(f"Metrics -> {self.metrics_dir.relative_to(self.run_dir)}")
        self._write_log(f"Figures -> {self.figures_dir.relative_to(self.run_dir)}")
        self._write_log(f"Done  run_dir={self.run_dir.relative_to(self.root)}")
        return summary_path
