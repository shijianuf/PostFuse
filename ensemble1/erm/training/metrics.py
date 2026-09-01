"""Training / evaluation metrics for SER (including WA and UA)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score


def compute_ser_metrics(
    y_true,
    y_pred,
    *,
    labels: list[str] | None = None,
    digits: int = 4,
) -> dict[str, float | str]:
    """Compute standard SER metrics.

    WA (Weighted Accuracy): overall accuracy (= sample-weighted correct rate).
    UA (Unweighted Accuracy): mean per-class recall (unweighted across classes).

    Note: UA is **not** macro F1; macro F1 averages F1 per class, UA averages recall.
    """
    wa = float(accuracy_score(y_true, y_pred))
    ua = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report_kwargs: dict = {"digits": digits, "zero_division": 0}
    if labels is not None:
        report_kwargs["labels"] = list(range(len(labels)))
        report_kwargs["target_names"] = labels
    report = classification_report(y_true, y_pred, **report_kwargs)

    rd = lambda x: round(x, digits)  # noqa: E731
    return {
        "weighted_accuracy": rd(wa),
        "unweighted_accuracy": rd(ua),
        "weighted_f1": rd(wf1),
        "macro_f1": rd(mf1),
        "report": report,
        # Aliases used in logs / JSON
        "test_accuracy": rd(wa),
        "test_weighted_accuracy": rd(wa),
        "test_unweighted_accuracy": rd(ua),
        "test_weighted_f1": rd(wf1),
        "test_macro_f1": rd(mf1),
        "test_wa": rd(wa),
        "test_ua": rd(ua),
    }


def save_summary(
    out_dir: Path,
    *,
    model: str,
    test_acc: float,
    test_wf1: float,
    test_mf1: float,
    test_ua: float | None = None,
    extra: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_acc": test_acc,
        "test_wa": test_acc,
        "test_weighted_accuracy": test_acc,
        "test_ua": test_ua,
        "test_unweighted_accuracy": test_ua,
        "test_weighted_f1": test_wf1,
        "test_macro_f1": test_mf1,
    }
    if extra:
        payload.update(extra)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
