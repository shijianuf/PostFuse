from __future__ import annotations


class EarlyStopping:
    """Stop when validation metric does not improve for `patience` epochs."""

    def __init__(
        self,
        *,
        patience: int = 10,
        min_delta: float = 1e-4,
        max_epochs: int = 100,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.best_score: float | None = None
        self.counter = 0
        self.stopped_epoch: int | None = None

    def step(self, score: float) -> tuple[bool, bool]:
        """Return (should_stop, improved)."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False, True

        self.counter += 1
        return self.counter >= self.patience, False

    def mark_stopped(self, epoch: int) -> None:
        self.stopped_epoch = epoch
