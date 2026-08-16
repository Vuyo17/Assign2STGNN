"""PyTorch Lightning callback that streams per-epoch progress into the shared
progress log -- this is what makes ``PROGRESS_LOG.md`` show live epoch-by-epoch
heartbeat for whichever training run is currently in the foreground/background,
without needing to tail PyTorch Lightning's own (much noisier) stdout.
"""
from __future__ import annotations

import time

from pytorch_lightning.callbacks import Callback

from code.utils.progress_logger import ProgressLogger


class ProgressLogCallback(Callback):
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.logger = ProgressLogger(run_name)
        self._epoch_start = None
        self._run_start = None

    def on_fit_start(self, trainer, pl_module):
        self._run_start = time.time()
        self.logger.milestone(
            f"Training started (max_epochs={trainer.max_epochs}, "
            f"accelerator={trainer.accelerator.__class__.__name__})"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Validation metrics for this epoch aren't populated until
        # on_validation_epoch_end, which Lightning calls right after this hook
        # in the same epoch -- so we log the compact summary there instead.
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._epoch_start is None:
            return
        epoch_time = time.time() - self._epoch_start
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_mae = metrics.get("val_mae")
        parts = [f"epoch {trainer.current_epoch}"]
        if train_loss is not None:
            parts.append(f"train_loss={float(train_loss):.4f}")
        if val_mae is not None:
            parts.append(f"val_mae={float(val_mae):.4f}")
        parts.append(f"({epoch_time:.1f}s)")
        self.logger.milestone(" ".join(parts))

    def on_fit_end(self, trainer, pl_module):
        total = time.time() - self._run_start if self._run_start else float("nan")
        self.logger.milestone(
            f"Training finished after {trainer.current_epoch + 1} epochs, "
            f"{total / 60:.1f} min total"
        )
