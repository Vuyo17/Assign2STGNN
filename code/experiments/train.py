"""Shared training harness. Every experiment (TTS, GWN-predefined, GWN-adaptive,
AGCRN) calls `run_training(model, dm, cfg, experiment_name)` so the optimiser,
loss, metrics, logging, checkpointing, and early-stopping setup are identical
across all four -- only the `model` and its own hyperparameters differ, which is
what makes the cross-model comparison fair.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from code.experiments.callbacks import ProgressLogCallback
from code.utils import status as status_mod
from code.utils.progress_logger import ProgressLogger
from code.utils.seed import set_seed
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_ROOT = Path(__file__).resolve().parents[2]


def _build_predictor(model, lr: float):
    from tsl.engines import Predictor
    from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE

    loss_fn = MaskedMAE()
    metrics = {
        "mae": MaskedMAE(),
        "mse": MaskedMSE(),
        "mape": MaskedMAPE(),
        "mae_at_15": MaskedMAE(at=2),
        "mae_at_30": MaskedMAE(at=5),
        "mae_at_60": MaskedMAE(at=11),
    }
    return Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": lr},
        loss_fn=loss_fn,
        metrics=metrics,
    )


def run_training(
    model,
    dm,
    experiment_name: str,
    lr: float = 0.001,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
    seed: int = 42,
    limit_train_batches=None,
    limit_val_batches=None,
):
    """Trains `model` (already constructed) via a tsl `Predictor` + PyTorch
    Lightning `Trainer`, with early stopping on `val_mae`, TensorBoard + CSV
    logging, and best-checkpoint saving. Returns
    (predictor, best_ckpt_path, training_summary: dict).

    `limit_train_batches`/`limit_val_batches` are only set by the timing pilot
    (Phase 3) to cap a run to a handful of batches; real experiments leave them
    as None (full data).
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    set_seed(seed)
    status_mod.set_stage(f"{experiment_name}_train", "running")
    log = ProgressLogger(experiment_name)
    log.milestone(f"Building predictor (lr={lr}, max_epochs={max_epochs}, seed={seed})")

    dm.setup()
    predictor = _build_predictor(model, lr)

    run_dir = _ROOT / "logs" / experiment_name
    tb_logger = TensorBoardLogger(save_dir=str(_ROOT / "logs"), name=experiment_name)
    csv_logger = CSVLogger(save_dir=str(_ROOT / "logs"), name=experiment_name)

    ckpt_dir = _ROOT / "results" / experiment_name / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), save_top_k=1, monitor="val_mae", mode="min",
        filename="best-{epoch}-{val_mae:.4f}",
    )
    early_stop_cb = EarlyStopping(monitor="val_mae", mode="min", patience=early_stopping_patience)
    progress_cb = ProgressLogCallback(experiment_name)

    trainer_kwargs = dict(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_cb, early_stop_cb, progress_cb],
        enable_progress_bar=False,  # progress bar output isn't useful in a captured log file
        deterministic=False,        # some tsl/PyG scatter ops lack a deterministic CPU kernel
    )
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)

    t0 = time.time()
    trainer.fit(predictor, datamodule=dm)
    total_seconds = time.time() - t0

    best_ckpt_path = checkpoint_cb.best_model_path
    best_val_mae = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else None
    epochs_run = trainer.current_epoch + 1

    # Parse the CSVLogger's metrics.csv into a compact per-epoch history for the
    # convergence-curve figures (train_loss / val_mae per epoch).
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if "epoch" in df.columns:
            train_col = "train_loss" if "train_loss" in df.columns else None
            val_col = "val_mae" if "val_mae" in df.columns else None
            grouped = df.groupby("epoch")
            for epoch, g in grouped:
                history["epoch"].append(int(epoch))
                history["train_loss"].append(float(g[train_col].dropna().mean()) if train_col and g[train_col].notna().any() else None)
                history["val_loss"].append(float(g[val_col].dropna().mean()) if val_col and g[val_col].notna().any() else None)

    summary = {
        "experiment": experiment_name,
        "total_training_seconds": total_seconds,
        "epochs_run": epochs_run,
        "seconds_per_epoch_avg": total_seconds / max(epochs_run, 1),
        "best_val_mae": best_val_mae,
        "best_checkpoint": best_ckpt_path,
        "early_stopped": trainer.current_epoch + 1 < max_epochs,
        "max_epochs_configured": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate": lr,
        "seed": seed,
    }

    out_dir = _ROOT / "results" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log.milestone(
        f"Training complete: {epochs_run} epochs, {total_seconds/60:.1f} min total, "
        f"best_val_mae={best_val_mae}, checkpoint={best_ckpt_path}"
    )
    status_mod.set_stage(
        f"{experiment_name}_train", "done",
        notes=f"{epochs_run} epochs, {total_seconds/60:.1f} min, best_val_mae={best_val_mae}",
    )
    regenerate_outstanding()

    return predictor, best_ckpt_path, summary
