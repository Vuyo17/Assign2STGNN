"""Shared training harness. Every experiment (TTS, GWN-predefined, GWN-adaptive,
AGCRN) calls `run_training(model, dm, cfg, experiment_name)` so the optimiser,
loss, metrics, logging, checkpointing, and early-stopping setup are identical
across all four -- only the `model` and its own hyperparameters differ, which is
what makes the cross-model comparison fair.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

# Running all 4 experiments concurrently (this project's deadline-driven
# strategy -- see PROGRESS_LOG.md ~17:18) means each process must NOT try to
# claim all logical CPUs, or the 4 processes thrash each other via thread
# oversubscription (observed: TTS epoch time went 2.0min isolated -> 5.6min
# under 4-way contention with default unbounded threading, a 2.8x slowdown
# far worse than the 4-way share alone would predict). Cap each process to a
# fair share of the 20 logical CPUs on this machine. Safe to import even when
# only one experiment runs at a time.
_N_THREADS = max(1, (os.cpu_count() or 4) // 3)
os.environ.setdefault("OMP_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_THREADS))
torch.set_num_threads(_N_THREADS)

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
    accelerator: str = "auto",
    resume_from_checkpoint: str | None = None,
):
    """Trains `model` (already constructed) via a tsl `Predictor` + PyTorch
    Lightning `Trainer`, with early stopping on `val_mae`, TensorBoard + CSV
    logging, and best-checkpoint saving. Returns
    (predictor, best_ckpt_path, training_summary: dict).

    `limit_train_batches`/`limit_val_batches` are only set by the timing pilot
    (Phase 3) to cap a run to a handful of batches; real experiments leave them
    as None (full data).

    `accelerator="auto"` (the default) lets PyTorch Lightning pick a GPU if one
    is available and fall back to CPU otherwise -- this is what makes the exact
    same entrypoint scripts (run_tts.py etc.) use CUDA automatically on a GPU
    machine (e.g. Google Colab) without any code change, while still running
    CPU-only on this project's local (no-GPU) development machine.

    `resume_from_checkpoint`, if given a checkpoint path, restores model
    weights, optimiser state, and the epoch counter from that checkpoint and
    continues training from there (via Lightning's own `ckpt_path=` resume
    mechanism) rather than starting from epoch 0 -- `max_epochs` must be set
    higher than the checkpoint's own epoch for this to do any further training.
    The returned `training_history.json` is the FULL curve (pre-resume epochs
    merged with the newly-run ones), not just the newly-run portion.
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
        accelerator=accelerator,
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

    if resume_from_checkpoint:
        # PyTorch >=2.6 defaults torch.load(weights_only=True) for security,
        # which Lightning's own ckpt_path= resume path does not override.
        # Our checkpoints embed tsl's metric objects (MaskedMAE etc.) inside
        # the Predictor's state, which weights_only=True refuses to unpickle
        # by default. Allowlisting these specific, known, trusted classes
        # (rather than disabling weights_only globally) is the officially
        # recommended fix.
        from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE

        torch.serialization.add_safe_globals([MaskedMAE, MaskedMAPE, MaskedMSE])
        log.milestone(f"Resuming from checkpoint {resume_from_checkpoint} (new ceiling max_epochs={max_epochs})")

    t0 = time.time()
    trainer.fit(predictor, datamodule=dm, ckpt_path=resume_from_checkpoint)
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

    out_dir = _ROOT / "results" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resumed_from_epochs = 0
    prior_total_seconds = 0.0
    if resume_from_checkpoint:
        old_summary_path = out_dir / "training_summary.json"
        if old_summary_path.exists():
            with open(old_summary_path, encoding="utf-8") as f:
                prior_total_seconds = json.load(f).get("total_training_seconds") or 0.0
        old_history_path = out_dir / "training_history.json"
        if old_history_path.exists():
            with open(old_history_path, encoding="utf-8") as f:
                old_history = json.load(f)
            resumed_from_epochs = len(old_history.get("epoch", []))
            # Merge: keep the pre-resume epochs, then append any new epoch
            # indices not already present (avoids double-counting the
            # checkpoint's own epoch, which Lightning may re-log on resume).
            seen = set(old_history.get("epoch", []))
            merged = {
                "epoch": list(old_history.get("epoch", [])),
                "train_loss": list(old_history.get("train_loss", [])),
                "val_loss": list(old_history.get("val_loss", [])),
            }
            for e, tl, vl in zip(history["epoch"], history["train_loss"], history["val_loss"]):
                if e not in seen:
                    merged["epoch"].append(e)
                    merged["train_loss"].append(tl)
                    merged["val_loss"].append(vl)
                    seen.add(e)
            history = merged
            log.milestone(
                f"Merged pre-resume history ({resumed_from_epochs} epochs) with "
                f"newly-run epochs -- full curve now {len(history['epoch'])} epochs"
            )

    cumulative_seconds = prior_total_seconds + total_seconds
    summary = {
        "experiment": experiment_name,
        "total_training_seconds": cumulative_seconds,  # cumulative across resumes
        "this_call_seconds": total_seconds,
        "epochs_run": len(history["epoch"]),  # total, including any pre-resume epochs
        "epochs_run_this_call": epochs_run,
        "resumed_from_checkpoint": resume_from_checkpoint,
        "seconds_per_epoch_avg": cumulative_seconds / max(len(history["epoch"]), 1),
        "best_val_mae": best_val_mae,
        "best_checkpoint": best_ckpt_path,
        "early_stopped": trainer.current_epoch + 1 < max_epochs,
        "max_epochs_configured": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate": lr,
        "seed": seed,
    }

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
