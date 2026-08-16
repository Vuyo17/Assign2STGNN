"""Resumes AGCRN training from its existing best checkpoint (epoch 9,
val_mae=2.753, still improving with no plateau -- see report Section 5.1)
rather than restarting from scratch. Extends the epoch ceiling and gives it a
fresh early-stopping patience so it can actually reach a genuine plateau,
strengthening the epoch-selection justification the assignment asks for.

Run: .venv/Scripts/python.exe -m code.experiments.resume_agcrn
"""
from __future__ import annotations

import glob
import json

import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.agcrn import build_agcrn_model
from code.experiments.train import run_training
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/agcrn.yaml"

# New ceiling: 9h remain until the 10:00 deadline as of this run (~01:03).
# AGCRN alone (no contention from the other 3, which are all finished) ran its
# last epoch in ~18.7 min -- 16 more epochs at that pace is ~5h worst case,
# leaving a real buffer for report updates and packaging.
NEW_MAX_EPOCHS = 25
NEW_PATIENCE = 10


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("agcrn")
    ckpts = sorted(glob.glob("results/agcrn/checkpoints/best-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No existing AGCRN checkpoint found to resume from.")
    ckpt_path = ckpts[-1]
    log.milestone(
        f"=== Q3 AGCRN: RESUMING from {ckpt_path} (new ceiling max_epochs="
        f"{NEW_MAX_EPOCHS}, patience={NEW_PATIENCE}) ==="
    )

    data_cfg = DataConfig(
        window=cfg["data"]["window"], horizon=cfg["data"]["horizon"], stride=cfg["data"]["stride"],
        val_len=cfg["data"]["val_len"], test_len=cfg["data"]["test_len"],
        batch_size=cfg["data"]["batch_size"],
        connectivity_threshold=cfg["data"]["connectivity_threshold"],
        connectivity_include_self=cfg["data"]["connectivity_include_self"],
        connectivity_normalize_axis=cfg["data"]["connectivity_normalize_axis"],
        scaler_axis=tuple(cfg["data"]["scaler_axis"]),
    )
    dm, torch_dataset, dataset, connectivity = build_datamodule(data_cfg)

    model = build_agcrn_model(
        n_nodes=torch_dataset.n_nodes, horizon=torch_dataset.horizon,
        input_size=torch_dataset.n_channels,
    )

    predictor, new_ckpt_path, summary = run_training(
        model, dm, "agcrn",
        lr=cfg["optim"]["lr"], max_epochs=NEW_MAX_EPOCHS,
        early_stopping_patience=NEW_PATIENCE, seed=cfg["seed"],
        resume_from_checkpoint=ckpt_path,
    )

    status_mod.set_stage("agcrn_eval", "running")
    overall = run_evaluation(predictor, dm, "agcrn", n_nodes=torch_dataset.n_nodes, out_dir="results/agcrn")
    status_mod.set_stage("agcrn_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f} (resumed run)")

    with open("results/agcrn/training_history.json", encoding="utf-8") as f:
        history = json.load(f)
    val_losses = [v for v in history["val_loss"] if v is not None]
    best_epoch = int(history["epoch"][val_losses.index(min(val_losses))]) if val_losses else None
    plateaued = summary["early_stopped"]
    epoch_selection = {
        "epochs_run_total": summary["epochs_run"],
        "resumed_from_checkpoint": ckpt_path,
        "early_stopped": plateaued,
        "max_epochs_ceiling": NEW_MAX_EPOCHS,
        "early_stopping_patience": NEW_PATIENCE,
        "best_val_mae": summary["best_val_mae"],
        "epoch_of_best_val_mae": best_epoch,
        "justification": (
            f"AGCRN was first trained for 10 epochs (max_epochs=10, deadline-"
            f"constrained) with validation MAE still improving and no plateau "
            f"observed. Training was resumed from that checkpoint with the "
            f"ceiling raised to {NEW_MAX_EPOCHS} and patience raised to "
            f"{NEW_PATIENCE}. " + (
                f"Early stopping triggered after {NEW_PATIENCE} epochs with no "
                f"improvement, giving a genuine, validation-curve-justified "
                f"epoch count (best epoch {best_epoch}), not an arbitrary or "
                f"deadline-truncated one."
                if plateaued else
                f"Training again reached the new epoch ceiling "
                f"({NEW_MAX_EPOCHS}) while still improving (best epoch "
                f"{best_epoch}), so a genuine plateau still was not observed "
                f"within the extended, still deadline-bounded budget; this "
                f"remains an honestly-reported limitation rather than a "
                f"claimed convergence."
            )
        ),
    }
    with open("results/agcrn/epoch_selection.json", "w", encoding="utf-8") as f:
        json.dump(epoch_selection, f, indent=2)

    regenerate_outstanding()
    log.milestone(f"=== Q3 AGCRN: RESUME COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/agcrn/metrics.json, training_history.json, epoch_selection.json")


if __name__ == "__main__":
    main()
