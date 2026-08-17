"""Resumes AGCRN training from its existing best checkpoint rather than
restarting from scratch. Extends the epoch ceiling and gives it a fresh
early-stopping patience so it can actually reach a genuine plateau,
strengthening the epoch-selection justification the assignment asks for.

`results/agcrn/checkpoints/` currently holds checkpoints from two prior
separate runs (the original 11-epoch run, best at epoch 9/val_mae=2.7526,
and an earlier resume that reached epoch 12/val_mae=2.7325 before being
stopped to switch machines) -- `find_best_checkpoint` picks whichever has
the lowest val_mae encoded in its filename, not just the most recent file,
so this always resumes from the genuinely best checkpoint regardless of how
many old files are sitting in that directory (see resume_utils.py).

Run: .venv/Scripts/python.exe -m code.experiments.resume_agcrn
"""
from __future__ import annotations

import json

import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.agcrn import build_agcrn_model
from code.experiments.train import run_training
from code.experiments.resume_utils import find_best_checkpoint, epoch_of
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/agcrn.yaml"

NEW_MAX_EPOCHS = 80
NEW_PATIENCE = 12


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("agcrn")
    ckpt_path = find_best_checkpoint("agcrn")
    log.milestone(
        f"=== Q3 AGCRN: RESUMING from {ckpt_path} (epoch {epoch_of(ckpt_path)}, "
        f"new ceiling max_epochs={NEW_MAX_EPOCHS}, patience={NEW_PATIENCE}, "
        f"now on GPU) ==="
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
    status_mod.set_stage("agcrn_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f} (resumed run, GPU)")

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
            f"AGCRN was first trained for 11 epochs on CPU (deadline-"
            f"constrained) with validation MAE still improving and no plateau "
            f"observed, then resumed once more on the same CPU to epoch 12 "
            f"before being stopped to switch to a GPU machine. Training was "
            f"resumed again from the best checkpoint on GPU with the ceiling "
            f"raised to {NEW_MAX_EPOCHS} and patience raised to {NEW_PATIENCE}. "
            + (
                f"Early stopping triggered after {NEW_PATIENCE} epochs with no "
                f"improvement, giving a genuine, validation-curve-justified "
                f"epoch count (best epoch {best_epoch}), not an arbitrary or "
                f"deadline-truncated one."
                if plateaued else
                f"Training again reached the new epoch ceiling "
                f"({NEW_MAX_EPOCHS}) while still improving (best epoch "
                f"{best_epoch}), so a genuine plateau still was not observed "
                f"within this extended budget; this remains an honestly-"
                f"reported limitation rather than a claimed convergence."
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
