"""Q3: train + evaluate AGCRN end to end. This run also serves as the
epoch-SELECTION experiment (generous max_epochs ceiling + early stopping;
the justified epoch count is read off training_history.json's validation
curve afterwards -- see report Section 5.1).

Run: .venv/Scripts/python.exe -m code.experiments.run_agcrn

Based on the CPU timing pilot, ~15 min/epoch, up to 60 epochs => up to ~15h
worst case. Watch progress via PROGRESS_LOG.md or logs/agcrn.log.

Saves: results/agcrn/{training_summary.json, training_history.json,
checkpoints/, metrics.json, metrics_per_node.csv, predictions.npz}.
"""
from __future__ import annotations

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


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("agcrn")
    log.milestone(f"=== Q3 AGCRN: full training run starting (config: {_CONFIG_PATH}) ===")

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

    predictor, ckpt_path, summary = run_training(
        model, dm, "agcrn",
        lr=cfg["optim"]["lr"], max_epochs=cfg["trainer"]["max_epochs"],
        early_stopping_patience=cfg["trainer"]["early_stopping_patience"], seed=cfg["seed"],
    )

    status_mod.set_stage("agcrn_eval", "running")
    overall = run_evaluation(predictor, dm, "agcrn", n_nodes=torch_dataset.n_nodes, out_dir="results/agcrn")
    status_mod.set_stage("agcrn_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f}")

    # Epoch-selection justification: read the actual validation curve and record
    # where it plateaued, alongside the early-stopped epoch count that was used.
    with open("results/agcrn/training_history.json", encoding="utf-8") as f:
        history = json.load(f)
    val_losses = [v for v in history["val_loss"] if v is not None]
    best_epoch = int(history["epoch"][val_losses.index(min(val_losses))]) if val_losses else None
    epoch_selection = {
        "epochs_run": summary["epochs_run"],
        "early_stopped": summary["early_stopped"],
        "max_epochs_ceiling": cfg["trainer"]["max_epochs"],
        "early_stopping_patience": cfg["trainer"]["early_stopping_patience"],
        "best_val_mae": summary["best_val_mae"],
        "epoch_of_best_val_mae": best_epoch,
        "justification": (
            f"Trained with a generous ceiling (max_epochs="
            f"{cfg['trainer']['max_epochs']}) and early stopping "
            f"(patience={cfg['trainer']['early_stopping_patience']} epochs of no "
            f"val_mae improvement). Validation MAE reached its best value at "
            f"epoch {best_epoch} of {summary['epochs_run']} run; the checkpoint "
            f"from that epoch is used for all reported AGCRN results, i.e. the "
            f"'selected' epoch count is the early-stopped best-validation epoch, "
            f"not an arbitrary fixed number."
        ),
    }
    with open("results/agcrn/epoch_selection.json", "w", encoding="utf-8") as f:
        json.dump(epoch_selection, f, indent=2)

    regenerate_outstanding()
    log.milestone(f"=== Q3 AGCRN: COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/agcrn/metrics.json and epoch_selection.json")


if __name__ == "__main__":
    main()
