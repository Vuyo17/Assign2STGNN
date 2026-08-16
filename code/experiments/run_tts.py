"""Q1: train + evaluate TimeThenSpaceModel end to end.

Run (from the project root, with the venv active or via its python directly):
    .venv/Scripts/python.exe -m code.experiments.run_tts

Watch progress live in another terminal with:
    Get-Content PROGRESS_LOG.md -Wait -Tail 20        (PowerShell)
    tail -f PROGRESS_LOG.md                            (bash)
or the more detailed per-run log at logs/tts.log.

Saves: results/tts/{training_summary.json, training_history.json,
checkpoints/, metrics.json, metrics_per_node.csv, predictions.npz},
logs/tts/ (TensorBoard + CSV logger output).
"""
from __future__ import annotations

import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.tts import TimeThenSpaceModel
from code.experiments.train import run_training
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/tts.yaml"


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("tts")
    log.milestone(f"=== Q1 TTS: full training run starting (config: {_CONFIG_PATH}) ===")

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

    model = TimeThenSpaceModel(
        input_size=torch_dataset.n_channels, n_nodes=torch_dataset.n_nodes,
        horizon=torch_dataset.horizon,
        hidden_size=cfg["model"]["hidden_size"], rnn_layers=cfg["model"]["rnn_layers"],
        gnn_kernel=cfg["model"]["gnn_kernel"],
    )

    predictor, ckpt_path, summary = run_training(
        model, dm, "tts",
        lr=cfg["optim"]["lr"], max_epochs=cfg["trainer"]["max_epochs"],
        early_stopping_patience=cfg["trainer"]["early_stopping_patience"], seed=cfg["seed"],
    )

    status_mod.set_stage("tts_eval", "running")
    overall = run_evaluation(predictor, dm, "tts", n_nodes=torch_dataset.n_nodes, out_dir="results/tts")
    status_mod.set_stage("tts_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f}")
    regenerate_outstanding()

    log.milestone(f"=== Q1 TTS: COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/tts/metrics.json for overall metrics.")


if __name__ == "__main__":
    main()
