"""Q2 Config A: train + evaluate GraphWaveNet with the PREDEFINED graph only
(learned_adjacency=False).

Run: .venv/Scripts/python.exe -m code.experiments.run_gwn_predefined

Based on the CPU timing pilot (results/timing_pilot.json), this is the most
expensive run in the whole project (~34 min/epoch, up to 30 epochs => up to
~17h worst case; early stopping with patience=8 is expected to cut this
short in practice, but plan for a long run). Watch progress via
PROGRESS_LOG.md or logs/gwn_predefined.log.

Saves: results/gwn_predefined/{training_summary.json, training_history.json,
checkpoints/, metrics.json, metrics_per_node.csv, predictions.npz}.
"""
from __future__ import annotations

import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.gwn import build_gwn_model
from code.experiments.train import run_training
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/gwn_predefined.yaml"


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("gwn_predefined")
    log.milestone(f"=== Q2 GWN (predefined only): full training run starting (config: {_CONFIG_PATH}) ===")

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

    model = build_gwn_model(
        n_nodes=torch_dataset.n_nodes, horizon=torch_dataset.horizon,
        input_size=torch_dataset.n_channels,
        learned_adjacency=cfg["model"]["learned_adjacency"],
    )

    predictor, ckpt_path, summary = run_training(
        model, dm, "gwn_predefined",
        lr=cfg["optim"]["lr"], max_epochs=cfg["trainer"]["max_epochs"],
        early_stopping_patience=cfg["trainer"]["early_stopping_patience"], seed=cfg["seed"],
    )

    status_mod.set_stage("gwn_predefined_eval", "running")
    overall = run_evaluation(predictor, dm, "gwn_predefined", n_nodes=torch_dataset.n_nodes,
                                 out_dir="results/gwn_predefined")
    status_mod.set_stage("gwn_predefined_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f}")
    regenerate_outstanding()

    log.milestone(f"=== Q2 GWN (predefined only): COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/gwn_predefined/metrics.json for overall metrics.")


if __name__ == "__main__":
    main()
