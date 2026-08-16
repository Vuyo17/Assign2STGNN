"""Q2 Config B: train + evaluate GraphWaveNet with the predefined graph PLUS
the learned adaptive adjacency (learned_adjacency=True).

Run: .venv/Scripts/python.exe -m code.experiments.run_gwn_adaptive

Based on the CPU timing pilot, ~38 min/epoch, up to 30 epochs => up to ~19h
worst case; early stopping (patience=8) is expected to cut this short. Watch
progress via PROGRESS_LOG.md or logs/gwn_adaptive.log.

Saves everything run_gwn_predefined.py saves, PLUS
results/gwn_adaptive/learned_adjacency.npy -- the extracted NxN learned
adaptive adjacency matrix, required for Q2.3's analysis.
"""
from __future__ import annotations

import numpy as np
import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.gwn import build_gwn_model, extract_learned_adjacency
from code.experiments.train import run_training
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/gwn_adaptive.yaml"


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("gwn_adaptive")
    log.milestone(f"=== Q2 GWN (predefined+adaptive): full training run starting (config: {_CONFIG_PATH}) ===")

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
        model, dm, "gwn_adaptive",
        lr=cfg["optim"]["lr"], max_epochs=cfg["trainer"]["max_epochs"],
        early_stopping_patience=cfg["trainer"]["early_stopping_patience"], seed=cfg["seed"],
    )

    status_mod.set_stage("gwn_adaptive_eval", "running")
    overall = run_evaluation(predictor, dm, "gwn_adaptive", n_nodes=torch_dataset.n_nodes,
                                 out_dir="results/gwn_adaptive")
    status_mod.set_stage("gwn_adaptive_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f}")

    learned_adj = extract_learned_adjacency(predictor.model)
    np.save("results/gwn_adaptive/learned_adjacency.npy", learned_adj)
    ProgressLogger("gwn_adaptive").milestone(
        f"Saved learned adjacency matrix ({learned_adj.shape}) to "
        f"results/gwn_adaptive/learned_adjacency.npy"
    )

    regenerate_outstanding()
    log.milestone(f"=== Q2 GWN (predefined+adaptive): COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/gwn_adaptive/metrics.json and learned_adjacency.npy")


if __name__ == "__main__":
    main()
