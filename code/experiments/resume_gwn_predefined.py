"""Resumes GraphWaveNet (predefined graph only) training from its existing
best checkpoint (epoch 2, val_mae=2.936) rather than restarting from scratch.
The original run was capped at 4 epochs purely for CPU/deadline reasons and
showed a possible overfitting uptick right at that ceiling (see report
Section 4.3); resuming with a real budget on GPU tells us whether that was a
genuine plateau or just noise from stopping too early.

Run: .venv/Scripts/python.exe -m code.experiments.resume_gwn_predefined
"""
from __future__ import annotations

import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.gwn import build_gwn_model
from code.experiments.train import run_training
from code.experiments.resume_utils import find_best_checkpoint, epoch_of
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/gwn_predefined.yaml"

NEW_MAX_EPOCHS = 60
NEW_PATIENCE = 12


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("gwn_predefined")
    ckpt_path = find_best_checkpoint("gwn_predefined")
    log.milestone(
        f"=== Q2 GWN (predefined only): RESUMING from {ckpt_path} (epoch "
        f"{epoch_of(ckpt_path)}, new ceiling max_epochs={NEW_MAX_EPOCHS}, "
        f"patience={NEW_PATIENCE}, now on GPU) ==="
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

    model = build_gwn_model(
        n_nodes=torch_dataset.n_nodes, horizon=torch_dataset.horizon,
        input_size=torch_dataset.n_channels,
        learned_adjacency=cfg["model"]["learned_adjacency"],
    )

    predictor, new_ckpt_path, summary = run_training(
        model, dm, "gwn_predefined",
        lr=cfg["optim"]["lr"], max_epochs=NEW_MAX_EPOCHS,
        early_stopping_patience=NEW_PATIENCE, seed=cfg["seed"],
        resume_from_checkpoint=ckpt_path,
    )

    status_mod.set_stage("gwn_predefined_eval", "running")
    overall = run_evaluation(predictor, dm, "gwn_predefined", n_nodes=torch_dataset.n_nodes,
                                 out_dir="results/gwn_predefined")
    status_mod.set_stage("gwn_predefined_eval", "done",
                             notes=f"60min MAE={overall['60min']['mae']:.3f} (resumed run, GPU)")

    regenerate_outstanding()
    log.milestone(f"=== Q2 GWN (predefined only): RESUME COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/gwn_predefined/metrics.json, training_history.json")


if __name__ == "__main__":
    main()
