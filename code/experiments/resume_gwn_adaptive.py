"""Resumes GraphWaveNet (predefined + adaptive graph) training from its
existing best checkpoint (epoch 2, val_mae=2.830) rather than restarting
from scratch. Same reasoning as resume_gwn_predefined.py; kept as an
identical epoch budget/patience to that script so the two GWN configs
remain a controlled ablation (only `learned_adjacency` differs) even in the
resumed runs. Re-extracts and overwrites `results/gwn_adaptive/
learned_adjacency.npy` from the resumed model afterwards, exactly as the
original run_gwn_adaptive.py does.

Run: .venv/Scripts/python.exe -m code.experiments.resume_gwn_adaptive
"""
from __future__ import annotations

import numpy as np
import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.gwn import build_gwn_model, extract_learned_adjacency
from code.experiments.train import run_training
from code.experiments.resume_utils import find_best_checkpoint, epoch_of
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/gwn_adaptive.yaml"

NEW_MAX_EPOCHS = 60
NEW_PATIENCE = 12


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("gwn_adaptive")
    ckpt_path = find_best_checkpoint("gwn_adaptive")
    log.milestone(
        f"=== Q2 GWN (predefined+adaptive): RESUMING from {ckpt_path} (epoch "
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
        model, dm, "gwn_adaptive",
        lr=cfg["optim"]["lr"], max_epochs=NEW_MAX_EPOCHS,
        early_stopping_patience=NEW_PATIENCE, seed=cfg["seed"],
        resume_from_checkpoint=ckpt_path,
    )

    status_mod.set_stage("gwn_adaptive_eval", "running")
    overall = run_evaluation(predictor, dm, "gwn_adaptive", n_nodes=torch_dataset.n_nodes,
                                 out_dir="results/gwn_adaptive")
    status_mod.set_stage("gwn_adaptive_eval", "done",
                             notes=f"60min MAE={overall['60min']['mae']:.3f} (resumed run, GPU)")

    learned_adj = extract_learned_adjacency(predictor.model)
    np.save("results/gwn_adaptive/learned_adjacency.npy", learned_adj)
    log.milestone(
        f"Saved learned adjacency matrix ({learned_adj.shape}) to "
        f"results/gwn_adaptive/learned_adjacency.npy (resumed run)"
    )

    regenerate_outstanding()
    log.milestone(f"=== Q2 GWN (predefined+adaptive): RESUME COMPLETE. Summary: {summary} ===")
    print("\nDone. See results/gwn_adaptive/metrics.json, learned_adjacency.npy")


if __name__ == "__main__":
    main()
