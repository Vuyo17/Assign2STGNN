"""Recovery path: evaluate TTS from its already-saved best checkpoint, used
when the live training process was deliberately stopped early (deadline-driven
resource reallocation -- see PROGRESS_LOG.md ~17:46) rather than left running
to completion. Rebuilds the exact same architecture from config, loads the
checkpoint's weights directly (bypassing Lightning's hparam-based
auto-reconstruction, which is unreliable for a manually-passed nn.Module --
see code/experiments/train.py's docstring reasoning), and runs the normal
evaluation pipeline.

Run: .venv/Scripts/python.exe -m code.experiments.evaluate_tts_checkpoint
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import torch
import yaml

from code.data.datamodule import DataConfig, build_datamodule
from code.evaluation.evaluate import run_evaluation
from code.models.tts import TimeThenSpaceModel
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding

_CONFIG_PATH = "code/configs/tts.yaml"


def main():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log = ProgressLogger("tts")
    ckpts = sorted(glob.glob("results/tts/checkpoints/best-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found under results/tts/checkpoints/ -- "
                                    "was training stopped before any epoch completed?")
    ckpt_path = ckpts[-1]
    log.milestone(f"=== Q1 TTS: evaluating from existing checkpoint {ckpt_path} "
                     f"(training deliberately stopped early -- deadline resource reallocation) ===")

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
    dm.setup()

    model = TimeThenSpaceModel(
        input_size=torch_dataset.n_channels, n_nodes=torch_dataset.n_nodes,
        horizon=torch_dataset.horizon,
        hidden_size=cfg["model"]["hidden_size"], rnn_layers=cfg["model"]["rnn_layers"],
        gnn_kernel=cfg["model"]["gnn_kernel"],
    )

    from tsl.engines import Predictor
    from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE

    predictor = Predictor(
        model=model, optim_class=torch.optim.Adam, optim_kwargs={"lr": cfg["optim"]["lr"]},
        loss_fn=MaskedMAE(),
        metrics={"mae": MaskedMAE(), "mse": MaskedMSE(), "mape": MaskedMAPE(),
                 "mae_at_15": MaskedMAE(at=2), "mae_at_30": MaskedMAE(at=5), "mae_at_60": MaskedMAE(at=11)},
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    predictor.load_state_dict(checkpoint["state_dict"])
    log.milestone(f"Loaded weights from {ckpt_path} (checkpoint epoch info: "
                     f"{checkpoint.get('epoch')}, global_step: {checkpoint.get('global_step')})")

    # Reconstruct training_history.json from PyTorch Lightning's CSVLogger
    # output (flushed to disk incrementally, so it survived the kill), and
    # best_val_mae from that same data -- both real, not fabricated, even
    # though the live process that would normally have written these itself
    # was stopped early.
    import glob as _glob
    import pandas as pd

    csv_candidates = sorted(_glob.glob("logs/tts/version_*/metrics.csv"),
                                key=lambda p: Path(p).stat().st_mtime)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_mae = None
    if csv_candidates:
        df = pd.read_csv(csv_candidates[-1])
        for epoch, g in df.groupby("epoch"):
            history["epoch"].append(int(epoch))
            history["train_loss"].append(
                float(g["train_loss"].dropna().mean()) if g["train_loss"].notna().any() else None)
            history["val_loss"].append(
                float(g["val_mae"].dropna().mean()) if g["val_mae"].notna().any() else None)
        valid_val = [v for v in history["val_loss"] if v is not None]
        best_val_mae = min(valid_val) if valid_val else None
    with open("results/tts/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "experiment": "tts",
        "epochs_run": len(history["epoch"]),
        "best_val_mae": best_val_mae,
        "best_checkpoint": ckpt_path,
        "total_training_seconds": None,  # not recoverable -- process was killed, not finished normally
        "seconds_per_epoch_avg": None,
        "note": ("Training was deliberately stopped early (after 3 epochs) to "
                 "reallocate CPU capacity to the more expensive GWN/AGCRN runs "
                 "under the submission deadline -- see PROGRESS_LOG.md ~17:46. "
                 "total_training_seconds is not available because the live "
                 "process (which was tracking it) was killed rather than "
                 "finishing normally; per-epoch times ARE available in "
                 "PROGRESS_LOG.md's [tts] entries (epoch 0: 232.2s, epoch 1: "
                 "490.5s, epoch 2: 494.9s -- all real, logged during training)."),
    }
    Path("results/tts").mkdir(parents=True, exist_ok=True)
    with open("results/tts/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    status_mod.set_stage("tts_eval", "running")
    overall = run_evaluation(predictor, dm, "tts", n_nodes=torch_dataset.n_nodes, out_dir="results/tts")
    status_mod.set_stage("tts_eval", "done", notes=f"60min MAE={overall['60min']['mae']:.3f} (from 3-epoch checkpoint)")
    status_mod.set_stage("tts_train", "done", notes="Deliberately stopped after 3 epochs (deadline resource reallocation)")
    regenerate_outstanding()

    log.milestone(f"=== Q1 TTS: COMPLETE (from checkpoint). Overall metrics: {overall} ===")
    print("\nDone. See results/tts/metrics.json")


if __name__ == "__main__":
    main()
