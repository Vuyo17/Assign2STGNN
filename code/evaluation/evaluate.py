"""Model-agnostic test-set evaluation, shared by every experiment.

Each experiment's own training script builds its `predictor` (tsl `Predictor`
wrapping that experiment's model) and its `dm` (datamodule from
`code.data.datamodule.build_datamodule`, so the split/scaling/connectivity are
identical across experiments), then calls `run_evaluation(...)` here at the end
of training. Keeping evaluation in one place guarantees every model is scored
by literally the same code, on literally the same test samples.

Horizon convention (confirmed against the tsl tutorial notebook, cell 74):
0-indexed step 2 = 15 minutes ahead, step 5 = 30 minutes, step 11 = 60 minutes,
at a 5-minute sampling interval with a 12-step horizon.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from code.utils.progress_logger import ProgressLogger

HORIZON_STEPS = {"15min": 2, "30min": 5, "60min": 11}
EPS = 1e-5  # MAPE denominator guard


def _masked_metrics(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> dict:
    """y, yhat, mask: same-shape arrays; mask is boolean, True = valid observation."""
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return {"mse": float("nan"), "mae": float("nan"), "mape": float("nan"), "n": 0}
    err = yhat[valid] - y[valid]
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(y[valid]) + EPS)))
    return {"mse": mse, "mae": mae, "mape": mape, "n": int(valid.sum())}


@torch.no_grad()
def run_evaluation(
    predictor,
    dm,
    experiment_name: str,
    n_nodes: int,
    out_dir: Path | str,
    n_stations_to_save: int = 3,
) -> dict:
    """Runs the (already-trained, weights-loaded) predictor over the whole test
    set exactly once, computing:
      - overall metrics (MSE/MAE/MAPE) at 15/30/60 min, averaged over all 207 nodes
      - per-node metrics at 15/30/60 min (one row per sensor)
      - full-horizon actual/predicted arrays for the first `n_stations_to_save`
        sensors (for the per-station time-series figures)

    Saves `metrics.json`, `metrics_per_node.csv`, and `predictions.npz` under
    `out_dir`, and returns the overall-metrics dict (also used directly by the
    cross-model comparison figures/tables).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(experiment_name)

    predictor.eval()
    device = next(predictor.parameters()).device

    horizon_idx = sorted(HORIZON_STEPS.values())  # [2, 5, 11]

    # Accumulators. Kept as lists of small per-batch arrays instead of one giant
    # pre-allocated tensor -- simpler and memory-safe on this CPU-only machine.
    y_h_chunks, yhat_h_chunks, mask_h_chunks = [], [], []       # [.., len(horizon_idx), n_nodes]
    y_station_chunks, yhat_station_chunks, mask_station_chunks = [], [], []  # [.., 12, n_stations]

    t0 = time.time()
    n_batches = 0
    for batch in dm.test_dataloader():
        batch = batch.to(device)
        y, y_hat, mask = predictor.predict_batch(batch, postprocess=True, return_target=True)
        # shapes: [batch, horizon(12), nodes(207), features(1)]
        y = y[..., 0].cpu().numpy()
        y_hat = y_hat[..., 0].cpu().numpy()
        if mask is not None:
            mask = mask[..., 0].cpu().numpy().astype(bool)
        else:
            mask = np.ones_like(y, dtype=bool)

        y_h_chunks.append(y[:, horizon_idx, :])
        yhat_h_chunks.append(y_hat[:, horizon_idx, :])
        mask_h_chunks.append(mask[:, horizon_idx, :])

        y_station_chunks.append(y[:, :, :n_stations_to_save])
        yhat_station_chunks.append(y_hat[:, :, :n_stations_to_save])
        mask_station_chunks.append(mask[:, :, :n_stations_to_save])

        n_batches += 1

    elapsed = time.time() - t0
    logger.milestone(f"Evaluation pass over {n_batches} test batches complete in {elapsed:.1f}s")

    Y = np.concatenate(y_h_chunks, axis=0)       # [N_test, 3, n_nodes]
    YH = np.concatenate(yhat_h_chunks, axis=0)
    M = np.concatenate(mask_h_chunks, axis=0)

    Y_st = np.concatenate(y_station_chunks, axis=0)   # [N_test, 12, n_stations]
    YH_st = np.concatenate(yhat_station_chunks, axis=0)
    M_st = np.concatenate(mask_station_chunks, axis=0)

    # --- Overall metrics (averaged over all nodes) per horizon ---
    overall = {}
    for label, pos in zip(["15min", "30min", "60min"], range(len(horizon_idx))):
        overall[label] = _masked_metrics(Y[:, pos, :], YH[:, pos, :], M[:, pos, :])

    # --- Per-node metrics per horizon ---
    rows = []
    for node in range(n_nodes):
        row = {"node_id": node}
        for label, pos in zip(["15min", "30min", "60min"], range(len(horizon_idx))):
            m = _masked_metrics(Y[:, pos, node], YH[:, pos, node], M[:, pos, node])
            row[f"mse_{label}"] = m["mse"]
            row[f"mae_{label}"] = m["mae"]
            row[f"mape_{label}"] = m["mape"]
        rows.append(row)
    per_node_df = pd.DataFrame(rows)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": experiment_name, "overall": overall,
                     "n_test_samples": int(Y.shape[0]), "eval_seconds": elapsed}, f, indent=2)

    per_node_path = out_dir / "metrics_per_node.csv"
    per_node_df.to_csv(per_node_path, index=False)

    predictions_path = out_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        horizon_labels=np.array(["15min", "30min", "60min"]),
        y_at_horizons=Y, yhat_at_horizons=YH, mask_at_horizons=M,
        y_first_stations_full_horizon=Y_st, yhat_first_stations_full_horizon=YH_st,
        mask_first_stations_full_horizon=M_st,
    )

    logger.milestone(
        f"Saved metrics.json / metrics_per_node.csv / predictions.npz to {out_dir} "
        f"(overall 60min MAE={overall['60min']['mae']:.3f} mph)"
    )

    return overall
