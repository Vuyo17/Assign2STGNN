"""Idempotent figure/table generator: builds every figure/table it CAN from
whatever results currently exist under results/, and clearly logs what it
skipped (missing dependency) rather than fabricating anything. Safe to re-run
repeatedly as each training run finishes -- run it again any time to pick up
newly-completed experiments.

Run: .venv/Scripts/python.exe -m code.experiments.build_all_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from code.data.datamodule import DataConfig, build_datamodule
from code.utils.progress_logger import ProgressLogger

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results"
_FIGURES = _ROOT / "figures"
_FIGURES.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = {
    "tts": "TTS",
    "gwn_predefined": "GWN (predefined)",
    "gwn_adaptive": "GWN (predefined+adaptive)",
    "agcrn": "AGCRN",
}


def _has(experiment: str, filename: str) -> bool:
    return (_RESULTS / experiment / filename).exists()


def _load_overall_metrics() -> dict:
    metrics_by_model = {}
    for exp, display in DISPLAY_NAMES.items():
        p = _RESULTS / exp / "metrics.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            metrics_by_model[display] = data["overall"]
    return metrics_by_model


def _load_histories() -> dict:
    histories = {}
    for exp, display in DISPLAY_NAMES.items():
        p = _RESULTS / exp / "training_history.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                histories[display] = json.load(f)
    return histories


def _load_training_times() -> dict:
    times = {}
    for exp, display in DISPLAY_NAMES.items():
        p = _RESULTS / exp / "training_summary.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            times[display] = {
                "total_seconds": s["total_training_seconds"],
                "epochs": s["epochs_run"],
                "best_epoch_seconds": None,
            }
    return times


def main():
    log = ProgressLogger("build_all_figures")
    log.milestone("Building whatever figures/tables the currently-available results support...")

    # --- Predefined adjacency (fig01) -- only needs the datamodule, no trained model ---
    from tsl.ops.connectivity import edge_index_to_adj
    from code.visualisation.adjacency import plot_adjacency_heatmap, plot_predefined_vs_learned

    dm, torch_dataset, dataset, connectivity = build_datamodule(DataConfig())
    edge_index, edge_weight = connectivity
    predefined_adj = edge_index_to_adj(edge_index, edge_weight, num_nodes=torch_dataset.n_nodes)
    if hasattr(predefined_adj, "numpy"):
        predefined_adj = predefined_adj.numpy()
    np.save(_RESULTS / "predefined_adjacency.npy", predefined_adj)

    plot_adjacency_heatmap(
        predefined_adj, str(_FIGURES / "fig01_adjacency_heatmap.png"),
        title="Predefined METR-LA Sensor Adjacency Matrix (207 sensors, distance-based similarity)",
    )
    log.milestone("fig01_adjacency_heatmap.png done (predefined graph, always available)")

    metrics_by_model = _load_overall_metrics()
    log.milestone(f"Models with overall metrics available: {list(metrics_by_model.keys())}")

    if metrics_by_model:
        from code.visualisation.performance import build_overall_table, plot_horizon_trend, plot_grouped_bar, save_table

        table = build_overall_table(metrics_by_model)
        save_table(table, str(_RESULTS / "overall_performance_table.csv"))
        for metric in ["mse", "mae", "mape"]:
            plot_horizon_trend(metrics_by_model, metric, str(_FIGURES / f"fig04_horizon_trend_{metric}.png"))
        plot_grouped_bar(metrics_by_model, "mae", str(_FIGURES / "fig04_grouped_bar_mae.png"))
        log.milestone(f"Overall performance table + horizon-trend figures done for {list(metrics_by_model.keys())}")
    else:
        log.milestone("SKIPPED overall performance table/figures -- no metrics.json exists yet")

    # --- TTS per-station (fig03) ---
    if _has("tts", "predictions.npz"):
        from code.visualisation.per_station import plot_timeseries_actual_vs_predicted

        preds = np.load(_RESULTS / "tts" / "predictions.npz")
        Y_st, YH_st = preds["y_first_stations_full_horizon"], preds["yhat_first_stations_full_horizon"]
        for i, sensor in enumerate(["Sensor 1", "Sensor 2", "Sensor 3"]):
            plot_timeseries_actual_vs_predicted(
                Y_st[:, 11, i], YH_st[:, 11, i], sensor, "60 min",
                str(_FIGURES / f"fig03_tts_station{i+1}_actual_vs_predicted.png"),
            )
        log.milestone("fig03_tts_station{1,2,3}_actual_vs_predicted.png done")
    else:
        log.milestone("SKIPPED TTS per-station figures -- results/tts/predictions.npz not ready")

    # --- Convergence curves (fig05) ---
    histories = _load_histories()
    if histories:
        from code.visualisation.convergence import plot_convergence_curves

        plot_convergence_curves(histories, str(_FIGURES / "fig05_convergence_curves.png"))
        log.milestone(f"fig05_convergence_curves.png done for {list(histories.keys())}")
    else:
        log.milestone("SKIPPED convergence curves -- no training_history.json exists yet")

    # --- Training time bar (part of fig05/table) ---
    times = _load_training_times()
    if times:
        from code.visualisation.convergence import plot_training_time_bar

        plot_training_time_bar(times, str(_FIGURES / "fig_training_time.png"))
        pd.DataFrame([
            {"Model": m, "Total training time (min)": round(t["total_seconds"] / 60, 1), "Epochs run": t["epochs"]}
            for m, t in times.items()
        ]).to_csv(_RESULTS / "training_time_table.csv", index=False)
        log.milestone(f"Training time figure/table done for {list(times.keys())}")

    # --- Per-station comparison across models (fig06/fig10) ---
    available_predictions = {exp: display for exp, display in DISPLAY_NAMES.items() if _has(exp, "predictions.npz")}
    if len(available_predictions) >= 2:
        from code.visualisation.per_station import plot_error_vs_horizon_per_sensor

        for i, sensor in enumerate(["Sensor 1", "Sensor 2", "Sensor 3"]):
            per_sensor_metrics = {}
            for exp, display in available_predictions.items():
                df = pd.read_csv(_RESULTS / exp / "metrics_per_node.csv")
                row = df[df["node_id"] == i]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                per_sensor_metrics[display] = {
                    "15min": row["mae_15min"], "30min": row["mae_30min"], "60min": row["mae_60min"],
                }
            if per_sensor_metrics:
                plot_error_vs_horizon_per_sensor(
                    per_sensor_metrics, sensor, "mae",
                    str(_FIGURES / f"fig06_per_station_mae_{sensor.replace(' ', '').lower()}.png"),
                )
        log.milestone(f"Per-station comparison figures done for {list(available_predictions.values())}")
    else:
        log.milestone("SKIPPED per-station cross-model comparison -- fewer than 2 models have predictions.npz")

    # --- Learned adjacency (fig07, fig08, top-15 table) ---
    learned_adj_path = _RESULTS / "gwn_adaptive" / "learned_adjacency.npy"
    if learned_adj_path.exists():
        from code.visualisation.learned_adjacency import (
            plot_learned_adjacency_first_n, top_k_influential_nodes,
        )

        learned_adj = np.load(learned_adj_path)
        plot_learned_adjacency_first_n(learned_adj, str(_FIGURES / "fig07_learned_adjacency_heatmap.png"))
        top15 = top_k_influential_nodes(learned_adj, k=15)
        top15.to_csv(_RESULTS / "gwn_adaptive" / "top15_influential_nodes.csv", index=False)
        plot_predefined_vs_learned(predefined_adj[:50, :50], learned_adj[:50, :50],
                                       str(_FIGURES / "fig08_predefined_vs_learned.png"))
        log.milestone("fig07/fig08 + top15_influential_nodes.csv done")
    else:
        log.milestone("SKIPPED learned-adjacency figures/table -- results/gwn_adaptive/learned_adjacency.npy not ready")

    log.milestone("build_all_figures pass complete.")
    print("Done -- see logs/build_all_figures.log for what was built vs skipped this pass.")


if __name__ == "__main__":
    main()
