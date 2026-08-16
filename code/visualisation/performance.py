"""Overall-performance tables and horizon-trend charts (Q1 fig02, Q2 fig04, Q3 fig09).

Expects metrics in the shape produced by `code/evaluation/evaluate.py`:

    metrics_by_model = {
        "TTS": {"15min": {"mse": .., "mae": .., "mape": ..},
                "30min": {...}, "60min": {...}},
        "GWN (predefined)": {...},
        ...
    }

MAPE is stored as a fraction (e.g. 0.081); it is only ever multiplied by 100 for
display, at the last possible step, so no intermediate table double-scales it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from code.visualisation.style import (
    GRIDLINE,
    INK_PRIMARY,
    MODEL_ORDER,
    model_color,
    new_figure,
    save_figure,
)

HORIZON_ORDER = ["15min", "30min", "60min"]
HORIZON_LABELS = {"15min": "15 min", "30min": "30 min", "60min": "60 min"}
METRIC_UNITS = {"mse": "(mph²)", "mae": "(mph)", "mape": "(%)"}
METRIC_DISPLAY = {"mse": "MSE", "mae": "MAE", "mape": "MAPE"}


def build_overall_table(metrics_by_model: dict, decimals: int = 3) -> pd.DataFrame:
    """Long-format table: one row per (model, horizon), one column per metric.
    MAPE is expressed as a percentage in the table (matching the assignment's
    reporting convention), MSE/MAE stay in the original speed units (mph).
    """
    rows = []
    for model in metrics_by_model:
        for horizon in HORIZON_ORDER:
            if horizon not in metrics_by_model[model]:
                continue
            m = metrics_by_model[model][horizon]
            rows.append({
                "Model": model,
                "Horizon": HORIZON_LABELS[horizon],
                "MSE (mph²)": round(m["mse"], decimals),
                "MAE (mph)": round(m["mae"], decimals),
                "MAPE (%)": round(m["mape"] * 100, decimals),
            })
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = pd.DataFrame(rows)
    df["_order"] = df["Model"].map(lambda m: order.get(m, 99))
    df = df.sort_values(["_order", "Horizon"]).drop(columns="_order").reset_index(drop=True)
    return df


def plot_horizon_trend(
    metrics_by_model: dict,
    metric: str,
    save_path: str,
    title: str | None = None,
) -> None:
    """One line per model, MSE/MAE/MAPE vs. forecast horizon (15/30/60 min)."""
    fig, ax = new_figure(figsize=(7, 5))

    x = list(range(len(HORIZON_ORDER)))
    for model in [m for m in MODEL_ORDER if m in metrics_by_model] + \
                 [m for m in metrics_by_model if m not in MODEL_ORDER]:
        ys = []
        for h in HORIZON_ORDER:
            if h not in metrics_by_model[model]:
                ys.append(float("nan"))
                continue
            v = metrics_by_model[model][h][metric]
            ys.append(v * 100 if metric == "mape" else v)
        ax.plot(x, ys, marker="o", markersize=6, linewidth=2,
                 color=model_color(model), label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZON_ORDER])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(f"{METRIC_DISPLAY[metric]} {METRIC_UNITS[metric]}")
    ax.set_title(title or f"{METRIC_DISPLAY[metric]} vs. prediction horizon "
                           f"(averaged over 207 sensors)", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="best")

    save_figure(fig, save_path)


def plot_grouped_bar(
    metrics_by_model: dict,
    metric: str,
    save_path: str,
    title: str | None = None,
) -> None:
    """Grouped bar chart (one group per horizon, one bar per model) -- an
    alternative view of the same data as `plot_horizon_trend`, useful when the
    absolute gap between models at each horizon matters more than the trend."""
    import numpy as np

    models = [m for m in MODEL_ORDER if m in metrics_by_model] + \
             [m for m in metrics_by_model if m not in MODEL_ORDER]
    n_models = len(models)
    x = np.arange(len(HORIZON_ORDER))
    width = 0.8 / max(n_models, 1)

    fig, ax = new_figure(figsize=(7.5, 5))
    for i, model in enumerate(models):
        ys = []
        for h in HORIZON_ORDER:
            if h not in metrics_by_model[model]:
                ys.append(0)
                continue
            v = metrics_by_model[model][h][metric]
            ys.append(v * 100 if metric == "mape" else v)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width=width * 0.9, color=model_color(model),
               label=model, zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZON_ORDER])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(f"{METRIC_DISPLAY[metric]} {METRIC_UNITS[metric]}")
    ax.set_title(title or f"{METRIC_DISPLAY[metric]} by model and horizon", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    save_figure(fig, save_path)


def save_table(df: pd.DataFrame, csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
