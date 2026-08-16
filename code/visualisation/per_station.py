"""Per-sensor actual-vs-predicted figures (Q1 fig03, Q2 fig06, Q3 fig10) and the
per-sensor error-vs-horizon view used alongside them.
"""
from __future__ import annotations

import numpy as np

from code.visualisation.style import INK_PRIMARY, MODEL_ORDER, model_color, new_figure, save_figure


def plot_timeseries_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    sensor_name: str,
    horizon_label: str,
    save_path: str,
    step_minutes: int = 5,
    n_points: int | None = 288,
) -> None:
    """Contiguous test-set window: actual vs. predicted traffic speed at a single
    forecast horizon (e.g. the 60-minute-ahead prediction), one line each.

    ``actual``/``predicted`` are 1-D arrays of consecutive test-set values for one
    sensor at one horizon step. Defaults to the first 288 points (= 1 day at a
    5-minute sampling interval) so the plot is readable rather than an unreadable
    smear across the whole test set.
    """
    if n_points is not None:
        actual = actual[:n_points]
        predicted = predicted[:n_points]

    t = np.arange(len(actual)) * step_minutes / 60.0  # hours

    fig, ax = new_figure(figsize=(10, 4.2))
    ax.plot(t, actual, color=INK_PRIMARY, linewidth=1.6, label="Actual", zorder=3)
    ax.plot(t, predicted, color=model_color(MODEL_ORDER[0]), linewidth=1.4,
             linestyle="--", label="Predicted", zorder=3, alpha=0.9)

    ax.set_xlabel("Time into test window (hours)")
    ax.set_ylabel("Traffic speed (mph)")
    ax.set_title(f"Actual vs. predicted speed — {sensor_name} ({horizon_label} horizon)",
                  fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    save_figure(fig, save_path)


def plot_error_vs_horizon_per_sensor(
    per_sensor_metrics: dict,
    sensor_name: str,
    metric: str,
    save_path: str,
) -> None:
    """``per_sensor_metrics``: {model_name: {"15min": value, "30min": value, "60min": value}}
    for a single sensor and a single metric (already extracted by the caller)."""
    from code.visualisation.performance import HORIZON_LABELS, HORIZON_ORDER, METRIC_DISPLAY, METRIC_UNITS

    fig, ax = new_figure(figsize=(6.5, 4.5))
    x = list(range(len(HORIZON_ORDER)))
    models = [m for m in MODEL_ORDER if m in per_sensor_metrics] + \
             [m for m in per_sensor_metrics if m not in MODEL_ORDER]
    for model in models:
        ys = [per_sensor_metrics[model].get(h, float("nan")) for h in HORIZON_ORDER]
        ys = [v * 100 if metric == "mape" and v == v else v for v in ys]
        ax.plot(x, ys, marker="o", markersize=6, linewidth=2,
                 color=model_color(model), label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZON_ORDER])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(f"{METRIC_DISPLAY[metric]} {METRIC_UNITS[metric]}")
    ax.set_title(f"{METRIC_DISPLAY[metric]} vs. horizon — {sensor_name}", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    save_figure(fig, save_path)
