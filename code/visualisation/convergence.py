"""Training/validation loss convergence curves (Q2 fig05), and the training-time
summary bar used alongside timing comparisons.

Expects each model's PyTorch Lightning `CSVLogger` metrics.csv (read by the
caller into a small dict of arrays) rather than re-implementing CSV parsing
here, so this module stays a pure plotting layer.
"""
from __future__ import annotations

from code.visualisation.style import INK_SECONDARY, MODEL_ORDER, model_color, new_figure, save_figure


def plot_convergence_curves(
    histories: dict,
    save_path: str,
    title: str = "Training and validation loss convergence",
) -> None:
    """``histories``: {model_name: {"epoch": [...], "train_loss": [...], "val_loss": [...]}}.
    Train loss is dashed, validation loss is solid, both in the model's fixed colour,
    so the same figure lets you compare convergence speed and over/underfitting
    (train vs. val gap) across models at a glance.
    """
    fig, ax = new_figure(figsize=(8.5, 5.2))

    models = [m for m in MODEL_ORDER if m in histories] + \
             [m for m in histories if m not in MODEL_ORDER]
    for model in models:
        h = histories[model]
        color = model_color(model)
        if h.get("train_loss"):
            ax.plot(h["epoch"], h["train_loss"], color=color, linewidth=1.4,
                     linestyle="--", alpha=0.75, zorder=3,
                     label=f"{model} — train")
        if h.get("val_loss"):
            ax.plot(h["epoch"], h["val_loss"], color=color, linewidth=2.0,
                     linestyle="-", zorder=3, label=f"{model} — val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (masked MAE, standardised scale)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")

    save_figure(fig, save_path)


def plot_training_time_bar(training_times: dict, save_path: str) -> None:
    """``training_times``: {model_name: {"total_seconds": ..., "epochs": ..., "best_epoch_seconds": ...}}."""
    import numpy as np

    models = [m for m in MODEL_ORDER if m in training_times] + \
             [m for m in training_times if m not in MODEL_ORDER]
    totals_min = [training_times[m]["total_seconds"] / 60.0 for m in models]

    fig, ax = new_figure(figsize=(6.5, 4.5))
    x = np.arange(len(models))
    bars = ax.bar(x, totals_min, color=[model_color(m) for m in models],
                    zorder=3, edgecolor="white", linewidth=0.5)
    for rect, m in zip(bars, models):
        epochs = training_times[m].get("epochs")
        label = f"{epochs} epochs" if epochs is not None else ""
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 label, ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Total training time (minutes)")
    ax.set_title("Total training time to early-stopping (CPU)", fontsize=11, fontweight="bold")

    save_figure(fig, save_path)
