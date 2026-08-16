"""Shared figure style for every plot in this project.

Centralising this means all report figures share one consistent, deliberately
chosen (not matplotlib-default) palette: a fixed-order categorical palette for
comparing the 4 models, and a single-hue sequential ramp for magnitude heatmaps
(adjacency matrices, influence scores). Values below are the validated default
palette from the project's dataviz guidance (light-mode, print/PDF context --
this report has no dark-mode reader).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")  # headless (no display) -- safe for a CPU-only training box

# --- Chart chrome (light surface, print-friendly) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# --- Categorical palette, FIXED order -- never re-cycled per chart ---
# Used consistently for: TTS, GWN (predefined), GWN (adaptive), AGCRN, in that order.
CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"
CAT_AQUA = "#1baf7a"
CAT_YELLOW = "#eda100"
CATEGORICAL = [CAT_BLUE, CAT_ORANGE, CAT_AQUA, CAT_YELLOW]

MODEL_ORDER = ["TTS", "GWN (predefined)", "GWN (predefined+adaptive)", "AGCRN"]
MODEL_COLORS = dict(zip(MODEL_ORDER, CATEGORICAL))


def model_color(name: str) -> str:
    """Fixed colour for a model name; falls back to muted grey for anything
    outside the four canonical models so an unexpected label never silently
    reuses another model's colour."""
    return MODEL_COLORS.get(name, INK_MUTED)


# --- Sequential single-hue (blue) ramp, for heatmaps / magnitude ---
_BLUE_RAMP_STOPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP_STOPS)

# --- Diverging blue<->red, neutral grey midpoint ---
_DIVERGING_STOPS = ["#0d366b", "#6da7ec", "#f0efec", "#ec835a", "#8a1f1f"]
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list("div_blue_red", _DIVERGING_STOPS)


def apply_axis_style(ax) -> None:
    """Recessive gridlines/axes, consistent ink colours -- applied to every figure."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)


def new_figure(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    apply_axis_style(ax)
    return fig, ax


def save_figure(fig, path, dpi=200) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
