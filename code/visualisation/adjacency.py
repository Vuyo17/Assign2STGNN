"""Adjacency-matrix figures: the predefined METR-LA graph (Q1/fig01), the learned
adaptive graph (Q2/fig07), and a predefined-vs-learned comparison (Q2/fig08).
"""
from __future__ import annotations

import numpy as np

from code.visualisation.style import (
    DIVERGING_BLUE_RED,
    INK_PRIMARY,
    SEQUENTIAL_BLUE,
    apply_axis_style,
    new_figure,
    save_figure,
)


def plot_adjacency_heatmap(
    adj: np.ndarray,
    save_path: str,
    title: str,
    n_labelled_ticks: int = 9,
    cbar_label: str = "Edge weight (normalised similarity, unitless)",
) -> None:
    """Full NxN adjacency heatmap with a labelled colourbar and node-index ticks.

    ``adj[i, j]`` is read as the weight used when node i aggregates information
    FROM node j (row i = destination/target, column j = origin/source) -- this
    is tsl's `edge_index_to_adj` convention (it builds `adj[src, dst] = weight`
    then returns the TRANSPOSE), verified directly against the installed
    library's source rather than assumed, and it matches the row=target,
    column=source convention GraphWaveNet's learned adjacency also uses (see
    `code/visualisation/learned_adjacency.py`).
    """
    n = adj.shape[0]
    fig, ax = new_figure(figsize=(8, 7))

    im = ax.imshow(adj, cmap=SEQUENTIAL_BLUE, aspect="auto", interpolation="nearest",
                    origin="upper", vmin=0)

    ticks = np.linspace(0, n - 1, n_labelled_ticks, dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    ax.set_xlabel("Source (origin) sensor node index")
    ax.set_ylabel("Target (destination) sensor node index")
    ax.set_title(title, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color=INK_PRIMARY)
    cbar.ax.tick_params(colors=INK_PRIMARY)

    # Grid lines from imshow's own axes are distracting on a dense heatmap.
    ax.grid(False)

    save_figure(fig, save_path)


def plot_predefined_vs_learned(
    predefined: np.ndarray,
    learned: np.ndarray,
    save_path: str,
    title: str = "Predefined vs. Learned Adaptive Adjacency (first 50 nodes)",
) -> None:
    """Three-panel comparison: predefined | learned | difference (learned - predefined,
    both min-max normalised to [0, 1] first so the two differently-scaled matrices
    are visually comparable)."""
    import matplotlib.pyplot as plt

    def _norm(m):
        m = np.asarray(m, dtype=float)
        rng = m.max() - m.min()
        return (m - m.min()) / rng if rng > 0 else m

    pre_n, learn_n = _norm(predefined), _norm(learned)
    diff = learn_n - pre_n

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax in axes:
        apply_axis_style(ax)
        ax.grid(False)

    im0 = axes[0].imshow(pre_n, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    axes[0].set_title("Predefined graph (distance-based)", fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(learn_n, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    axes[1].set_title("Learned adaptive graph", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap=DIVERGING_BLUE_RED, vmin=-1, vmax=1)
    axes[2].set_title("Difference (learned − predefined)", fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Source (origin) node index")
    axes[0].set_ylabel("Target (destination) node index")

    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK_PRIMARY)
    save_figure(fig, save_path)
