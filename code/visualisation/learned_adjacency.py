"""Learned adaptive-adjacency analysis: heatmap of the first 50 nodes, an
explicitly-defined influence score, and the top-15 influential nodes table
(Q2 section 2.3 of the assignment).

Influence-score definition (documented here, not just in the report, so the
code and the write-up cannot drift apart):

GraphWaveNet's adaptive adjacency `A_adp = softmax(relu(E1 @ E2^T))` is a
directed, row-normalised (each row sums to ~1 after softmax) matrix, so
`A_adp[i, j]` is "how much node i's representation is updated FROM node j"
under one common convention, or the reverse under another -- the exact
direction depends on how it is multiplied against the node features inside
the model, which is confirmed empirically per the installed tsl source at
implementation time (see `code/API_NOTES.md`) rather than assumed here.

Given that direction, we define a node's **influence score** as its
**out-degree strength**: the sum of the edge weights it *sends* to all other
nodes, i.e. `influence[i] = sum_j A_adp[i, j] for j != i`. This is the natural
reading of "influence" for a directed weighted graph -- a node with high
row-sum is one whose state strongly informs many other nodes' updates, which
is the graph-convolution-relevant notion of importance (as opposed to
in-degree, which would instead measure how strongly a node is *listened to*
by others -- reported alongside as a secondary, clearly-labelled score rather
than conflated with the primary one).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from code.visualisation.style import INK_PRIMARY, SEQUENTIAL_BLUE, apply_axis_style, new_figure, save_figure


def compute_influence_scores(adj: np.ndarray) -> pd.DataFrame:
    """Returns a DataFrame indexed by node id with columns:
    out_influence (row sum, excluding self), in_influence (column sum,
    excluding self), and out_influence_normalised (out_influence / max).
    """
    n = adj.shape[0]
    a = adj.copy().astype(float)
    np.fill_diagonal(a, 0.0)  # exclude self-loops from the influence measure

    out_influence = a.sum(axis=1)   # row sum: weight node i SENDS to others
    in_influence = a.sum(axis=0)    # column sum: weight node i RECEIVES

    df = pd.DataFrame({
        "node_id": np.arange(n),
        "out_influence": out_influence,
        "in_influence": in_influence,
    })
    max_out = df["out_influence"].max()
    df["out_influence_normalised"] = df["out_influence"] / max_out if max_out > 0 else 0.0
    return df


def top_k_influential_nodes(adj: np.ndarray, k: int = 15, top_targets: int = 3) -> pd.DataFrame:
    """Top-k nodes by out_influence (see module docstring for the justification),
    with, for each, the `top_targets` most strongly-influenced other nodes
    (highest outgoing edge weight from that source)."""
    n = adj.shape[0]
    a = adj.copy().astype(float)
    np.fill_diagonal(a, 0.0)

    scores = compute_influence_scores(adj).sort_values("out_influence", ascending=False)
    top = scores.head(k).reset_index(drop=True)

    most_influenced = []
    for node_id in top["node_id"]:
        row = a[node_id]
        target_idx = np.argsort(row)[::-1][:top_targets]
        pairs = [f"node {j} (w={row[j]:.3f})" for j in target_idx if row[j] > 0]
        most_influenced.append(", ".join(pairs) if pairs else "none")

    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["most_influenced_nodes"] = most_influenced
    return top[["rank", "node_id", "out_influence", "out_influence_normalised", "most_influenced_nodes"]]


def plot_learned_adjacency_first_n(
    adj: np.ndarray,
    save_path: str,
    n_nodes: int = 50,
    title: str = "Learned Adaptive Adjacency Matrix (first 50 nodes)",
) -> None:
    sub = adj[:n_nodes, :n_nodes]
    fig, ax = new_figure(figsize=(7.5, 6.5))
    im = ax.imshow(sub, cmap=SEQUENTIAL_BLUE, aspect="auto", interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("Target node index (0-49)")
    ax.set_ylabel("Source node index (0-49)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Learned edge weight (post-softmax, unitless)", color=INK_PRIMARY)
    cbar.ax.tick_params(colors=INK_PRIMARY)
    save_figure(fig, save_path)
