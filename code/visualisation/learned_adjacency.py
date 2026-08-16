"""Learned adaptive-adjacency analysis: heatmap of the first 50 nodes, an
explicitly-defined influence score, and the top-15 influential nodes table
(Q2 section 2.3 of the assignment).

Influence-score definition -- and why -- verified directly against the
installed tsl 0.9.5 source (not assumed):

`GraphWaveNetModel.get_learned_adj()` computes
``adj = softmax(relu(source_embeddings() @ target_embeddings().T), dim=1)``,
i.e. **each row is softmax-normalised over its columns, so every row sums to
~1 regardless of that node's actual importance** -- row sums are a
normalisation artefact, not a meaningful signal.

This `adj` matrix is then consumed by `DenseGraphConvOrderK.forward` via
``torch.einsum('ncvl, wv -> ncwl', (x, a))``, i.e. the *new* representation of
node `w` (a row index of `a`) is a weighted sum over `v` (a column index of
`a`) of node `v`'s current representation: ``x_new[w] = sum_v a[w, v] * x[v]``.
So **row `i` = the destination/target node being updated, column `j` = the
origin/source node contributing to it** -- `adj[i, j]` is "how much weight
node i's update gives to node j".

Given that, and given rows are normalised to ~1 (making row sums useless),
the only quantity that meaningfully varies across nodes and captures "how much
this node influences the rest of the graph" is the **column sum**: how much
total (un-normalised) weight node j contributes across *all* of its targets --
`influence(j) = sum_i adj[i, j] for i != j`. A node with a high column sum is
one whose state is heavily weighted by many other nodes when they update --
exactly the graph-convolution-relevant notion of an "influential" node. The
row sum is reported alongside, clearly labelled, purely as a diagnostic (it
should sit close to 1 for every node by construction; a node deviating
noticeably would indicate `include_self`/numerical edge cases worth flagging,
not genuine importance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from code.visualisation.style import INK_PRIMARY, SEQUENTIAL_BLUE, new_figure, save_figure


def compute_influence_scores(adj: np.ndarray) -> pd.DataFrame:
    """Returns a DataFrame indexed by node id with columns:
    ``influence`` (column sum excluding self -- the primary, justified score;
    total weight node j contributes to all other nodes' updates),
    ``row_sum_diagnostic`` (row sum including self -- expected to be ~1 for
    every node by construction; not used for ranking), and
    ``influence_normalised`` (influence / max influence).
    """
    n = adj.shape[0]
    a = adj.copy().astype(float)
    a_no_self = a.copy()
    np.fill_diagonal(a_no_self, 0.0)

    influence = a_no_self.sum(axis=0)      # column sum: weight node j CONTRIBUTES to others
    row_sum_diagnostic = a.sum(axis=1)     # should be ~1 everywhere (softmax normalisation)

    df = pd.DataFrame({
        "node_id": np.arange(n),
        "influence": influence,
        "row_sum_diagnostic": row_sum_diagnostic,
    })
    max_inf = df["influence"].max()
    df["influence_normalised"] = df["influence"] / max_inf if max_inf > 0 else 0.0
    return df


def top_k_influential_nodes(adj: np.ndarray, k: int = 15, top_targets: int = 3) -> pd.DataFrame:
    """Top-k nodes by `influence` (column sum -- see module docstring), with,
    for each, the `top_targets` nodes whose update weights it most strongly
    (highest `adj[target, source_j]` values in source_j's column)."""
    n = adj.shape[0]
    a = adj.copy().astype(float)
    a_no_self = a.copy()
    np.fill_diagonal(a_no_self, 0.0)

    scores = compute_influence_scores(adj).sort_values("influence", ascending=False)
    top = scores.head(k).reset_index(drop=True)

    most_influenced = []
    for node_id in top["node_id"]:
        col = a_no_self[:, node_id]   # weight this source contributes to each target row
        target_idx = np.argsort(col)[::-1][:top_targets]
        pairs = [f"node {i} (w={col[i]:.3f})" for i in target_idx if col[i] > 0]
        most_influenced.append(", ".join(pairs) if pairs else "none")

    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["most_influenced_nodes"] = most_influenced
    return top[["rank", "node_id", "influence", "influence_normalised",
                "row_sum_diagnostic", "most_influenced_nodes"]]


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
    ax.set_xlabel("Source (origin) node index (0-49)")
    ax.set_ylabel("Target (destination) node index (0-49)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Learned edge weight (post row-softmax, unitless)", color=INK_PRIMARY)
    cbar.ax.tick_params(colors=INK_PRIMARY)
    save_figure(fig, save_path)
