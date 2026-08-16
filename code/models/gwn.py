"""GraphWaveNet builder for both Q2 configurations, using tsl's own
`GraphWaveNetModel` (`tsl.nn.models.GraphWaveNetModel`) with its DEFAULT
hyperparameters (per the assignment's instruction to use tsl defaults),
confirmed against the installed tsl==0.9.5 source via
`code/experiments/introspect_api.py` (see `code/API_NOTES.md`):

    GraphWaveNetModel(input_size, output_size, horizon, exog_size=0,
        hidden_size=32, ff_size=256, n_layers=8, temporal_kernel_size=2,
        spatial_kernel_size=2, learned_adjacency=True, n_nodes=None,
        emb_size=10, dilation=2, dilation_mod=2, norm='batch', dropout=0.3)

Only `learned_adjacency` (and `n_nodes`, required when it's True) are varied
between Config A and Config B -- every other hyperparameter is left at its tsl
default so the two configs are otherwise identical.

Config A (predefined graph only) still receives `edge_index`/`edge_weight`
from the datamodule at every forward call -- `learned_adjacency=False` only
disables the *additional* adaptive-adjacency branch (the dense spatial convs
using `get_learned_adj()`); the ordinary diffusion convs over the predefined
graph run in both configs. This is what makes A vs. B an apples-to-apples
ablation of the adaptive-adjacency component specifically, not a change to
how the predefined graph is used.
"""
from __future__ import annotations


def build_gwn_model(n_nodes: int, horizon: int, input_size: int = 1, output_size: int = 1,
                       learned_adjacency: bool = True):
    from tsl.nn.models import GraphWaveNetModel

    kwargs = dict(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        learned_adjacency=learned_adjacency,
    )
    if learned_adjacency:
        kwargs["n_nodes"] = n_nodes

    return GraphWaveNetModel(**kwargs)


def extract_learned_adjacency(model) -> "np.ndarray":
    """Only valid for a GWN model built with learned_adjacency=True. Returns the
    NxN learned adaptive adjacency as a detached numpy array (row = target/
    destination node, column = source/origin node -- see
    `code/visualisation/learned_adjacency.py` for the verified convention and
    the influence-score definition built on top of it)."""
    import torch

    if not hasattr(model, "source_embeddings") or model.source_embeddings is None:
        raise ValueError("This GWN model was built with learned_adjacency=False; "
                            "there is no learned adjacency to extract.")
    with torch.no_grad():
        return model.get_learned_adj().detach().cpu().numpy()
