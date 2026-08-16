"""AGCRN builder (Q3), using tsl's own `AGCRNModel`
(`tsl.nn.models.AGCRNModel`) with its DEFAULT hyperparameters, confirmed
against the installed tsl==0.9.5 source (`code/API_NOTES.md`):

    AGCRNModel(input_size, output_size, horizon, n_nodes,
        hidden_size=64, emb_size=10, exog_size=0, n_layers=1)

Unlike TTS and both GWN configs, AGCRN's `forward(x, u=None)` takes NO
`edge_index`/`edge_weight` at all -- it never consumes the predefined METR-LA
distance graph. Spatial structure is captured entirely through its own
learned per-node embeddings (used internally by its adaptive graph
convolution cells), which is precisely the architectural contrast the
assignment asks Q3 to draw out against GWN's hybrid predefined+adaptive
approach.
"""
from __future__ import annotations


def build_agcrn_model(n_nodes: int, horizon: int, input_size: int = 1, output_size: int = 1):
    from tsl.nn.models import AGCRNModel

    return AGCRNModel(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        n_nodes=n_nodes,
    )
