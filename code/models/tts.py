"""TimeThenSpaceModel (TTS) -- the Q1 baseline.

`torch-spatiotemporal` does not ship a built-in class literally named
"TimeThenSpaceModel"; the tsl documentation/tutorials instead demonstrate a
canonical *pattern* for this architecture family (encode -> summarise time per
node with an RNN -> propagate across space with a graph conv -> decode), built
from tsl's own reusable blocks. This is exactly the implementation shown in
this project's `a_gentle_introduction_to_tsl.ipynb` (cell 67), which we treat as
the reference TTS implementation for this library and reuse verbatim (only the
docstring/formatting changed). Using tsl's own `RNN`, `NodeEmbedding`, and
`DiffConv` blocks -- rather than hand-rolled layers -- keeps this consistent
with how tsl expects spatiotemporal models to be composed, and with how GWN/
AGCRN (Q2/Q3) are also built on tsl primitives.
"""
from __future__ import annotations

import torch.nn as nn
from einops.layers.torch import Rearrange


class TimeThenSpaceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_nodes: int,
        horizon: int,
        hidden_size: int = 32,
        rnn_layers: int = 1,
        gnn_kernel: int = 2,
    ):
        super().__init__()

        from tsl.nn.blocks.encoders import RNN
        from tsl.nn.layers import DiffConv, NodeEmbedding

        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell="gru",
            return_only_last_state=True,
        )

        self.space_nn = DiffConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            k=gnn_kernel,
        )

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange("b n (t f) -> b t n f", t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch, time, nodes, input features]
        x_enc = self.encoder(x)

        # Add a learnable node-specific representation.
        x_emb = x_enc + self.node_embeddings()

        # Summarise the historical sequence at each node.
        # [batch, time, nodes, hidden] -> [batch, nodes, hidden]
        h = self.time_nn(x_emb)

        # Propagate the temporal summary across the sensor graph.
        h = self.space_nn(h, edge_index, edge_weight)

        # Decode into a per-node horizon-length forecast.
        out = self.decoder(h)
        return self.rearrange(out)
