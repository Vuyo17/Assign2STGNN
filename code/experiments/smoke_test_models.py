"""Fast sanity check (no training): build all 4 models and run one real forward
pass through one real batch from the datamodule, to catch shape/API mistakes
cheaply before committing to the (much slower) timing pilot or full training.

Run: .venv/Scripts/python.exe -m code.experiments.smoke_test_models
"""
from __future__ import annotations

import torch

from code.data.datamodule import DataConfig, build_datamodule
from code.models.agcrn import build_agcrn_model
from code.models.gwn import build_gwn_model, extract_learned_adjacency
from code.models.tts import TimeThenSpaceModel
from code.utils.seed import set_seed


def main():
    set_seed(42)
    dm, torch_dataset, dataset, connectivity = build_datamodule(DataConfig())
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    x = batch.input.x
    edge_index = batch.input.edge_index
    edge_weight = getattr(batch.input, "edge_weight", None)
    n_nodes = torch_dataset.n_nodes
    horizon = torch_dataset.horizon
    input_size = torch_dataset.n_channels

    print(f"batch x shape: {tuple(x.shape)}  (expect [batch, window, nodes, features])")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"n_nodes={n_nodes}, horizon={horizon}, input_size={input_size}")

    print("\n--- TTS ---")
    tts = TimeThenSpaceModel(input_size=input_size, n_nodes=n_nodes, horizon=horizon)
    out = tts(x, edge_index, edge_weight)
    print("output shape:", tuple(out.shape), " (expect [batch, horizon, nodes, features])")

    print("\n--- GWN (predefined only) ---")
    gwn_a = build_gwn_model(n_nodes=n_nodes, horizon=horizon, input_size=input_size,
                                learned_adjacency=False)
    out_a = gwn_a(x, edge_index, edge_weight)
    print("output shape:", tuple(out_a.shape))

    print("\n--- GWN (predefined + adaptive) ---")
    gwn_b = build_gwn_model(n_nodes=n_nodes, horizon=horizon, input_size=input_size,
                                learned_adjacency=True)
    out_b = gwn_b(x, edge_index, edge_weight)
    print("output shape:", tuple(out_b.shape))
    learned_adj = extract_learned_adjacency(gwn_b)
    print("learned adjacency shape:", learned_adj.shape,
            "row sums (first 5, should be ~1.0):", learned_adj.sum(axis=1)[:5])

    print("\n--- AGCRN ---")
    agcrn = build_agcrn_model(n_nodes=n_nodes, horizon=horizon, input_size=input_size)
    out_c = agcrn(x)  # no edge_index/edge_weight -- confirmed from forward signature
    print("output shape:", tuple(out_c.shape))

    print("\nAll 4 models built and ran a forward pass successfully.")


if __name__ == "__main__":
    main()
