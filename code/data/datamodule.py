"""Build the METR-LA SpatioTemporalDataModule shared by every experiment.

Mirrors the pattern verified against the installed tsl version in the project's
own `a_gentle_introduction_to_tsl.ipynb` tutorial notebook (cells 12-55): load
`MetrLA`, build a predefined graph via `get_connectivity`, wrap into a
`SpatioTemporalDataset` (window/horizon/stride), then a
`SpatioTemporalDataModule` with a `TemporalSplitter` (chronological, no shuffling
across the boundary) and a `StandardScaler` that is fit ONLY on the training
fold -- this is what prevents validation/test leakage into preprocessing
statistics. This exact same datamodule (same seed, same connectivity, same
split) is reused for TTS, both GWN configs, and AGCRN so that all four models
are compared on identical data.

Every experiment MUST go through `build_datamodule()` below rather than
constructing its own pipeline, so the split/scaling/connectivity are
guaranteed identical across Q1/Q2/Q3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "data"


@dataclass
class DataConfig:
    window: int = 12          # 12 steps x 5 min = 60 min input history
    horizon: int = 12         # 12 steps x 5 min = 60 min forecast horizon
    stride: int = 1
    val_len: float = 0.1
    test_len: float = 0.2     # => train = 0.7 (matches assignment split)
    batch_size: int = 64
    connectivity_threshold: float = 0.1
    connectivity_include_self: bool = False
    connectivity_normalize_axis: int = 1
    scaler_axis: tuple = (0, 1)


def build_datamodule(cfg: DataConfig = DataConfig()):
    """Returns (datamodule, torch_dataset, raw_dataset, connectivity).

    `datamodule.setup()` is NOT called here -- callers should call it once they
    are ready (Lightning's Trainer.fit also calls it automatically). Kept
    separate so scripts can inspect dataset metadata (n_nodes, dataframe, dist
    matrix) before triggering the (potentially slow, disk-caching) setup.
    """
    from tsl.data import SpatioTemporalDataset
    from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
    from tsl.data.preprocessing import StandardScaler
    from tsl.datasets import MetrLA

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = MetrLA(root=str(_DATA_DIR))

    connectivity = dataset.get_connectivity(
        threshold=cfg.connectivity_threshold,
        include_self=cfg.connectivity_include_self,
        normalize_axis=cfg.connectivity_normalize_axis,
        layout="edge_index",
    )

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        mask=dataset.mask,
        window=cfg.window,
        horizon=cfg.horizon,
        stride=cfg.stride,
    )

    scalers = {"target": StandardScaler(axis=cfg.scaler_axis)}
    splitter = TemporalSplitter(val_len=cfg.val_len, test_len=cfg.test_len)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=cfg.batch_size,
    )

    return dm, torch_dataset, dataset, connectivity
