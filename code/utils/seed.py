"""Central random-seed control so every experiment in this project is reproducible.

Call ``set_seed(SEED)`` once at the start of any script (data prep, training,
evaluation, plotting with stochastic elements) before any random number is drawn.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

#: Default project-wide seed. Overridden per-experiment via config files, but this
#: is the value used unless a config explicitly specifies another.
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python's ``random``, NumPy, PyTorch (CPU + CUDA if present), and set
    the ``PYTHONHASHSEED`` env var so hash-based iteration order is fixed too.

    Also tries to seed PyTorch Lightning's ``seed_everything`` if it is
    importable, which additionally seeds dataloader worker processes.

    Returns the seed used, for logging purposes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in this environment
        torch.cuda.manual_seed_all(seed)

    try:
        from pytorch_lightning import seed_everything

        seed_everything(seed, workers=True)
    except ImportError:
        pass

    return seed
