"""CPU timing pilot (Phase 3 of the plan): for each of the 4 architectures, run a
handful of train/val batches for 1 epoch and extrapolate full-epoch /
full-training wall-clock time on THIS machine, before committing to any
multi-hour run. Never invents numbers -- everything here is a real (if capped)
forward/backward pass.

Usage: called once per model from a small driver after each model's wrapper is
implemented, e.g.:

    from code.experiments.pilot_timing import time_pilot
    time_pilot(model, dm, "tts", n_train_batches=20, n_val_batches=5)

Writes results/timing_pilot.json (accumulated across calls) and appends a
milestone to PROGRESS_LOG.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from code.experiments.train import run_training
from code.utils.progress_logger import ProgressLogger
from code.utils import status as status_mod

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_PATH = _ROOT / "results" / "timing_pilot.json"


def _n_batches(dataloader) -> int:
    try:
        return len(dataloader)
    except TypeError:
        return -1  # unknown (e.g. iterable-only dataset)


def time_pilot(
    model,
    dm,
    experiment_name: str,
    n_train_batches: int = 20,
    n_val_batches: int = 5,
    seed: int = 42,
) -> dict:
    log = ProgressLogger("timing_pilot")
    log.milestone(f"[{experiment_name}] starting capped 1-epoch pilot "
                    f"({n_train_batches} train / {n_val_batches} val batches)")

    dm.setup()
    full_train_batches = _n_batches(dm.train_dataloader())
    full_val_batches = _n_batches(dm.val_dataloader())

    t0 = time.time()
    _, _, summary = run_training(
        model, dm, f"{experiment_name}_pilot",
        max_epochs=1, early_stopping_patience=999, seed=seed,
        limit_train_batches=n_train_batches, limit_val_batches=n_val_batches,
    )
    wall = time.time() - t0

    capped_batches = min(n_train_batches, full_train_batches) if full_train_batches > 0 else n_train_batches
    sec_per_batch = wall / max(capped_batches, 1)
    est_full_epoch_sec = sec_per_batch * full_train_batches if full_train_batches > 0 else None

    result = {
        "experiment": experiment_name,
        "pilot_wall_seconds": wall,
        "capped_train_batches_run": capped_batches,
        "full_train_batches_per_epoch": full_train_batches,
        "full_val_batches_per_epoch": full_val_batches,
        "estimated_seconds_per_batch": sec_per_batch,
        "estimated_full_epoch_seconds": est_full_epoch_sec,
        "estimated_full_epoch_minutes": est_full_epoch_sec / 60 if est_full_epoch_sec else None,
    }

    all_results = {}
    if _RESULTS_PATH.exists():
        with open(_RESULTS_PATH, encoding="utf-8") as f:
            all_results = json.load(f)
    all_results[experiment_name] = result
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log.milestone(
        f"[{experiment_name}] pilot done: ~{sec_per_batch:.2f}s/batch, "
        f"{full_train_batches} batches/epoch => "
        f"est. {result['estimated_full_epoch_minutes']:.1f} min/epoch"
        if est_full_epoch_sec else f"[{experiment_name}] pilot done: {sec_per_batch:.2f}s/batch"
    )
    status_mod.set_stage("timing_pilot", "running", notes=f"{experiment_name} done")

    return result
