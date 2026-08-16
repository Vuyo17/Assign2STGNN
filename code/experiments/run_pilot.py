"""Driver for Phase 3: runs the capped timing pilot for all 4 architectures back
to back and prints a summary table. Real forward/backward passes, real wall-clock
timing -- just capped to a handful of batches so it finishes quickly and tells us
how long a FULL epoch/run will actually take on this CPU before we commit to it.

Run: .venv/Scripts/python.exe -m code.experiments.run_pilot
"""
from __future__ import annotations

from code.data.datamodule import DataConfig, build_datamodule
from code.models.agcrn import build_agcrn_model
from code.models.gwn import build_gwn_model
from code.models.tts import TimeThenSpaceModel
from code.experiments.pilot_timing import time_pilot
from code.utils.progress_logger import ProgressLogger
from code.utils.seed import set_seed
from code.utils import status as status_mod
from code.utils.update_outstanding import regenerate as regenerate_outstanding


def main():
    set_seed(42)
    log = ProgressLogger("timing_pilot")
    status_mod.set_stage("timing_pilot", "running")

    dm, torch_dataset, dataset, connectivity = build_datamodule(DataConfig())
    n_nodes = torch_dataset.n_nodes
    horizon = torch_dataset.horizon
    input_size = torch_dataset.n_channels

    results = {}

    log.milestone("=== Pilot 1/4: TTS ===")
    tts = TimeThenSpaceModel(input_size=input_size, n_nodes=n_nodes, horizon=horizon)
    results["TTS"] = time_pilot(tts, dm, "tts", n_train_batches=20, n_val_batches=5)

    log.milestone("=== Pilot 2/4: GWN (predefined) ===")
    gwn_a = build_gwn_model(n_nodes=n_nodes, horizon=horizon, input_size=input_size,
                                learned_adjacency=False)
    results["GWN (predefined)"] = time_pilot(gwn_a, dm, "gwn_predefined",
                                                 n_train_batches=20, n_val_batches=5)

    log.milestone("=== Pilot 3/4: GWN (predefined+adaptive) ===")
    gwn_b = build_gwn_model(n_nodes=n_nodes, horizon=horizon, input_size=input_size,
                                learned_adjacency=True)
    results["GWN (predefined+adaptive)"] = time_pilot(gwn_b, dm, "gwn_adaptive",
                                                            n_train_batches=20, n_val_batches=5)

    log.milestone("=== Pilot 4/4: AGCRN ===")
    agcrn = build_agcrn_model(n_nodes=n_nodes, horizon=horizon, input_size=input_size)
    results["AGCRN"] = time_pilot(agcrn, dm, "agcrn", n_train_batches=20, n_val_batches=5)

    print("\n\n=== TIMING PILOT SUMMARY ===")
    print(f"{'Model':<28} {'s/batch':>10} {'batches/epoch':>14} {'est. min/epoch':>16}")
    for name, r in results.items():
        print(f"{name:<28} {r['estimated_seconds_per_batch']:>10.2f} "
              f"{r['full_train_batches_per_epoch']:>14} "
              f"{r['estimated_full_epoch_minutes']:>16.1f}")

    status_mod.set_stage("timing_pilot", "done", notes=str({
        k: round(v["estimated_full_epoch_minutes"], 1) for k, v in results.items()
    }))
    regenerate_outstanding()
    log.milestone("Timing pilot complete for all 4 models.")


if __name__ == "__main__":
    main()
