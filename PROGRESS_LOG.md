# Progress Log

Auto-updated running log of every experiment/script executed for this assignment. Newest entries at the bottom. Full verbose logs live under `logs/<run_name>.log`.

- [session start] Project scaffold created (code/, results/, figures/, logs/, report/). Plan approved. Beginning Phase 0 (environment setup).
- [2026-08-16 16:23:22] [env_setup] Discovered tsl==0.9.5 hard-imports torch_scatter (no optional fallback in this version). No prebuilt torch_scatter/torch_sparse Windows wheel exists for Python 3.13, and no MSVC compiler is available on this machine to build from source.
- [2026-08-16 16:23:22] [env_setup] Resolution: installed Python 3.12 (via winget) alongside 3.13, rebuilt the project venv on 3.12, and pinned torch==2.9.0+cpu -- the newest CPU torch version with published Windows cp312 wheels for torch_scatter/torch_sparse on data.pyg.org. Re-installing full stack now.
