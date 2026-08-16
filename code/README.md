# CSC5025 Assignment 2 — STGNN Traffic Forecasting — Code

Implementation of TimeThenSpaceModel (TTS), GraphWaveNet (predefined-graph-only and
predefined+adaptive), and AGCRN for traffic forecasting on the full METR-LA dataset,
using the `torch-spatiotemporal` (tsl) library.

## Reproducing the environment

```powershell
# Requires Python 3.12 (NOT 3.13 -- torch_scatter/torch_sparse have no Windows
# wheel for 3.13 at time of writing; see environment.json / PROGRESS_LOG.md for
# the full story of why this project pins 3.12).
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
.venv\Scripts\python.exe -m pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` (repo root) is a full `pip freeze` of the exact environment these
results were produced in, including `pandas==2.2.3` pinned specifically because
`tsl==0.9.5`'s METR-LA loader uses the deprecated pandas frequency alias `'5T'`,
which pandas 3.0 removes entirely (see `PROGRESS_LOG.md` ~17:29 for the diagnosis).

## Directory structure

```
code/
├── configs/          One YAML per experiment (tts, gwn_predefined, gwn_adaptive, agcrn) --
│                      hyperparameters, epoch budgets, and the reasoning behind them
├── data/              datamodule.py: the ONE shared METR-LA/split/scaler/connectivity
│                      builder every experiment uses, so all 4 models see identical data
├── models/            tts.py, gwn.py, agcrn.py -- thin builders around tsl's own model
│                      classes (GraphWaveNetModel, AGCRNModel) using tsl defaults
├── experiments/       Entry points: run_tts.py, run_gwn_predefined.py,
│                      run_gwn_adaptive.py, run_agcrn.py (each trains + evaluates one
│                      model end to end), plus pilot_timing.py, introspect_api.py,
│                      verify_datapipeline.py, smoke_test_models.py, build_all_figures.py,
│                      build_tables.py -- see each file's docstring
├── evaluation/        evaluate.py: shared, model-agnostic test-set evaluation
├── visualisation/      One module per figure family (style.py has the shared palette)
└── utils/              seed.py, env_report.py, progress_logger.py, status.py,
                        update_outstanding.py, build_report_pdf.py, build_report_docx.py,
                        package_submission.py
```

## Running an experiment

```powershell
.venv\Scripts\python.exe -m code.experiments.run_tts
.venv\Scripts\python.exe -m code.experiments.run_gwn_predefined
.venv\Scripts\python.exe -m code.experiments.run_gwn_adaptive
.venv\Scripts\python.exe -m code.experiments.run_agcrn
```

Each writes `results/<experiment>/{training_summary.json, training_history.json,
checkpoints/, metrics.json, metrics_per_node.csv, predictions.npz}` (plus
`learned_adjacency.npy` and `top15_influential_nodes.csv` for `gwn_adaptive`, and
`epoch_selection.json` for `agcrn`), and streams live progress to `PROGRESS_LOG.md`
(compact, project-wide) and `logs/<experiment>.log` (verbose, per-run).

After any experiment finishes, regenerate figures/tables from whatever is currently
available:

```powershell
.venv\Scripts\python.exe -m code.experiments.build_all_figures
.venv\Scripts\python.exe -m code.experiments.build_tables
```

Both are idempotent and safe to re-run repeatedly -- they build what they can from
current `results/` contents and clearly log what they skip.

## OUTSTANDING.md / PROGRESS_LOG.md

`OUTSTANDING.md` (repo root) is a machine-regenerated checklist of every deliverable
this assignment requires, cross-checked against `results/status.json` and what
actually exists on disk -- never hand-edited. `PROGRESS_LOG.md` is the append-only,
project-wide heartbeat log every script writes to. Both are intended to be read
directly (e.g. `Get-Content PROGRESS_LOG.md -Wait -Tail 20` while something is
training) rather than parsed programmatically.

## Building the report

```powershell
.venv\Scripts\python.exe -m code.utils.build_report_docx   # report/report.md -> report/report.docx
.venv\Scripts\python.exe -m code.utils.build_report_pdf    # report/report.md -> report/report.pdf (fallback)
.venv\Scripts\python.exe -m code.utils.package_submission --id <ID> --surname <SURNAME> --initials <XX>
```

## A note on this project's computational constraints

This project was run entirely on a CPU-only machine (no GPU) under a hard submission
deadline. `PROGRESS_LOG.md` and the comments in `code/configs/*.yaml` document, in
real time, the empirical timing pilot that was run before any full training, the CPU
thread-oversubscription issue discovered when training 4 models concurrently and how
it was fixed, and the resulting epoch-budget decisions. These are genuine, load-bearing
engineering decisions made under real constraints, not post-hoc rationalisation --
kept in the repo's history deliberately so the process is auditable.
