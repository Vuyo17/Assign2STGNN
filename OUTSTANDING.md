# Outstanding / Status Board

**Auto-generated** by `code/utils/update_outstanding.py` -- do not hand-edit; it is overwritten after every pipeline stage. Reflects `results/status.json` cross-checked against files actually present on disk.

Progress: **9/28** stages complete.

Last regenerated: 2026-08-17 04:05:33

## Setup

- ✅ **env_setup**: Python venv + torch/tsl/PyG installed, versions recorded -- Python 3.12.10, torch 2.9.0+cpu, torch_geometric 2.8.0.post1, tsl 0.9.5, pandas pinned 2.2.3 (tsl 0.9.5 uses deprecated 5T freq alias removed in pandas 3.0), no GPU (Intel CPU, 15.7GB RAM)
- ✅ **data_pipeline_verified**: METR-LA loaded, splits/scaler verified (no leakage) -- 207 nodes; split 24648/2728/6849 (0.720/0.080/0.200 realised vs 0.7/0.1/0.2 configured, due to windowing edge effects); scaler.bias=58.478 (train-fold-only) vs full-series mean=58.368 -- confirms train-only fit
- ✅ **timing_pilot**: CPU timing pilot run for all 4 architectures -- {'TTS': 2.0, 'GWN (predefined)': 33.9, 'GWN (predefined+adaptive)': 37.6, 'AGCRN': 15.0}

## Q1 TimeThenSpace

- ✅ **tts_train**: TTS trained to convergence (early stopping) -- 101 epochs, 8.5 min, best_val_mae=2.924220561981201
- ✅ **tts_eval**: TTS evaluated: overall + per-horizon + per-node metrics -- 60min MAE=4.383 (resumed run, GPU)
- ⬜ **fig_adjacency_heatmap**: Predefined adjacency matrix heatmap
- ⬜ **fig_tts_overall**: TTS overall performance table + horizon-trend chart
- ⬜ **fig_tts_per_station**: TTS actual-vs-predicted, sensors 1-3

## Q2 GraphWaveNet

- ✅ **gwn_predefined_train**: GWN (predefined graph only) trained -- 26 epochs, 12.5 min, best_val_mae=2.685002326965332
- ✅ **gwn_adaptive_train**: GWN (predefined + adaptive) trained -- 16 epochs, 3.3 min, best_val_mae=2.6228384971618652
- ⬜ **gwn_eval**: Both GWN configs evaluated: overall + per-horizon + per-node
- ⬜ **fig_gwn_vs_tts**: TTS vs GWN-predefined vs GWN-adaptive comparison
- ⬜ **fig_convergence**: Training/validation convergence curves (TTS+GWN)
- ⬜ **fig_gwn_per_station**: Per-station comparison, nodes 1-3, all 3 models
- ⬜ **fig_learned_adjacency**: Learned adaptive adjacency heatmap (first 50 nodes)
- ⬜ **top15_influential_nodes**: Top-15 influential nodes table (defined influence score)
- ⬜ **fig_predefined_vs_learned**: Predefined vs learned adjacency comparison
- ⬜ **gwn_paper_comparison**: Written comparison with Wu et al. 2019 GWN paper

## Q3 AGCRN

- ⬜ **agcrn_epoch_selection**: Epoch-selection experiment + justification
- ✅ **agcrn_train**: AGCRN final training run -- 25 epochs, 5.6 min, best_val_mae=2.7074668407440186
- ✅ **agcrn_eval**: AGCRN evaluated: overall + per-horizon + per-node -- 60min MAE=3.673 (resumed run, GPU)
- ⬜ **fig_agcrn_vs_gwn**: AGCRN vs GWN performance/training-time comparison
- ⬜ **fig_agcrn_per_station**: AGCRN vs best GWN, per-station (nodes 1-3)
- ⬜ **weather_paper_comparison**: Written reflection vs Gaibie et al. 2024 weather paper

## Synthesis

- ⬜ **final_synthesis**: Cross-model final comparison + overall discussion drafted

## Report

- ⬜ **report_written**: report/report.md written (all required sections)
- ⬜ **report_pdf**: report.md rendered to PDF
- ⬜ **final_zip**: Final [id][surname][initials].zip assembled (PDF at root)
