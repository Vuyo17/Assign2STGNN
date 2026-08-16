# Outstanding / Status Board

**Auto-generated** by `code/utils/update_outstanding.py` -- do not hand-edit; it is overwritten after every pipeline stage. Reflects `results/status.json` cross-checked against files actually present on disk.

Progress: **0/28** stages complete.

Last regenerated: 2026-08-16 16:07:08

## Setup

- ⬜ **env_setup**: Python venv + torch/tsl/PyG installed, versions recorded
- ⬜ **data_pipeline_verified**: METR-LA loaded, splits/scaler verified (no leakage)
- ⬜ **timing_pilot**: CPU timing pilot run for all 4 architectures

## Q1 TimeThenSpace

- ⬜ **tts_train**: TTS trained to convergence (early stopping)
- ⬜ **tts_eval**: TTS evaluated: overall + per-horizon + per-node metrics
- ⬜ **fig_adjacency_heatmap**: Predefined adjacency matrix heatmap
- ⬜ **fig_tts_overall**: TTS overall performance table + horizon-trend chart
- ⬜ **fig_tts_per_station**: TTS actual-vs-predicted, sensors 1-3

## Q2 GraphWaveNet

- ⬜ **gwn_predefined_train**: GWN (predefined graph only) trained
- ⬜ **gwn_adaptive_train**: GWN (predefined + adaptive) trained
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
- ⬜ **agcrn_train**: AGCRN final training run
- ⬜ **agcrn_eval**: AGCRN evaluated: overall + per-horizon + per-node
- ⬜ **fig_agcrn_vs_gwn**: AGCRN vs GWN performance/training-time comparison
- ⬜ **fig_agcrn_per_station**: AGCRN vs best GWN, per-station (nodes 1-3)
- ⬜ **weather_paper_comparison**: Written reflection vs Gaibie et al. 2024 weather paper

## Synthesis

- ⬜ **final_synthesis**: Cross-model final comparison + overall discussion drafted

## Report

- ⬜ **report_written**: report/report.md written (all required sections)
- ⬜ **report_pdf**: report.md rendered to PDF
- ⬜ **final_zip**: Final [id][surname][initials].zip assembled (PDF at root)
