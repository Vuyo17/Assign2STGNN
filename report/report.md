<!-- title: STUDENTID SURNAME INITIALS -->
# Spatial-Temporal Graph Neural Networks for Traffic Forecasting on METR-LA

**CSC5025 — Intelligent Systems — Assignment 2**

*Student ID / Surname / Initials: [PENDING — placeholder until provided]*
*Date: [PENDING]*

> **Status of this document:** this is a live, auto-assembled draft. Sections marked
> `[PENDING — <reason>]` have not yet been filled with real experimental output and
> must not be read as final. Every numeric result, figure, and table in the finished
> version is generated directly from files under `results/` and `figures/` — nothing
> in this report is hand-typed or estimated.

---

## Abstract
[PENDING — written last, once all experiments are complete.]

## 1. Introduction

Traffic forecasting — predicting quantities such as vehicle speed or flow at a set of
road-network sensors some minutes into the future — is a canonical case of
**spatio-temporal** prediction: the value at one sensor depends both on its own recent
history and on the state of *other* sensors connected to it by the road network. Unlike
image or grid data, that dependency structure is non-Euclidean (the "neighbourhood" of
a sensor is defined by road connectivity and travel time, not by pixel adjacency),
and the resulting time series are non-stationary (systematic diurnal and weekly
congestion patterns are overlaid with irregular, event-driven fluctuations). Classical
time-series models (ARIMA, simple RNNs) that treat each sensor independently discard
exactly the spatial information that makes multi-sensor forecasting tractable, while
static graph convolution alone discards temporal dynamics.

**Spatio-Temporal Graph Neural Networks (STGNNs)** address this by combining a graph
convolution component, which propagates information along a (predefined or learned)
adjacency structure, with a temporal component (recurrent, convolutional, or
attention-based) that models how each node's state evolves. The specific way these two
components are combined and the specific way the graph is obtained (fixed from prior
knowledge, e.g. road distance, vs. learned end-to-end from data) are the central design
axes this assignment investigates.

This report uses the **METR-LA** dataset — 207 loop-detector sensors on Los Angeles
County highways, recording average vehicle speed (mph) at 5-minute intervals — as
distributed through the **Torch Spatiotemporal (tsl)** library, and implements and
compares four STGNN configurations built on tsl's own model/layer primitives:

1. **TimeThenSpaceModel (TTS)** — an RNN-then-graph-convolution baseline, using tsl's
   `RNN`, `NodeEmbedding`, and `DiffConv` building blocks (Section 3).
2. **GraphWaveNet (GWN), predefined graph only** — `tsl.nn.models.GraphWaveNetModel`
   with `learned_adjacency=False` (Section 4, Configuration A).
3. **GraphWaveNet (GWN), predefined + learned adaptive adjacency** — the same model
   with `learned_adjacency=True` (Section 4, Configuration B).
4. **Adaptive Graph Convolutional Recurrent Network (AGCRN)** —
   `tsl.nn.models.AGCRNModel`, which learns its spatial structure entirely from data
   with no predefined graph input at all (Section 5).

The purpose of comparing configurations 2 and 3 specifically is to isolate the effect
of *learning* an adaptive adjacency matrix on top of a fixed, physically-derived graph
— holding every other hyperparameter, the data split, the preprocessing, the hardware,
and the training protocol constant. Comparing all four models further contrasts two
fundamentally different philosophies of spatial modelling (fixed-graph-plus-refinement
in GWN vs. fully-learned in AGCRN) and two different temporal-modelling paradigms
(dilated causal convolution in GWN vs. gated recurrence in AGCRN and TTS).

## 2. Experimental Setup

### 2.1 Dataset

The full **METR-LA** dataset (207 sensors, average speed in mph, 5-minute sampling
interval, 34,272 raw timesteps ≈ 4 months) is loaded via `tsl.datasets.MetrLA`, which
downloads and caches the raw HDF5 data on first run. Windowing (`tsl.data.
SpatioTemporalDataset`, `window=12`, `horizon=12`, `stride=1`) turns the raw series
into 34,249 input/target sample pairs (60 minutes of history → 60 minutes of forecast,
at 5-minute resolution, 12 steps each way).

### 2.2 Predefined Graph Construction

The predefined adjacency is built with `dataset.get_connectivity(threshold=0.1,
include_self=False, normalize_axis=1, layout="edge_index")`, tsl's standard
thresholded-Gaussian-kernel construction over pairwise road-network distances between
sensors (see Section 3.1 for the full description of what this matrix represents).
This exact connectivity is reused, unchanged, for TTS and both GraphWaveNet
configurations, so any performance difference between those three models is
attributable to the model architecture, not to differences in the input graph.

### 2.3 Splitting and Scaling (Leakage Control)

Samples are split **chronologically** (`tsl.data.datamodule.TemporalSplitter`,
`val_len=0.1`, `test_len=0.2`) into train/validation/test folds — train always precedes
validation, which always precedes test, with no shuffling across fold boundaries.
Verified realised split sizes (`code/experiments/verify_datapipeline.py`): **24,648 /
2,728 / 6,849** samples (train/val/test), i.e. **0.720 / 0.080 / 0.200** of the total —
close to but not exactly the configured 0.7/0.1/0.2, because a handful of samples whose
input/target window straddles a fold boundary are dropped by the splitter to prevent
leakage across that boundary; this shrinks the train fold's share slightly less than
val/test's (train only loses samples at the *end* of its range, while val/test lose
samples at the *start* of theirs, and the splitter also can't form full windows in the
very first `window` steps of the whole series). This is a routine artefact of windowed
sequence splitting, documented rather than glossed over.

Features are standardised with `tsl.data.preprocessing.StandardScaler(axis=(0, 1))`
(one global mean/std, shared across all nodes and timesteps, applied uniformly), fit
**only on the training fold** by `SpatioTemporalDataModule`. This was explicitly
verified rather than assumed: the fitted scaler's mean (`58.478` mph) matches a
train-fold-only computation, not the full-series mean (`58.368` mph) — the two differ,
confirming the split boundary is actually respected during fitting.

### 2.4 Model Configuration

All four models are built directly from tsl's own model classes with tsl's own default
hyperparameters (confirmed by introspecting the installed `tsl==0.9.5` package — see
`code/API_NOTES.md` — rather than assumed from documentation or online examples that
may not match this version):

- **TTS**: the tsl tutorial reference architecture (`RNN` (GRU, 1 layer) → `DiffConv`
  (k=2) over the predefined graph), `hidden_size=32`.
- **GWN (predefined)**: `GraphWaveNetModel(learned_adjacency=False)`, all other args at
  tsl defaults (`hidden_size=32, ff_size=256, n_layers=8, temporal_kernel_size=2,
  spatial_kernel_size=2, dilation=2, dilation_mod=2, norm='batch', dropout=0.3`).
- **GWN (predefined+adaptive)**: identical, with `learned_adjacency=True,
  n_nodes=207, emb_size=10` — the *only* difference from the previous configuration,
  by design, so this is a controlled ablation of the adaptive-adjacency component.
- **AGCRN**: `AGCRNModel(hidden_size=64, emb_size=10, n_layers=1)`, tsl defaults.
  AGCRN's `forward(x, u=None)` takes **no** `edge_index`/`edge_weight` at all — unlike
  the other three models, it never consumes the predefined graph; all of its spatial
  structure comes from its own learned node embeddings.

Optimiser (Adam), learning rate (0.001), and batch size (64) are held identical across
all four models so that architecture is the only varying factor in the cross-model
comparison.

### 2.5 Training Protocol, Hardware, and a Deadline-Driven Limitation

Training uses PyTorch Lightning (`tsl.engines.Predictor` wrapping each model,
`pytorch_lightning.Trainer`), masked MAE as the training loss, and `ModelCheckpoint`
(monitor `val_mae`, save best) + `EarlyStopping` (monitor `val_mae`) for model
selection.

**Hardware/software** (`environment.json`, captured automatically, not hand-typed):
Python 3.12.10, PyTorch 2.9.0+cpu, tsl 0.9.5, torch_geometric 2.8.0.post1, PyTorch
Lightning 2.6.5, **no GPU** (Intel64 Family 6 Model 191 CPU, 15.7 GB RAM, Windows 11).
Random seed 42, applied to Python/NumPy/PyTorch/PyTorch-Lightning via a single
`set_seed()` utility used by every script.

Two computational realities materially shaped the epoch budgets used for the final
results, and are stated explicitly rather than hidden:

1. **No GPU is available.** An empirical CPU timing pilot (one capped epoch per
   architecture, `results/timing_pilot.json`) measured real per-epoch cost before any
   full run was launched: TTS ≈2.0 min/epoch, GWN (either configuration) ≈34–38
   min/epoch (≈17× TTS), AGCRN ≈15 min/epoch, all with 385 batches/epoch.
2. **The four full training runs were executed concurrently** (on the user's own
   machine, in four parallel processes, to fit the assignment's submission deadline)
   rather than sequentially. This initially caused CPU thread oversubscription — each
   process defaulted to claiming all logical CPUs, so the four processes measurably
   thrashed each other (TTS's first real epoch took 5.6 min instead of the pilot's
   isolated 2.0 min, a 2.8× slowdown). This was mitigated by capping each process to a
   fair thread share (`cpu_count() // 4`), which reduced the slowdown to ≈1.9×, and
   final epoch budgets (Table with training times, Section 4.3/5.2) were set
   conservatively from that measured, *contended* per-epoch cost, not from the
   isolated pilot number.

Given (1) and (2), max-epoch ceilings are lower than would be used with unconstrained
time/GPU access (GWN: 6 epochs/config; AGCRN: 16 epochs; TTS: 60 epochs, effectively
unconstrained since it is cheap). Early stopping is still active as the primary
convergence criterion; the epoch *ceiling* is the deadline-driven limitation. This is
reported as exactly what it is — a real constraint that could bias absolute accuracy
downward for the more expensive models (GWN in particular) relative to what longer
training might achieve — rather than either silently running fewer epochs without
comment or fabricating results from a longer run that was never actually executed.

### 2.6 Metrics and Evaluation Protocol

All models are evaluated identically on the same (held-out, never-trained-on,
never-used-for-checkpoint-selection) test fold, using the **same** trained-model
loading and prediction code (`code/evaluation/evaluate.py`) for all four models:

- **MSE**, **MAE** (both in mph² / mph, the original unscaled speed units — predictions
  are inverse-transformed back from the standardised scale before computing metrics),
  and **MAPE** (%), each masked to exclude originally-missing observations.
- Reported at three horizons: **15 minutes** (step 3, 0-indexed step 2), **30 minutes**
  (step 6, index 5), **60 minutes** (step 12, index 11) — of the model's full 12-step
  (60-minute) forecast.
- Overall figures average over all 207 sensors; per-station figures (Sections 3.3, 4.5,
  5.3) isolate sensors 1–3 (node indices 0–2 in the dataframe's column order).

No hyperparameter was tuned against the test set at any point — model selection uses
only the validation fold (`val_mae`, via `EarlyStopping`/`ModelCheckpoint`), and the
test fold is touched exactly once per model, at final evaluation.

## 3. Question 1 — TimeThenSpaceModel
### 3.1 Adjacency Matrix
[PENDING — Figure 1, `figures/fig01_adjacency_heatmap.png`]

### 3.2 Overall Performance
[PENDING — Table 1 + Figure 2, from `results/tts/metrics.json`]

### 3.3 Per-Station Analysis
[PENDING — Figure 3, sensors 1-3, from `results/tts/predictions.npz`]

### 3.4 Discussion
[PENDING]

## 4. Question 2 — GraphWaveNet
### 4.1 Configurations

Both configurations use `tsl.nn.models.GraphWaveNetModel` (Wu et al., 2019) with tsl's
own default architecture hyperparameters (`hidden_size=32, ff_size=256, n_layers=8`
dilated-convolution blocks, `temporal_kernel_size=2, spatial_kernel_size=2, dilation=2,
dilation_mod=2, norm='batch', dropout=0.3`), identical optimiser/lr/batch size to TTS
(Section 2.4), and are trained/evaluated with exactly the same protocol
(Section 2.5–2.6). The two configurations differ in **exactly one** constructor
argument:

- **Configuration A (predefined only)**: `learned_adjacency=False`. The model still
  receives and uses the predefined graph (`edge_index`/`edge_weight`) at every forward
  pass through its diffusion-convolution (`DiffConv`) layers — `learned_adjacency`
  only controls whether the *additional* dense adaptive-adjacency branch (built from
  learned node embeddings) is present at all.
- **Configuration B (predefined + adaptive)**: `learned_adjacency=True, n_nodes=207,
  emb_size=10`. In addition to the predefined-graph diffusion convolutions, each of
  the 8 GWN blocks also runs a dense graph convolution over a **learned** NxN
  adjacency matrix, computed once per forward pass as
  `softmax(relu(E_src @ E_tgt^T), dim=1)` from two independently learned N×10 node
  embedding tables (`E_src`, `E_tgt`) — confirmed directly from the installed tsl
  source (`GraphWaveNetModel.get_learned_adj`), not assumed.

Because only this one argument differs, Configuration A vs. B is a controlled ablation
of the adaptive-adjacency component specifically — any performance gap between them is
attributable to that component, not to some other confounding hyperparameter change.

### 4.2 Overall Comparison (TTS vs GWN-predefined vs GWN-adaptive)
[PENDING — Table 2 + Figure 4]

### 4.3 Training Time and Convergence
[PENDING — Table 3 (training time) + Figure 5 (loss curves)]

### 4.4 Comparison With Wu et al. (2019)
[PENDING — explicit statement of what is/isn't comparable]

### 4.5 Per-Station Analysis
[PENDING — Figure 6, Table 4]

### 4.6 Learned Adaptive Adjacency Analysis
[PENDING — Figure 7 (heatmap, first 50 nodes), Table 5 (top-15 influential nodes)]

### 4.7 Predefined vs Learned Adjacency
[PENDING — Figure 8]

## 5. Question 3 — AGCRN
### 5.1 Epoch-Selection Experiment
[PENDING — `results/agcrn/epoch_selection.json`]

### 5.2 Overall Performance, Training Time, Convergence
[PENDING — Figure 9, Table 6]

### 5.3 Per-Station Analysis (AGCRN vs best GWN)
[PENDING — Figure 10]

### 5.4 Comparison With Gaibie et al. (2024)
[PENDING]

## 6. Overall Discussion
[PENDING — synthesised only after all four models are evaluated.]

## 7. Conclusion
[PENDING]

## References
1. Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C., 2019. Graph WaveNet for Deep
   Spatial-Temporal Graph Modeling. *arXiv:1906.00121*.
2. Bai, L., Yao, L., Li, C., Wang, X. and Wang, C., 2020. Adaptive Graph Convolutional
   Recurrent Network for Traffic Forecasting. *NeurIPS 33*, pp. 17804-17815.
3. Cini, A., Marisca, I., Zambon, D. and Alippi, C., 2023. Graph Deep Learning for Time
   Series Forecasting. *arXiv:2310.15978*.
4. Gaibie, A., Amir, H., Nandutu, I. and Moodley, D., 2024. Predicting and Discovering
   Weather Patterns in South Africa Using Spatial-Temporal Graph Neural Networks.
   *Southern African Conference for Artificial Intelligence Research*, pp. 144-160.
5. Torch Spatiotemporal documentation. https://torch-spatiotemporal.readthedocs.io/en/latest/
