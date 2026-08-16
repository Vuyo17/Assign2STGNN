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
[PENDING]
- Traffic forecasting motivation and challenges (non-Euclidean spatial structure,
  non-stationary temporal dynamics).
- STGNNs as a modelling paradigm: combining graph convolution with temporal sequence
  models.
- METR-LA dataset overview.
- Purpose and scope of these experiments (TTS baseline, GWN with/without adaptive
  adjacency, AGCRN; full 207-sensor dataset; consistent evaluation protocol).

## 2. Experimental Setup
[PENDING — will be generated from `environment.json`, `code/API_NOTES.md`, and the
config YAMLs under `code/configs/`, not retyped.]
- Dataset: full METR-LA (207 sensors).
- Preprocessing / window generation / train-val-test split (0.7/0.1/0.2, chronological).
- Scaling (StandardScaler, fit on training fold only — leakage check documented).
- Hardware & software environment (see `environment.json`).
- Metrics: MSE, MAE, MAPE, masked to exclude missing observations.
- Evaluation horizons: 15 / 30 / 60 minutes.
- Random seed and reproducibility notes.
- **Computational limitations statement:** this machine has no GPU; all training is
  CPU-only. Epoch budgets were set using an empirical timing pilot (Section 2.x) and
  early stopping rather than the assignment's implicit assumption of GPU-scale epoch
  counts, and this is stated explicitly rather than fabricating faster results.

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
[PENDING — from `code/API_NOTES.md` + `code/configs/gwn_*.yaml`]

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
