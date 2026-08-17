<!-- title: STUDENTID SURNAME INITIALS -->
# Spatial-Temporal Graph Neural Networks for Traffic Forecasting on METR-LA

**CSC5025, Intelligent Systems, Assignment 2**

*Student ID / Surname / Initials: [PENDING, placeholder until provided]*
*Date: 17 August 2026*

---

## Abstract

This report implements and compares four spatial-temporal graph neural network configurations on the full 207-sensor METR-LA traffic dataset: TimeThenSpaceModel (TTS), GraphWaveNet (GWN) with a predefined graph only, GWN with a predefined graph plus a learned adaptive graph, and AGCRN. All four use the Torch Spatiotemporal library's own model classes and default hyperparameters, under one shared data pipeline and evaluation protocol.

Training began on CPU under a submission deadline, then was **resumed** (not restarted) on GPU hardware once available, with materially larger epoch budgets. On current evidence: GWN's adaptive adjacency improves accuracy over the predefined-graph-only configuration at every horizon (0.6% at 15 min, growing to 5.8% at 60 min); and GWN-adaptive is now the strongest model overall at every horizon, narrowly ahead of AGCRN (0.6-2.7%) — a reversal of the CPU-era result, where AGCRN led. Training is still being extended for all four models at time of writing, so these standings, while the best currently available evidence, remain provisional.

## 1. Introduction

Traffic forecasting is a spatio-temporal prediction problem: a sensor's future speed depends on its own recent history *and* on connected sensors' states, via a non-Euclidean, road-network-defined neighbourhood. Spatial-temporal graph neural networks (STGNNs) combine graph convolution (propagation over a fixed or learned adjacency) with a temporal model. This report investigates four configurations on METR-LA (207 loop-detector sensors, 5-minute average speed):

1. **TTS** — GRU then diffusion convolution over the predefined graph (Section 3).
2. **GWN, predefined graph only** (Section 4A).
3. **GWN, predefined + learned adaptive graph** (Section 4B) — isolates the adaptive component vs. (2).
4. **AGCRN** — fully learned spatial structure, no predefined graph (Section 5).

Section 2 covers the shared setup. Sections 3-5 answer Q1-Q3. Section 6 synthesises; Section 7 concludes.

## 2. Experimental Setup

**Dataset & graph.** 207 sensors, 5-minute speed, ~34k timesteps → 34,249 windowed samples (12-step/60-min input, 12-step/60-min horizon). The predefined adjacency is a thresholded Gaussian kernel over road-network distance, row-normalised: sparse (1,515 of 42,849 entries non-zero, 3.5%), directed, asymmetric. Reused unchanged for TTS and both GWN configs.

**Split & scaling.** Chronological 0.7:0.1:0.2 split, realised as 24,648 / 2,728 / 6,849 samples (72.0/8.0/20.0%, minor deviation from windowing edge effects). Standard scaler fit on the training fold only — verified directly (train-fold mean 58.478 mph vs. full-series mean 58.368 mph), confirming no leakage.

**Model configuration** (all defaults confirmed against the installed `tsl==0.9.5` source):

| Model | Key hyperparameters |
|---|---|
| TTS | hidden size 32, 1 GRU layer, 1-hop diffusion conv |
| GWN (both configs) | hidden 32, ff 256, 8 dilated-conv blocks, kernel size 2, dropout 0.3; adaptive config adds `learned_adjacency=True`, `emb_size=10` |
| AGCRN | hidden 64, embedding size 10, 1 recurrent layer, no predefined graph |

Adam, lr=0.001, batch size 64, identical across all four models. Loss: masked MAE. Seed 42 throughout.

**Training protocol and a mid-project hardware change.** Training began CPU-only (Intel CPU, no GPU) under a hard deadline: 3 epochs for TTS (stopped manually, still improving), 4 for each GWN config, 11 for AGCRN. Partway through, GPU hardware became available (NVIDIA RTX 3070). Every model was **resumed from its CPU checkpoint** — full optimiser/epoch state, not just weights — on a CUDA-enabled environment, everything else held identical. Per-epoch cost dropped 50-100x (TTS ~7s, GWN ~58-64s, AGCRN ~30s, down from minutes-to-tens-of-minutes), so each model was given a much larger epoch ceiling and a fresh early-stopping patience. **This GPU training is still being extended at time of writing**; results below are the best currently available snapshot, not a final converged result — flagged once here rather than repeated throughout.

**Evaluation.** Same test fold, same code, all models. MSE and MAE in original mph units (inverse-scaled), plus MAPE, all masked for missing observations; reported at 15/30/60-min horizons. Overall = mean over 207 sensors; per-station = first 3 sensors. No test-set tuning; model selection via validation-fold early stopping only.

## 3. Question 1: TimeThenSpaceModel

![Predefined adjacency matrix heatmap](../figures/fig01_adjacency_heatmap.png)

*Figure 1. Predefined adjacency matrix (207 sensors). Row = destination, column = source.*

The graph is sparse (3.5% density), directed, and asymmetric, reflecting one-way segments and directional travel-time differences; each row sums to ~1 (self-excluded), so it encodes a relative weighting over each sensor's connected neighbours, not an absolute distance.

*Table 1. TTS overall performance (all 207 sensors), 100 epochs (3 CPU + 97 GPU).*

| Horizon | MSE (mph²) | MAE (mph) | MAPE (%) |
|---|---|---|---|
| 15 min | 32.792 | 2.946 | 7.78 |
| 30 min | 49.780 | 3.525 | 9.97 |
| 60 min | 76.577 | 4.383 | 13.18 |

![TTS horizon trend](../figures/fig04_horizon_trend_mae.png)

*Figure 2. TTS mean absolute error vs. horizon, averaged over all sensors.*

Error grows monotonically with horizon (MSE ×2.3, MAE ×1.5, MAPE ×1.7 from 15 to 60 min) — expected, since uncertainty compounds further ahead.

*Table 2. TTS per-station performance, Sensors 1-3.*

| Sensor | Horizon | MSE (mph²) | MAE (mph) | MAPE (%) |
|---|---|---|---|---|
| 1 | 15/30/60 min | 35.540 / 58.898 / 92.655 | 2.503 / 3.241 / 4.175 | 6.67 / 9.60 / 13.68 |
| 2 | 15/30/60 min | 7.803 / 8.398 / 8.960 | 1.589 / 1.641 / 1.737 | 2.72 / 2.83 / 2.98 |
| 3 | 15/30/60 min | 18.986 / 34.460 / 73.160 | 1.971 / 2.458 / 3.386 | 4.27 / 5.84 / 8.93 |

![TTS actual vs predicted, Sensor 1](../figures/fig03_tts_station1_actual_vs_predicted.png)

*Figure 3a-c. Actual vs. predicted speed, 60-min horizon, Sensors 1 (below), 2, 3 — see also the adjoining figures in `figures/`.*

![TTS actual vs predicted, Sensor 2](../figures/fig03_tts_station2_actual_vs_predicted.png)

![TTS actual vs predicted, Sensor 3](../figures/fig03_tts_station3_actual_vs_predicted.png)

Sensor difficulty varies sharply: Sensor 2 is easiest and flattest (MAE 1.59→1.74 mph, MAPE <3% even at 60 min — a stable, likely free-flowing segment). Sensor 1 is hardest, with the steepest horizon degradation (MAE nearly doubling, MAPE reaching 13.7%), consistent with a more volatile, congestion-prone segment. TTS tracks the coarse diurnal shape well; validation error was still fluctuating without a clean plateau at 100 epochs, so a modestly better result is plausible with further training.

## 4. Question 2: GraphWaveNet

Both configurations use tsl's `GraphWaveNetModel` (Wu et al., 2019) with library defaults; the *only* difference is Config B's additional learned-adjacency branch (`learned_adjacency=True`), making this a controlled ablation.

*Table 3. Overall performance, all three models so far.*

| Model | Horizon | MSE (mph²) | MAE (mph) | MAPE (%) |
|---|---|---|---|---|
| TTS | 15/30/60 min | 32.792/49.780/76.577 | 2.946/3.525/4.383 | 7.78/9.97/13.18 |
| GWN, predefined | 15/30/60 min | 28.252/41.773/61.930 | 2.792/3.252/3.873 | 7.20/8.95/11.32 |
| GWN, adaptive | 15/30/60 min | 27.277/38.775/53.810 | 2.774/3.170/3.650 | 7.35/8.86/10.73 |

TTS < GWN-predefined < GWN-adaptive at every horizon on MAE/MSE (MAPE is a near-tie between the two GWN configs, predefined edging ahead only at 15 min). The adaptive graph's MAE improvement over predefined-only grows with horizon: 0.6% → 2.5% → 5.8%. Full TTS→GWN-adaptive step: 18.6% MAE reduction at 60 min.

*Table 4. Training progress (GPU epoch cost excludes the earlier, much slower CPU epochs).*

| Model | Total epochs | Best val MAE | GPU epoch cost | Plateaued? |
|---|---|---|---|---|
| TTS | 100 (3+97) | 2.924 | ~7 s | No — hit ceiling |
| GWN, predefined | 25 (4+21)* | 2.685 | ~58 s | Early-stopped; extension in progress |
| GWN, adaptive | 15 (4+11)* | 2.623 | ~64 s | Early-stopped; extension in progress |

*\*Further GPU epochs beyond this snapshot are running as of writing.*

![Convergence curves](../figures/fig05_convergence_curves.png)

*Figure 5. Training/validation loss per epoch.*

GWN-predefined showed a real transient spike at epoch 4 (val MAE briefly 3.76) before recovering and continuing to a new best; GWN-adaptive's curve was smoother. Both converge in far fewer, cheaper epochs on GPU than the original CPU-only budget allowed.

**Comparison with Wu et al. (2019).** Published METR-LA numbers (MAE/RMSE/MAPE): 15 min 2.69/5.15/6.90%; 30 min 3.07/6.22/8.37%; 60 min 3.53/7.37/10.01%. Converting this report's GWN-adaptive MSE to RMSE gives 5.223/6.227/7.336 — within ~1% of published RMSE at 60 min. This is a marked improvement over the original CPU-constrained run and consistent with epoch budget, not architecture, having been the main limiting factor. Not-fully-comparable factors: no LR decay here, single seed, and tsl's re-implementation is not guaranteed identical to the original.

*Table 5. Per-station comparison, Sensors 1-3, both GWN configs.*

| Sensor | Horizon | MSE / MAE / MAPE, predefined | MSE / MAE / MAPE, adaptive |
|---|---|---|---|
| 1 | 15 min | 24.104 / 2.186 / 5.81% | 23.568 / 2.172 / 5.66% |
| 1 | 30 min | 36.279 / 2.528 / 7.05% | 35.434 / 2.540 / 6.95% |
| 1 | 60 min | 49.763 / 2.909 / 8.58% | 49.761 / 2.930 / 8.51% |
| 2 | 15 min | 7.670 / 1.566 / 2.69% | 7.914 / 1.551 / 2.67% |
| 2 | 30 min | 8.552 / 1.623 / 2.81% | 8.336 / 1.573 / 2.73% |
| 2 | 60 min | 9.061 / 1.700 / 2.93% | 8.820 / 1.636 / 2.84% |
| 3 | 15 min | 13.563 / 1.787 / 3.75% | 13.393 / 1.770 / 3.77% |
| 3 | 30 min | 17.493 / 1.935 / 4.26% | 16.923 / 1.889 / 4.05% |
| 3 | 60 min | 29.452 / 2.333 / 5.55% | 25.369 / 2.164 / 4.75% |

![Per-station MAE, Sensor 1](../figures/fig06_per_station_mae_sensor1.png)

*Figure 6a-c. Per-station MAE vs. horizon, all three models, Sensors 1-3.*

The adaptive graph's effect is **sensor-dependent, not uniform**: consistently positive and growing with horizon at Sensor 3 (up to 7.2% at 60 min) and modest at Sensor 2, but roughly a wash — even marginally negative at 30/60 min — at Sensor 1, once both configs get comparable training. This is a change from earlier, shorter training, where Sensor 1 had appeared to benefit most; that gain was likely partly an artefact of the predefined-only config being comparatively under-trained.

![Learned adjacency heatmap](../figures/fig07_learned_adjacency_heatmap.png)

*Figure 7. Learned adaptive adjacency, first 50 nodes.*

Node influence is defined as total outgoing weight (row-normalised graph, so only the column/source-side total is informative). Influence is concentrated but not cliquish (top node ≈1.9× the 15th-ranked node).

*Table 6. Top-15 most influential nodes (GWN-adaptive).*

| Rank | Node | Influence | Most-influenced nodes |
|---|---|---|---|
| 1 | 9 | 3.651 | 176, 148, 122 |
| 2 | 35 | 3.138 | 136, 93, 107 |
| 3 | 183 | 2.745 | 107, 166, 99 |
| 4 | 6 | 2.578 | 149, 119, 97 |
| 5 | 28 | 2.437 | 108, 201, 183 |
| 6-15 | 64, 200, 177, 40, 90, 185, 92, 29, 197, 84 | 2.42→1.96 | (see `results/gwn_adaptive/top15_influential_nodes.csv`) |

![Predefined vs learned adjacency](../figures/fig08_predefined_vs_learned.png)

*Figure 8. Predefined graph, learned graph, and their difference (first 50 nodes).*

**Learned vs. predefined (TTS's) graph:** partial, not total, overlap. Node 35 appears in both the predefined graph's top-10 best-connected nodes and the learned graph's top-2 most influential; the single most influential learned node (9) does not appear in the predefined top-10 at all. The model is not simply reproducing physical proximity, but is not ignoring it either — consistent with the sensor-dependent adaptive-graph benefit above.

## 5. Question 3: AGCRN

**Epoch selection.** AGCRN trained 11 epochs on CPU (manually stopped, still improving), then resumed on GPU with ceiling 25 / patience 8. Validation MAE reached a new best of 2.7075 at epoch 18 of 21 recorded, *still without a plateau* — `epoch_selection.json` states training "reached the new epoch ceiling... while still improving." (A separate interrupted-resume session left a small gap in the recorded history, undercounting true epochs slightly — a logging artefact, not a modelling one.) AGCRN shows the least sign of convergence of the four models and is the most likely to improve further.

*Table 7. Overall performance, all four models.*

| Model | Horizon | MSE (mph²) | MAE (mph) | MAPE (%) |
|---|---|---|---|---|
| TTS | 15/30/60 min | 32.792/49.780/76.577 | 2.946/3.525/4.383 | 7.78/9.97/13.18 |
| GWN, predefined | 15/30/60 min | 28.252/41.773/61.930 | 2.792/3.252/3.873 | 7.20/8.95/11.32 |
| GWN, adaptive | 15/30/60 min | 27.277/38.775/53.810 | 2.774/3.170/3.650 | 7.35/8.86/10.73 |
| AGCRN | 15/30/60 min | 30.337/42.475/57.561 | 2.850/3.238/3.673 | 7.63/9.14/10.76 |

*Table 8. Training progress, AGCRN vs. best GWN.*

| Model | Total epochs | Best val MAE | GPU epoch cost | Plateaued? |
|---|---|---|---|---|
| GWN, adaptive | 15 (4+11)* | 2.623 | ~64 s | Early-stopped; extension in progress |
| AGCRN | 21 recorded (11+~10)* | 2.707 | ~30 s | No — least converged of the four |

*\*Both still being extended as of writing.*

On current evidence, **GWN-adaptive leads AGCRN at every horizon** (0.6-2.7% MAE), narrowing rather than growing with horizon — the one clear exception to this report's usual "gap grows with horizon" pattern, plausibly because AGCRN has not plateaued while GWN-adaptive has (provisionally) early-stopped. AGCRN clearly beats GWN-predefined-only, and that gap *does* grow with horizon (trails by 2.0% at 15 min, leads by 5.2% at 60 min). On MAPE, AGCRN trails GWN-adaptive at every horizon, though narrowly at 60 min (10.76% vs. 10.73%). AGCRN remains cheapest per epoch (~30s vs. ~58-64s for GWN), a favourable cost profile even before accounting for its unfinished convergence.

*Table 9. Per-station comparison, Sensors 1-3, GWN-adaptive vs. AGCRN.*

| Sensor | Horizon | MSE / MAE / MAPE, GWN-adaptive | MSE / MAE / MAPE, AGCRN |
|---|---|---|---|
| 1 | 15 min | 23.568 / 2.172 / 5.66% | 31.252 / 2.382 / 6.36% |
| 1 | 30 min | 35.434 / 2.540 / 6.95% | 48.419 / 2.829 / 8.01% |
| 1 | 60 min | 49.761 / 2.930 / 8.51% | 62.888 / 3.173 / 9.56% |
| 2 | 15 min | 7.914 / 1.551 / 2.67% | 7.671 / 1.559 / 2.68% |
| 2 | 30 min | 8.336 / 1.573 / 2.73% | 8.090 / 1.579 / 2.73% |
| 2 | 60 min | 8.820 / 1.636 / 2.84% | 8.323 / 1.629 / 2.81% |
| 3 | 15 min | 13.393 / 1.770 / 3.77% | 13.663 / 1.870 / 3.95% |
| 3 | 30 min | 16.923 / 1.889 / 4.05% | 17.550 / 2.006 / 4.30% |
| 3 | 60 min | 25.369 / 2.164 / 4.75% | 25.457 / 2.225 / 4.79% |

GWN-adaptive leads clearly at Sensors 1 and 3 (up to 10.4% MAE at Sensor 1); Sensor 2 is essentially a dead heat, AGCRN's only (razor-thin) edge across the three sensors. This reverses the CPU-era per-station result, where AGCRN led at Sensors 2 and 3.

**Summary:** on this snapshot, GWN-adaptive is the stronger model overall and per-station, but AGCRN is the least-converged of the four and the comparison is provisional (Section 6).

**Comparison with Gaibie et al. (2024)**, who compared AGCRN, CLCRN, and GWN on South African weather forecasting (45 stations, hourly, 3-24h horizons): they found AGCRN clearly and consistently ahead of GWN, which sometimes underperformed even a purely temporal baseline. This report's traffic-domain result does **not** reproduce that direction — GWN-adaptive is narrowly ahead here. A plausible reconciliation: traffic congestion propagates directly and mechanically along a road network, giving GWN's graph-based mechanisms (fixed *and* learned) more genuine signal to exploit than a coarser, less direct weather-station graph does; in both domains, though, it is specifically the *learned/adaptive* component of each architecture — not the fixed graph — that appears to carry most of the useful spatial signal once given enough training.

## 6. Overall Discussion

The central finding — GWN-adaptive narrowly ahead of AGCRN at every horizon — is a reversal of the CPU-era result and should be read as **provisional**: AGCRN had not plateaued at its most recent checkpoint while both GWN configs had (provisionally) early-stopped, and GWN's proximity to Wu et al.'s published numbers suggests it is nearer its ceiling than AGCRN is to its own. The margin is also small (0.6-2.7%) relative to other gaps in this report (e.g. 18.6% TTS-to-GWN-adaptive at 60 min).

More robust findings, since they compare configurations trained under identical schedules: the adaptive adjacency helps GWN at every horizon, growing with horizon, for a modest 12% per-epoch cost — but its benefit is clearly sensor-dependent (strong at Sensor 3, negligible-to-negative at Sensor 1), consistent with the partial (not total) overlap found between the predefined graph's best-connected nodes and the learned graph's most influential ones. The three architectures also represent distinct spatial-modelling philosophies — fixed-graph diffusion (TTS, both GWN configs), fixed-plus-learned (GWN-adaptive), and fully learned (AGCRN) — and distinct temporal mechanisms (dilated convolution vs. gated recurrence); that AGCRN and GWN-adaptive are this close suggests the predefined road-distance graph is not the dominant factor separating them once both are properly trained. The horizon-scaling pattern ("gap grows with horizon") holds throughout except for the AGCRN-vs-GWN-adaptive margin itself, plausibly because AGCRN's incomplete convergence disproportionately affects its longer-horizon accuracy. No model wins everywhere: Sensor 2 is a near-tie across all four models; Sensor 1 and 3 now favour GWN-adaptive, reversing AGCRN's earlier edge there.

## 7. Conclusion

This report implemented and compared TTS, both GWN configurations, and AGCRN on the full METR-LA dataset under one consistent pipeline, addressing every required element: adjacency visualisation, overall and per-station performance across all three metrics at all three horizons, training-time/convergence comparison, the Wu et al. and Gaibie et al. paper comparisons, and a fully justified learned-adjacency/influence analysis. Training was resumed from CPU checkpoints onto GPU hardware mid-project rather than restarted — a real demonstration of the checkpointing/resuming workflow this assignment targets — and is still being extended at time of writing. The current standings (GWN-adaptive narrowly ahead of AGCRN, a reversal of the CPU-era result) are the best available evidence, not a final answer; AGCRN in particular is most likely to move with further training. The more robust findings — that GWN's adaptive adjacency helps, in a sensor-dependent way; that no single model wins everywhere; and that this traffic-domain AGCRN-vs-GWN comparison does not straightforwardly reproduce Gaibie et al.'s weather-domain result — are drawn from multiple independent angles of analysis and are unlikely to be pure artefacts of the remaining training gap.

## References

1. Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C., 2019. Graph WaveNet for Deep Spatial-Temporal Graph Modeling. *arXiv:1906.00121*.
2. Bai, L., Yao, L., Li, C., Wang, X. and Wang, C., 2020. Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting. *NeurIPS 33*, pp. 17804-17815.
3. Cini, A., Marisca, I., Zambon, D. and Alippi, C., 2023. Graph Deep Learning for Time Series Forecasting. *arXiv:2310.15978*.
4. Gaibie, A., Amir, H., Nandutu, I. and Moodley, D., 2024. Predicting and Discovering Weather Patterns in South Africa Using Spatial-Temporal Graph Neural Networks. *Southern African Conference for Artificial Intelligence Research*, pp. 144-160.
5. Torch Spatiotemporal documentation. https://torch-spatiotemporal.readthedocs.io/en/latest/
