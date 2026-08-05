# Lightning Prediction — Research Log
**Region:** Eastern Mediterranean  
**Goal:** Predict lightning occurrence from ERA5 atmospheric reanalysis data using tree-based ML models. Compare against a future U-Net CNN to evaluate whether spatial context adds predictive value.

---

## Pipeline Overview

The full pipeline runs in 4 stages:

1. **`download_era5_arco.py`** — Downloads ERA5 single-level and pressure-level variables from the ARCO-ERA5 public Zarr store on Google Cloud Storage (anonymous access via `gcsfs`). Saves as NetCDF4. Output: `era5_single_level_{ts}.nc`, `era5_pressure_level_{ts}.nc`

2. **`ildn_to_era5_grid.py`** / **`entln_to_era5_grid.py`** — Grids ILDN/ENTLN lightning strike observations onto the ERA5 grid (counts per cell per hour). Output: `ildn_on_era5_grid_{ts}.nc`

3. **`build_tabular_dataset.py`** — Merges ERA5 features and lightning grid into a flat tabular parquet file. Each row = one ERA5 grid cell at one hour. Output: `tabular_dataset_{ts}.parquet`

4. **`balance_dataset.py`** — Undersamples the raw dataset to 50/50 lightning/no-lightning for training. Keeps all lightning rows, samples equal number of no-lightning rows. Output: `tabular_dataset_{ts}_balanced.parquet`

5. **`unite_parquets.py`** — Concatenates balanced parquets from multiple years into one training file. Output: `tabular_dataset_{years}_balanced.parquet`

6. **`train_lightgbm_xgboost.py`** — Trains LightGBM and XGBoost on the balanced training set, evaluates on the unbalanced test set.

---

## Data

| Split | Years | Dataset | Balanced? |
|-------|-------|---------|-----------|
| Train | 2023 + 2024 | `tabular_dataset_2023_2024_balanced.parquet` | Yes (50/50) |
| Test  | 2025 | `tabular_dataset_2025.parquet` | No (realistic imbalance) |

**Class imbalance in raw data:** ~0.08% of cells have lightning at any given hour.  
**Why test is unbalanced:** Reflects real-world conditions. A balanced test set would give an unrealistically optimistic view of precision.

---

## Model Configuration

### LightGBM
```
objective:         binary
n_estimators:      500 (with early stopping, patience=50)
learning_rate:     0.05
num_leaves:        64
min_child_samples: 20
class_weight:      balanced
random_state:      42
feature importance: gain (not default split count)
```

### XGBoost
```
objective:          binary:logistic
n_estimators:       500 (with early stopping, patience=50)
learning_rate:      0.05
max_depth:          6
min_child_weight:   20
scale_pos_weight:   neg/pos ratio (~1.0 on balanced train set)
random_state:       42
eval_metric:        logloss
feature importance: gain (XGBoost default)
```

### Threshold Optimization
Both models use a threshold optimization step after training:
- **Goal:** Maximize precision while keeping recall ≥ 0.30 (MIN_RECALL)
- **Rationale:** False alarms (low precision) are costly. Missing some lightning (lower recall) is more acceptable.
- Default threshold = 0.50 is also reported for comparison.

---

## Experiment 1 — Train: 2023, Test: 2025

*Date: July 2026*

### XGBoost Results

**Default threshold (0.50):**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| No Lightning | 1.00 | 0.95 | 0.98 | 14,897,479 |
| Lightning | 0.01 | 0.59 | 0.02 | 12,041 |
| **Accuracy** | | | **0.95** | 14,909,520 |

ROC-AUC: **0.9056**

**Optimized threshold (0.8709):**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| No Lightning | 1.00 | 0.99 | 0.99 | 14,897,479 |
| Lightning | 0.02 | 0.30 | 0.03 | 12,041 |
| **Accuracy** | | | **0.99** | 14,909,520 |

Precision: 0.0179 | Recall: 0.3001 at threshold 0.8709

### Top 20 Feature Importances (Gain)

**LightGBM:**
| Feature | Gain |
|---------|------|
| total_totals_index | 741,821 |
| total_column_cloud_ice_water | 236,463 |
| total_column_cloud_liquid_water | 53,334 |
| convective_available_potential_energy | 51,840 |
| surface_pressure | 15,803 |
| k_index | 15,459 |
| temperature_225hPa | 14,386 |
| temperature_250hPa | 13,943 |
| specific_humidity_70hPa | 12,599 |
| specific_humidity_30hPa | 12,595 |
| specific_humidity_50hPa | 12,043 |
| specific_humidity_20hPa | 11,893 |
| specific_humidity_5hPa | 11,525 |
| specific_humidity_2hPa | 11,519 |
| specific_humidity_7hPa | 11,264 |
| specific_humidity_3hPa | 10,283 |
| specific_humidity_10hPa | 10,127 |
| temperature_200hPa | 9,162 |
| specific_humidity_1hPa | 8,477 |
| specific_humidity_1000hPa | 8,265 |

**XGBoost:**
| Feature | Gain |
|---------|------|
| total_totals_index | 0.3947 |
| total_column_cloud_ice_water | 0.1164 |
| total_column_cloud_liquid_water | 0.0336 |
| convective_available_potential_energy | 0.0274 |
| k_index | 0.0116 |
| temperature_225hPa | 0.0090 |
| vertical_velocity_850hPa | 0.0089 |
| temperature_250hPa | 0.0074 |
| temperature_600hPa | 0.0070 |
| vertical_velocity_825hPa | 0.0065 |
| vertical_velocity_800hPa | 0.0063 |
| specific_humidity_7hPa | 0.0057 |
| specific_humidity_650hPa | 0.0055 |
| temperature_650hPa | 0.0055 |
| specific_humidity_225hPa | 0.0054 |
| temperature_200hPa | 0.0054 |
| temperature_350hPa | 0.0053 |
| specific_humidity_250hPa | 0.0052 |
| specific_humidity_700hPa | 0.0050 |
| specific_humidity_70hPa | 0.0049 |

---

## Key Observations & Interpretation

### On the metrics
- **ROC-AUC of 0.91** is genuine signal — not an artifact of class imbalance. It means the model ranks lightning cells above no-lightning cells with high accuracy.
- **Precision of ~0.02 at the optimized threshold** means ~2% of lightning predictions are correct. This sounds low, but given only 0.08% of cells have lightning, it represents a 25x improvement over random guessing.
- **Recall of 0.30** means the model catches 30% of actual lightning events. This was set as the minimum acceptable floor.
- The PR curve is more informative than ROC for this problem because of extreme class imbalance.

### On feature importance
- Both models agree on the top 4 predictors: **Total Totals Index, cloud ice water, cloud liquid water, CAPE** — all directly related to convective instability, which is the physical driver of lightning.
- XGBoost additionally highlights **vertical velocity at 850/825/800hPa** (~1.5–2 km altitude, lower troposphere). Upward vertical velocity at these levels is the physical trigger for convective storms — lifted moist air is what creates the thunderstorm cells that produce lightning. This is one of the most physically meaningful features the model could have identified.
- LightGBM ranks **stratospheric specific humidity (1–10hPa, ~48–30 km altitude)** highly — these levels are far above the troposphere and physically irrelevant to lightning. Likely spurious correlations.
- **XGBoost aligns significantly better with known atmospheric physics than LightGBM.** This is not just a performance observation — it suggests that XGBoost's level-wise tree growth captures the physical relationships between atmospheric variables more faithfully than LightGBM's leaf-wise approach for this problem.
- Future work: consider removing stratospheric features (above 100hPa) to reduce noise in LightGBM.

### On the training strategy
- Train set balanced 50/50 to overcome extreme class imbalance (~0.08% lightning in raw data).
- Test set kept unbalanced to reflect real-world conditions.
- Time-based split: train on past years, test on future year — standard practice for weather/climate ML to prevent data leakage.

---

## Proxy LPI — Plan & Design Decisions

The Lightning Potential Index (LPI) was developed by Yair et al. (2010) and adapted as a proxy version by the thesis mentor. It estimates lightning potential by integrating updraft velocity and cloud microphysics through the charging zone (0°C to −20°C).

### Formula
```
LPI = (1/ΔZ) ∫ w · ε dz    over the charging zone
```
where ε is a microphysical charging efficiency term based on the coexistence of ice and supercooled water.

### Variable Mapping (ERA5 → mentor's code)
| Mentor variable | ERA5 variable | Source | Status |
|----------------|---------------|--------|--------|
| `t` | `temperature` | Pressure levels | ✅ Already downloaded |
| `z` | `geopotential` | Pressure levels | ❌ Need to add |
| `w_pa` | `vertical_velocity` | Pressure levels | ✅ Already downloaded |
| `q_i` | `specific_cloud_ice_water_content` (ciwc) | Pressure levels | ❌ Need to add |
| `q_s` | `specific_cloud_liquid_water_content` (clwc) | Pressure levels | ❌ Need to add |
| `cape` | `convective_available_potential_energy` | Single level | ✅ Already downloaded |

### Key Design Decision: clwc instead of snow
The mentor's original code used specific snow water content (`cswc`) for `q_s`. We substitute this with **specific cloud liquid water content (`clwc`)** for the following physical reason:

The non-inductive charging mechanism — the primary cause of charge separation in thunderstorms — requires the coexistence of **ice crystals and supercooled liquid water droplets** in the charging zone. `clwc` represents exactly these supercooled liquid droplets. Snow (`cswc`) is aggregated precipitating ice that has already fallen out of the cloud and is less relevant to in-cloud electrification. Both `ciwc` and `clwc` are available in the ARCO-ERA5 dataset at pressure levels.

### Integration into Pipeline
LPI will be computed as a new script (`calculate_lpi.py`) after ERA5 download and before tabular dataset construction. The output is a 2D (time × lat × lon) array of proxy LPI values that gets merged into the tabular dataset as a new feature column.

---

---

## Experiment 2 — Train: 2023 (with LPI + new variables), Test: 2025

*Date: July 2026*

### Changes vs Experiment 1
- Added 3 new pressure-level variables: `geopotential`, `specific_cloud_ice_water_content`, `specific_cloud_liquid_water_content`
- Added `proxy_lpi` as a feature column

### Results
Metrics virtually identical to Experiment 1 (ROC-AUC: LightGBM 0.9095, XGBoost 0.9072). The proxy LPI did not appear in the top 50 most important features for either model.

### Key Finding: Proxy LPI adds no predictive value
The proxy LPI is computed from vertical velocity, cloud ice, cloud liquid water, and CAPE — all of which are already present as raw features. The model can already capture their combined effect directly, so the derived LPI index is redundant. This is a valid scientific finding: **the raw ERA5 variables carry the full predictive signal; the LPI does not add information beyond what is already available to the model.**

---

## Current Modeling Assumption: Synchronous Prediction (T → T)

The current setup predicts lightning occurrence at time T using atmospheric variables **at the same time T**. This is a diagnostic relationship — the model learns the atmospheric state associated with lightning, not a forecast.

### Future Direction: Lagged Prediction (T-k → T)
A natural next step is to shift the atmospheric features back by k hours (T-1, T-2, etc.) and predict lightning at time T. This would turn the model into a genuine **short-range forecast** — predicting whether lightning will occur in the next 1-3 hours based on current atmospheric state.

This is more operationally useful but harder, since the atmospheric signal at T-1 is weaker than at T. Testing both approaches and comparing skill scores would be a strong thesis contribution.

---

---

## Experiment 3 — Train: 2004–2024 (all available years), Test: 2025

*Date: August 2026*

### Training Data
All available lightning data combined into one training set:

| Source | Years | Coverage | Strikes |
|--------|-------|----------|---------|
| LPATS (IEC, mentor data) | 2004 | Sep–Dec | ~51K |
| LPATS (IEC, mentor data) | 2005 | Jan–Nov (missing May) | ~41K |
| LPATS (IEC, mentor data) | 2006 | Jan–Aug | ~38K |
| LPATS (IEC, mentor data) | 2008 | Sep–Dec | ~98K |
| LPATS (IEC, mentor data) | 2009 | Jan–Sep | ~56K |
| ILDN (modern) | 2023 | Full year | — |
| ILDN (modern) | 2024 | Full year | — |

284,778 unique LPATS strikes after deduplication (overlapping event files removed).  
ERA5 downloaded for each year covering the same date ranges as the lightning data.

### Test Data
- 2025 (full year, unbalanced, ILDN modern format) — same as Experiments 1 & 2.

### Results

#### LightGBM
**Default threshold (0.50):**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| No Lightning | 1.00 | 0.96 | 0.98 | 14,897,479 |
| Lightning | 0.01 | 0.61 | 0.02 | 12,041 |
| **Accuracy** | | | **0.96** | 14,909,520 |

ROC-AUC: **0.9231**

**Optimized threshold (0.8944):** Precision: 0.0245 | Recall: 0.3001

**Top 20 features (gain):**
| Feature | Gain |
|---------|------|
| specific_cloud_ice_water_content_600hPa | 983,817 |
| specific_cloud_ice_water_content_550hPa | 397,456 |
| total_totals_index | 191,047 |
| convective_available_potential_energy | 89,549 |
| specific_cloud_ice_water_content_650hPa | 85,139 |
| specific_cloud_ice_water_content_500hPa | 65,091 |
| total_column_cloud_ice_water | 46,780 |
| surface_pressure | 33,937 |
| total_column_cloud_liquid_water | 32,815 |
| specific_cloud_ice_water_content_700hPa | 16,466 |
| specific_cloud_liquid_water_content_700hPa | 15,966 |
| specific_humidity_20hPa | 16,251 |
| temperature_225hPa | 14,488 |
| k_index | 13,968 |
| specific_humidity_5hPa | 13,669 |
| temperature_250hPa | 13,289 |
| specific_humidity_50hPa | 12,215 |
| specific_humidity_2hPa | 11,072 |
| specific_humidity_3hPa | 10,689 |
| specific_humidity_7hPa | 10,680 |

#### XGBoost
**Default threshold (0.50):**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| No Lightning | 1.00 | 0.95 | 0.97 | 14,897,479 |
| Lightning | 0.01 | 0.64 | 0.02 | 12,041 |
| **Accuracy** | | | **0.95** | 14,909,520 |

ROC-AUC: **0.9169**

**Optimized threshold (0.9055):** Precision: 0.0224 | Recall: 0.3001

**Top 20 features (gain):**
| Feature | Gain |
|---------|------|
| specific_cloud_ice_water_content_600hPa | 0.3790 |
| specific_cloud_ice_water_content_550hPa | 0.2476 |
| specific_cloud_ice_water_content_650hPa | 0.0381 |
| specific_cloud_ice_water_content_500hPa | 0.0240 |
| total_totals_index | 0.0223 |
| convective_available_potential_energy | 0.0113 |
| specific_cloud_liquid_water_content_700hPa | 0.0099 |
| specific_cloud_ice_water_content_700hPa | 0.0096 |
| total_column_cloud_ice_water | 0.0091 |
| total_column_cloud_liquid_water | 0.0064 |
| specific_cloud_liquid_water_content_750hPa | 0.0050 |
| specific_cloud_liquid_water_content_850hPa | 0.0050 |
| proxy_lpi | 0.0043 |
| specific_cloud_liquid_water_content_825hPa | 0.0040 |
| specific_cloud_ice_water_content_400hPa | 0.0036 |
| specific_cloud_ice_water_content_450hPa | 0.0031 |
| vertical_velocity_800hPa | 0.0027 |
| specific_cloud_liquid_water_content_800hPa | 0.0027 |
| specific_cloud_liquid_water_content_775hPa | 0.0026 |
| k_index | 0.0026 |

### Key Findings

- **ROC-AUC improved significantly** vs Experiment 2: LightGBM 0.9095 → 0.9231, XGBoost 0.9072 → 0.9169. Adding more training years had a clear positive effect.
- **LightGBM now outperforms XGBoost** (0.9231 vs 0.9169) — a reversal from Experiments 1 & 2.
- **Both models agree on the dominant feature**: `specific_cloud_ice_water_content` at 500–700 hPa (the mixed-phase charging zone). This is a physically very meaningful finding — this altitude range is exactly where non-inductive charge separation occurs in thunderstorms.
- **proxy_lpi appears in XGBoost top 20** (rank 13, gain 0.004) with more training data, suggesting marginal but non-zero signal. Still not present in LightGBM top 20.
- Stratospheric humidity features (2–50 hPa) still appear in LightGBM top 20 — likely spurious correlations. Not present in XGBoost top 20, which continues to show cleaner physical feature selection.

---

---

## Experiment 4 — Lag Experiments (T-k → T), k = 1..6

*Date: August 2026*

### Setup
For each lag k, ERA5 features at time T-k are used to predict lightning occurrence at time T. This turns the diagnostic model (Experiment 3) into a genuine short-range forecast. Training data is identical to Experiment 3 (2004–2024). Test data is 2025 (unbalanced).

### ROC-AUC Results

| Lag | ERA5 time | Lightning time | LightGBM AUC | XGBoost AUC |
|-----|-----------|----------------|-------------|-------------|
| 0 (sync) | T | T | 0.9231 | 0.9169 |
| 1 h | T-1 | T | 0.9214 | 0.9166 |
| 2 h | T-2 | T | 0.9099 | 0.9124 |
| 3 h | T-3 | T | 0.9115 | 0.9095 |
| 4 h | T-4 | T | 0.9010 | 0.9045 |
| 5 h | T-5 | T | 0.8979 | 0.8946 |
| 6 h | T-6 | T | 0.8897 | 0.8937 |

### Key Findings

- **The T → T-1 drop is very small** (LightGBM: 0.9231 → 0.9214, XGBoost: 0.9169 → 0.9166). The atmospheric state 1 hour before is nearly as predictive as the concurrent state. Operationally significant — a 1-hour forecast retains almost all the skill of a diagnosis.
- **Gradual decline through T-6**: total drop over 6 hours is ~0.033 for LightGBM and ~0.023 for XGBoost. At 6 hours lead time the models still achieve AUC ~0.89 — well above random.
- **XGBoost is more robust to longer lags**: the models converge at longer lead times, with XGBoost matching or beating LightGBM at lags 2–6.
- **Feature importance shifts at higher lags**: `total_totals_index` and `total_column_cloud_ice_water` gain relative importance at lags 4–6, while pressure-level `specific_cloud_ice_water_content` at 500–700 hPa diminishes. Physically meaningful — large-scale thermodynamic indices are more persistent in time than instantaneous cloud microphysics.
- **proxy_lpi** appears in XGBoost top 20 at lags 5 and 6 (rank ~11), suggesting it captures a lagged synoptic signal.

---

## Experiment 5 — Convective Cloud Mask (ciwc × w < 0 at 500–700 hPa)

*Date: August 2026*

### Motivation
Lightning forms exclusively in cumulonimbus clouds, not in cirrus or clear-sky grid cells. Previous experiments trained on all atmospheric states, including rows with no convective activity. This experiment filters the dataset to keep only rows where convective ice is present: `vertical_velocity × specific_cloud_ice_water_content < 0` at any pressure level between 500–700 hPa.

**ERA5 sign convention:** vertical_velocity < 0 = updraft. So `w * ciwc < 0` means updraft + ice = cumulonimbus cell.

### Mask Details
- Levels checked: 500, 525, 550, 575, 600, 625, 650, 675, 700 hPa
- A row is kept if the condition holds at **any** of these levels
- Applied to both train and test sets
- Output files have `_convmask` suffix

### Dataset Size After Masking
| Split | Before mask | After mask | Lightning rows |
|-------|-------------|------------|----------------|
| Test (2025) | 14,907,818 | 8,385,916 | 9,360 (was 12,041) |

Note: **2,681 real lightning events (~22%) were masked out** — these occurred in grid cells where ERA5 did not show convective ice at 500–700 hPa. Possible explanations: ERA5 resolution limitations, or lightning in non-classical convective regimes (e.g. warm rain lightning, shallow convection).

### Results

| Metric | Exp 3 (no mask) | Exp 5 (convmask) | Change |
|--------|----------------|-----------------|--------|
| LightGBM AUC | 0.9231 | **0.9239** | +0.0008 |
| XGBoost AUC | 0.9169 | **0.9174** | +0.0005 |
| LightGBM precision (opt.) | 0.0245 | **0.0288** | +18% |
| XGBoost precision (opt.) | 0.0224 | **0.0275** | +23% |
| Recall (both, opt.) | 0.30 | 0.30 | unchanged |

#### LightGBM Top 20 Features (convmask)
| Rank | Feature | Gain |
|------|---------|------|
| 1 | specific_cloud_ice_water_content_600hPa | 939,970 |
| 2 | specific_cloud_ice_water_content_550hPa | 252,312 |
| 3 | total_totals_index | 177,334 |
| 4 | convective_available_potential_energy | 93,835 |
| 5 | total_column_cloud_ice_water | 39,147 |
| 6 | specific_cloud_ice_water_content_650hPa | 30,801 |
| 7 | specific_cloud_ice_water_content_500hPa | 29,547 |
| 8 | surface_pressure | 28,371 |
| 9 | total_column_cloud_liquid_water | 27,643 |
| 10 | temperature_250hPa | 16,842 |
| 11 | temperature_225hPa | 15,978 |
| 12 | specific_cloud_liquid_water_content_700hPa | 14,805 |
| 13 | specific_humidity_20hPa | 13,873 |
| 14 | k_index | 13,055 |
| 15–20 | specific_humidity_30/50/10/5/3/2 hPa | ~10,000–11,000 |

#### XGBoost Top 20 Features (convmask)
| Rank | Feature | Gain |
|------|---------|------|
| 1 | specific_cloud_ice_water_content_600hPa | 0.5314 |
| 2 | specific_cloud_ice_water_content_550hPa | 0.1500 |
| 3 | specific_cloud_ice_water_content_650hPa | 0.0388 |
| 4 | total_totals_index | 0.0178 |
| 5 | specific_cloud_ice_water_content_500hPa | 0.0121 |
| 6 | convective_available_potential_energy | 0.0100 |
| 7 | total_column_cloud_ice_water | 0.0068 |
| 8 | specific_cloud_ice_water_content_700hPa | 0.0061 |
| 9 | specific_cloud_liquid_water_content_700hPa | 0.0055 |
| 10 | specific_cloud_liquid_water_content_850hPa | 0.0050 |
| 11 | specific_cloud_liquid_water_content_500hPa | 0.0050 |
| 12 | specific_cloud_liquid_water_content_750hPa | 0.0045 |
| 13 | total_column_cloud_liquid_water | 0.0043 |
| 14 | specific_cloud_liquid_water_content_825hPa | 0.0033 |
| 15 | specific_cloud_ice_water_content_450hPa | 0.0030 |
| 16 | temperature_250hPa | 0.0026 |
| 17 | **proxy_lpi** | **0.0026** |
| 18 | specific_cloud_ice_water_content_400hPa | 0.0025 |
| 19 | specific_cloud_liquid_water_content_775hPa | 0.0023 |
| 20 | specific_cloud_liquid_water_content_800hPa | 0.0022 |

### Key Findings

- **AUC marginally improved** in both models — the mask doesn't dramatically change discriminative ability because the model was already good at separating convective from non-convective states.
- **Precision improved significantly** — LightGBM +18%, XGBoost +23%. By removing easy negatives (clear sky, cirrus), the remaining negatives are all convective cells without lightning — a harder and more meaningful classification problem.
- **Most important finding: XGBoost top 20 is now entirely physically meaningful.** Stratospheric humidity (2–50 hPa) that appeared in previous experiments has completely disappeared. The convective mask removed enough non-convective rows that spurious stratospheric correlations vanished. Every feature in the top 20 is directly relevant to convective lightning physics.
- **LightGBM still shows stratospheric humidity** (20–50 hPa) in positions 13–20. This persists despite the mask, suggesting LightGBM's leaf-wise growth still finds spurious patterns that XGBoost's level-wise approach avoids.
- **proxy_lpi ranks 17 in XGBoost** — consistent across experiments with more training data and now with convective filtering. Small but stable signal.
- **22% of lightning events don't meet the convective mask criteria** — scientifically interesting. These may represent ERA5 resolution limitations or non-classical lightning regimes.

---

## Experiment 6 — Training Metric Experiments (aucpr early stopping)

*Date: August 2026*

### Motivation
The PR curve from Experiment 5 showed very low precision (~0.022–0.028) across all operating points, even with AUC ~0.92. The question was whether this could be improved by changing the training objective/metric rather than the data or architecture.

### Experiments Attempted

**6a — aucpr early stopping (balanced 50/50 training):**
Changed XGBoost `eval_metric` from `'logloss'` to `'aucpr'`, and added a custom PR-AUC eval function to LightGBM (`average_precision_score`). Training data remained the balanced 50/50 convmask parquet.

Results (XGBoost):
- ROC-AUC: 0.9214 (vs 0.9174 in Exp 5) — slight improvement
- Precision at recall=0.30: 0.0267 (vs 0.022 in Exp 5) — slight improvement
- PR curve shape: unchanged — flat at ~0.027 across all recall values

**6b — Unbalanced training (scale_pos_weight=99):**
Attempted to train on raw unbalanced parquets (all non-lightning rows included) with `scale_pos_weight` set to the true class ratio (~99), and `eval_metric='aucpr'`. Process killed (OOM) on EC2 after loading only 4 years of data — unbalanced data is ~20M rows for 4 years, too large for available RAM.

**6c — Intermediate ratio (10:1 sampling):**
Discussed but not attempted. Theoretically equivalent to 50:50 + scale_pos_weight=99 in terms of gradient signal — not worth implementing.

### Key Finding

**The precision ceiling is fundamental to ERA5 resolution, not a training configuration issue.** No combination of class weights, sampling ratios, or training metrics can raise the PR curve when the bottleneck is that ERA5 at 0.25° cannot resolve individual storm cells. The model assigns moderately high probabilities (~0.5–0.9) to many convective-looking cells, but needs a threshold of ~0.90 to achieve even 2.7% precision — showing that most convective environments don't produce lightning in any specific cell at any specific hour.

### Conclusions on Precision/Recall

- **ROC-AUC is a genuine and meaningful metric** — it measures ranking ability, not threshold-dependent precision. AUC 0.92 means the model correctly ranks lightning cells above non-lightning cells 92% of the time.
- **PR-AUC and precision are fundamentally limited** by the ~1% base rate and 28 km grid resolution. Many non-lightning cells look atmospherically identical to lightning cells at ERA5 scale, because the sub-grid storm initiation processes are invisible to the model.
- **The model is best framed as a regional lightning risk index**, not a cell-level predictor. This is consistent with how other ERA5-scale lightning papers interpret their results (e.g., Ehrensperger et al. 2025, MCC=0.278 on Eastern Alps).
- **Improvement requires higher-resolution features** (storm-resolving NWP, radar) and spatial context (U-Net) — both of which are the stated next step of this thesis.

---

## Next Steps
- [x] Complete Experiment 3 training and evaluate results
- [x] Test lagged prediction (T-1 through T-6)
- [x] Apply convective cloud mask and compare vs baseline
- [ ] Investigate the 22% of lightning events masked out — are they real convective events ERA5 missed, or a different lightning regime?
- [ ] Consider dropping stratospheric features (>100 hPa) from LightGBM to test if it removes spurious correlations
- [ ] Meet with Vlad to discuss results and U-Net next steps
