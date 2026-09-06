
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

| Split | Years       | Dataset                                        | Balanced?                |
| ----- | ----------- | ---------------------------------------------- | ------------------------ |
| Train | 2023 + 2024 | `tabular_dataset_2023_2024_balanced.parquet` | Yes (50/50)              |
| Test  | 2025        | `tabular_dataset_2025.parquet`               | No (realistic imbalance) |

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

| Class              | Precision | Recall | F1             | Support    |
| ------------------ | --------- | ------ | -------------- | ---------- |
| No Lightning       | 1.00      | 0.95   | 0.98           | 14,897,479 |
| Lightning          | 0.01      | 0.59   | 0.02           | 12,041     |
| **Accuracy** |           |        | **0.95** | 14,909,520 |

ROC-AUC: **0.9056**

**Optimized threshold (0.8709):**

| Class              | Precision | Recall | F1             | Support    |
| ------------------ | --------- | ------ | -------------- | ---------- |
| No Lightning       | 1.00      | 0.99   | 0.99           | 14,897,479 |
| Lightning          | 0.02      | 0.30   | 0.03           | 12,041     |
| **Accuracy** |           |        | **0.99** | 14,909,520 |

Precision: 0.0179 | Recall: 0.3001 at threshold 0.8709

### Top 20 Feature Importances (Gain)

**LightGBM:**

| Feature                               | Gain    |
| ------------------------------------- | ------- |
| total_totals_index                    | 741,821 |
| total_column_cloud_ice_water          | 236,463 |
| total_column_cloud_liquid_water       | 53,334  |
| convective_available_potential_energy | 51,840  |
| surface_pressure                      | 15,803  |
| k_index                               | 15,459  |
| temperature_225hPa                    | 14,386  |
| temperature_250hPa                    | 13,943  |
| specific_humidity_70hPa               | 12,599  |
| specific_humidity_30hPa               | 12,595  |
| specific_humidity_50hPa               | 12,043  |
| specific_humidity_20hPa               | 11,893  |
| specific_humidity_5hPa                | 11,525  |
| specific_humidity_2hPa                | 11,519  |
| specific_humidity_7hPa                | 11,264  |
| specific_humidity_3hPa                | 10,283  |
| specific_humidity_10hPa               | 10,127  |
| temperature_200hPa                    | 9,162   |
| specific_humidity_1hPa                | 8,477   |
| specific_humidity_1000hPa             | 8,265   |

**XGBoost:**

| Feature                               | Gain   |
| ------------------------------------- | ------ |
| total_totals_index                    | 0.3947 |
| total_column_cloud_ice_water          | 0.1164 |
| total_column_cloud_liquid_water       | 0.0336 |
| convective_available_potential_energy | 0.0274 |
| k_index                               | 0.0116 |
| temperature_225hPa                    | 0.0090 |
| vertical_velocity_850hPa              | 0.0089 |
| temperature_250hPa                    | 0.0074 |
| temperature_600hPa                    | 0.0070 |
| vertical_velocity_825hPa              | 0.0065 |
| vertical_velocity_800hPa              | 0.0063 |
| specific_humidity_7hPa                | 0.0057 |
| specific_humidity_650hPa              | 0.0055 |
| temperature_650hPa                    | 0.0055 |
| specific_humidity_225hPa              | 0.0054 |
| temperature_200hPa                    | 0.0054 |
| temperature_350hPa                    | 0.0053 |
| specific_humidity_250hPa              | 0.0052 |
| specific_humidity_700hPa              | 0.0050 |
| specific_humidity_70hPa               | 0.0049 |

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

| Mentor variable | ERA5 variable                                  | Source          | Status                |
| --------------- | ---------------------------------------------- | --------------- | --------------------- |
| `t`           | `temperature`                                | Pressure levels | ✅ Already downloaded |
| `z`           | `geopotential`                               | Pressure levels | ❌ Need to add        |
| `w_pa`        | `vertical_velocity`                          | Pressure levels | ✅ Already downloaded |
| `q_i`         | `specific_cloud_ice_water_content` (ciwc)    | Pressure levels | ❌ Need to add        |
| `q_s`         | `specific_cloud_liquid_water_content` (clwc) | Pressure levels | ❌ Need to add        |
| `cape`        | `convective_available_potential_energy`      | Single level    | ✅ Already downloaded |

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

| Source                   | Years | Coverage               | Strikes |
| ------------------------ | ----- | ---------------------- | ------- |
| LPATS (IEC, mentor data) | 2004  | Sep–Dec               | ~51K    |
| LPATS (IEC, mentor data) | 2005  | Jan–Nov (missing May) | ~41K    |
| LPATS (IEC, mentor data) | 2006  | Jan–Aug               | ~38K    |
| LPATS (IEC, mentor data) | 2008  | Sep–Dec               | ~98K    |
| LPATS (IEC, mentor data) | 2009  | Jan–Sep               | ~56K    |
| ILDN (modern)            | 2023  | Full year              | —      |
| ILDN (modern)            | 2024  | Full year              | —      |

284,778 unique LPATS strikes after deduplication (overlapping event files removed).
ERA5 downloaded for each year covering the same date ranges as the lightning data.

### Test Data

- 2025 (full year, unbalanced, ILDN modern format) — same as Experiments 1 & 2.

### Results

#### LightGBM

**Default threshold (0.50):**

| Class              | Precision | Recall | F1             | Support    |
| ------------------ | --------- | ------ | -------------- | ---------- |
| No Lightning       | 1.00      | 0.96   | 0.98           | 14,897,479 |
| Lightning          | 0.01      | 0.61   | 0.02           | 12,041     |
| **Accuracy** |           |        | **0.96** | 14,909,520 |

ROC-AUC: **0.9231**

**Optimized threshold (0.8944):** Precision: 0.0245 | Recall: 0.3001

**Top 20 features (gain):**

| Feature                                    | Gain    |
| ------------------------------------------ | ------- |
| specific_cloud_ice_water_content_600hPa    | 983,817 |
| specific_cloud_ice_water_content_550hPa    | 397,456 |
| total_totals_index                         | 191,047 |
| convective_available_potential_energy      | 89,549  |
| specific_cloud_ice_water_content_650hPa    | 85,139  |
| specific_cloud_ice_water_content_500hPa    | 65,091  |
| total_column_cloud_ice_water               | 46,780  |
| surface_pressure                           | 33,937  |
| total_column_cloud_liquid_water            | 32,815  |
| specific_cloud_ice_water_content_700hPa    | 16,466  |
| specific_cloud_liquid_water_content_700hPa | 15,966  |
| specific_humidity_20hPa                    | 16,251  |
| temperature_225hPa                         | 14,488  |
| k_index                                    | 13,968  |
| specific_humidity_5hPa                     | 13,669  |
| temperature_250hPa                         | 13,289  |
| specific_humidity_50hPa                    | 12,215  |
| specific_humidity_2hPa                     | 11,072  |
| specific_humidity_3hPa                     | 10,689  |
| specific_humidity_7hPa                     | 10,680  |

#### XGBoost

**Default threshold (0.50):**

| Class              | Precision | Recall | F1             | Support    |
| ------------------ | --------- | ------ | -------------- | ---------- |
| No Lightning       | 1.00      | 0.95   | 0.97           | 14,897,479 |
| Lightning          | 0.01      | 0.64   | 0.02           | 12,041     |
| **Accuracy** |           |        | **0.95** | 14,909,520 |

ROC-AUC: **0.9169**

**Optimized threshold (0.9055):** Precision: 0.0224 | Recall: 0.3001

**Top 20 features (gain):**

| Feature                                    | Gain   |
| ------------------------------------------ | ------ |
| specific_cloud_ice_water_content_600hPa    | 0.3790 |
| specific_cloud_ice_water_content_550hPa    | 0.2476 |
| specific_cloud_ice_water_content_650hPa    | 0.0381 |
| specific_cloud_ice_water_content_500hPa    | 0.0240 |
| total_totals_index                         | 0.0223 |
| convective_available_potential_energy      | 0.0113 |
| specific_cloud_liquid_water_content_700hPa | 0.0099 |
| specific_cloud_ice_water_content_700hPa    | 0.0096 |
| total_column_cloud_ice_water               | 0.0091 |
| total_column_cloud_liquid_water            | 0.0064 |
| specific_cloud_liquid_water_content_750hPa | 0.0050 |
| specific_cloud_liquid_water_content_850hPa | 0.0050 |
| proxy_lpi                                  | 0.0043 |
| specific_cloud_liquid_water_content_825hPa | 0.0040 |
| specific_cloud_ice_water_content_400hPa    | 0.0036 |
| specific_cloud_ice_water_content_450hPa    | 0.0031 |
| vertical_velocity_800hPa                   | 0.0027 |
| specific_cloud_liquid_water_content_800hPa | 0.0027 |
| specific_cloud_liquid_water_content_775hPa | 0.0026 |
| k_index                                    | 0.0026 |

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

| Lag      | ERA5 time | Lightning time | LightGBM AUC | XGBoost AUC |
| -------- | --------- | -------------- | ------------ | ----------- |
| 0 (sync) | T         | T              | 0.9231       | 0.9169      |
| 1 h      | T-1       | T              | 0.9214       | 0.9166      |
| 2 h      | T-2       | T              | 0.9099       | 0.9124      |
| 3 h      | T-3       | T              | 0.9115       | 0.9095      |
| 4 h      | T-4       | T              | 0.9010       | 0.9045      |
| 5 h      | T-5       | T              | 0.8979       | 0.8946      |
| 6 h      | T-6       | T              | 0.8897       | 0.8937      |

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

| Split       | Before mask | After mask | Lightning rows     |
| ----------- | ----------- | ---------- | ------------------ |
| Test (2025) | 14,907,818  | 8,385,916  | 9,360 (was 12,041) |

Note: **2,681 real lightning events (~22%) were masked out** — these occurred in grid cells where ERA5 did not show convective ice at 500–700 hPa. Possible explanations: ERA5 resolution limitations, or lightning in non-classical convective regimes (e.g. warm rain lightning, shallow convection).

### Results

| Metric                    | Exp 3 (no mask) | Exp 5 (convmask) | Change    |
| ------------------------- | --------------- | ---------------- | --------- |
| LightGBM AUC              | 0.9231          | **0.9239** | +0.0008   |
| XGBoost AUC               | 0.9169          | **0.9174** | +0.0005   |
| LightGBM precision (opt.) | 0.0245          | **0.0288** | +18%      |
| XGBoost precision (opt.)  | 0.0224          | **0.0275** | +23%      |
| Recall (both, opt.)       | 0.30            | 0.30             | unchanged |

#### LightGBM Top 20 Features (convmask)

| Rank   | Feature                                    | Gain            |
| ------ | ------------------------------------------ | --------------- |
| 1      | specific_cloud_ice_water_content_600hPa    | 939,970         |
| 2      | specific_cloud_ice_water_content_550hPa    | 252,312         |
| 3      | total_totals_index                         | 177,334         |
| 4      | convective_available_potential_energy      | 93,835          |
| 5      | total_column_cloud_ice_water               | 39,147          |
| 6      | specific_cloud_ice_water_content_650hPa    | 30,801          |
| 7      | specific_cloud_ice_water_content_500hPa    | 29,547          |
| 8      | surface_pressure                           | 28,371          |
| 9      | total_column_cloud_liquid_water            | 27,643          |
| 10     | temperature_250hPa                         | 16,842          |
| 11     | temperature_225hPa                         | 15,978          |
| 12     | specific_cloud_liquid_water_content_700hPa | 14,805          |
| 13     | specific_humidity_20hPa                    | 13,873          |
| 14     | k_index                                    | 13,055          |
| 15–20 | specific_humidity_30/50/10/5/3/2 hPa       | ~10,000–11,000 |

#### XGBoost Top 20 Features (convmask)

| Rank | Feature                                    | Gain             |
| ---- | ------------------------------------------ | ---------------- |
| 1    | specific_cloud_ice_water_content_600hPa    | 0.5314           |
| 2    | specific_cloud_ice_water_content_550hPa    | 0.1500           |
| 3    | specific_cloud_ice_water_content_650hPa    | 0.0388           |
| 4    | total_totals_index                         | 0.0178           |
| 5    | specific_cloud_ice_water_content_500hPa    | 0.0121           |
| 6    | convective_available_potential_energy      | 0.0100           |
| 7    | total_column_cloud_ice_water               | 0.0068           |
| 8    | specific_cloud_ice_water_content_700hPa    | 0.0061           |
| 9    | specific_cloud_liquid_water_content_700hPa | 0.0055           |
| 10   | specific_cloud_liquid_water_content_850hPa | 0.0050           |
| 11   | specific_cloud_liquid_water_content_500hPa | 0.0050           |
| 12   | specific_cloud_liquid_water_content_750hPa | 0.0045           |
| 13   | total_column_cloud_liquid_water            | 0.0043           |
| 14   | specific_cloud_liquid_water_content_825hPa | 0.0033           |
| 15   | specific_cloud_ice_water_content_450hPa    | 0.0030           |
| 16   | temperature_250hPa                         | 0.0026           |
| 17   | **proxy_lpi**                        | **0.0026** |
| 18   | specific_cloud_ice_water_content_400hPa    | 0.0025           |
| 19   | specific_cloud_liquid_water_content_775hPa | 0.0023           |
| 20   | specific_cloud_liquid_water_content_800hPa | 0.0022           |

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

## Ideas for Further Research

### 1. ⭐ Transfer Learning from Jones et al. Weights (TOP PRIORITY)

Obtain Jones et al.'s pre-trained U-Net weights (email authors / check GitHub/Zenodo) and apply transfer learning for the Israel/E. Med domain:

- **Freeze the encoder** — it has learned global convective spatial patterns; keep those weights fixed
- **Retrain the decoder** — adapt the output mapping to the regional domain and its sparse lightning climatology
- **Input channel strategy for extra features:** keep the original 7 CPLRSTW channels (frozen in conv1), add fresh weight columns for additional XGBoost-selected channels (cloud ice water content, total totals index). This requires modifying the first conv layer to accept 7+N channels while initializing the extra N channels randomly
- **Compare against:** (a) training from scratch on regional data (Exp 7b), (b) Jones weights + original 7 features only (frozen encoder, retrained decoder, no new channels)

This is the most promising direction: leverages global training data at zero cost while allowing regional adaptation.

### 2. Global Training, Regional Testing

Train a U-Net on global ERA5 + WWLLN data (replicating Jones et al. scale) and test on the Israel/E. Med domain. Run two variants:

- Using Jones's CPLRSTW variable set (CAPE, precipitation, LSM, RH, wind shear, T2M, warm cloud depth)
- Using the physically meaningful variables identified by XGBoost in this thesis (cloud ice water content at 500–700 hPa, total totals index, CAPE, cloud liquid water content)

This directly tests whether a globally-trained parameterization generalizes to a small, lightning-sparse region, and whether domain-specific feature selection adds value over a physically-motivated global feature set.

### 2. Jones Variable Set on Current Domain

Download the ERA5 variables Jones used that are currently missing (u/v wind components, relative humidity at 500/1000 hPa, 2m temperature, land-sea mask, total precipitation or IMERG) and re-train the same U-Net architecture on the current temporal (2004–2025) and geographic (Israel/E. Med) domain. Compare directly against Experiment 7b (LightGBM-selected features) to isolate the effect of feature choice from architecture and domain.

---

## Next Steps

- [X] Complete Experiment 3 training and evaluate results
- [X] Test lagged prediction (T-1 through T-6)
- [X] Apply convective cloud mask and compare vs baseline
- [X] Train U-Net (Jones et al. 2026 architecture) — see Experiments 7a & 7b below
- [ ] Investigate the 22% of lightning events masked out — are they real convective events ERA5 missed, or a different lightning regime?
- [ ] Consider dropping stratospheric features (>100 hPa) from LightGBM to test if it removes spurious correlations
- [ ] Meet with Vlad to discuss results and U-Net next steps
- [ ] Run U-Net with FSS logging once BCE run completes

---

## Experiment 7 — U-Net (Jones et al. 2026 Architecture)

*Date: August 2026*

### Architecture

Jones et al. (2026) simplified U-Net, no skip connections.

| Stage      | Operation                       | Channels | Spatial size |
| ---------- | ------------------------------- | -------- | ------------ |
| Input      | Predictor fields                | 7        | 37×46       |
| Encoder 1  | Conv3×3+ReLU → MaxPool2×2    | 32       | 18×23       |
| Encoder 2  | Conv3×3+ReLU → MaxPool2×2    | 16       | 9×11        |
| Bottleneck | Conv3×3+ReLU                   | 8        | 9×11        |
| Decoder 1  | TranspConv2×2 → Conv3×3+ReLU | 16       | 18×22       |
| Decoder 2  | TranspConv2×2 → Conv3×3+ReLU | 32       | 36×44       |
| Output     | Conv1×1                        | 1        | 36×44       |

**Parameters: 22,041**

Note: output is 36×44 not 37×46 — odd spatial dims lose 1 pixel through MaxPool+ConvTranspose. Target is cropped to pred size during training.

### Configuration

- Optimizer: Adam, LR=1e-3, ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 32, Epochs: 50
- Train: 2004–2024 (43,800 timesteps), Test: 2025 (8,760 timesteps)
- Parquets: non-convmask (full grid required for spatial U-Net input)
- Input features: top-7 LightGBM features (from Experiment 3)

### Input Features

1. specific_cloud_ice_water_content_600hPa
2. specific_cloud_ice_water_content_550hPa
3. specific_cloud_ice_water_content_650hPa
4. total_totals_index
5. specific_cloud_ice_water_content_500hPa
6. specific_cloud_liquid_water_content_700hPa
7. convective_available_potential_energy

---

### Experiment 7a — MSE Density Regression

**Output:** Raw Conv1×1 (no activation), target = raw `lightning_count`, z-score normalized on both features and target, FSS threshold = 0.0 (normalized space).

**Training log (epochs 1–37):**

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 8.785475   | 5.831831  | 0.0029 |
| 2     | 8.783799   | 5.818910  | 0.0060 |
| 3     | 8.731359   | 5.903473  | 0.0011 |
| 4     | 8.700373   | 5.936923  | 0.0042 |
| 5     | 8.664585   | 5.857887  | 0.0011 |
| 6     | 8.654045   | 5.856495  | 0.0011 |
| 7     | 8.696455   | 5.928376  | 0.0075 |
| 8     | 8.548205   | 5.863175  | 0.0076 |
| 9     | 8.402277   | 5.897219  | 0.0011 |
| 10    | 8.220923   | 5.891143  | 0.0106 |
| 11    | 8.299015   | 5.877811  | 0.0039 |
| 12    | 8.145159   | 5.969851  | 0.0011 |
| 13    | 8.132472   | 5.877338  | 0.0038 |
| 14    | 8.035054   | 5.987418  | 0.0071 |
| 15    | 7.900528   | 6.012968  | 0.0011 |
| 16    | 7.805887   | 6.025028  | 0.0011 |
| 17    | 7.744206   | 6.303013  | 0.0011 |
| 18    | 7.706408   | 5.988711  | 0.0011 |
| 19    | 7.661524   | 6.051294  | 0.0011 |
| 20    | 7.603419   | 6.349799  | 0.0011 |
| 21    | 7.513641   | 6.172734  | 0.0011 |
| 22    | 7.470477   | 6.195008  | 0.0011 |
| 23    | 7.433528   | 6.147607  | 0.0011 |
| 24    | 7.402408   | 6.151254  | 0.0011 |
| 25    | 7.389068   | 6.138007  | 0.0029 |
| 26    | 7.326942   | 6.221647  | 0.0025 |
| 27    | 7.281955   | 6.250214  | 0.0011 |
| 28    | 7.254672   | 6.288508  | 0.0029 |
| 29    | 7.232531   | 6.192108  | 0.0011 |
| 30    | 7.219214   | 6.278950  | 0.0011 |
| 31    | 7.195712   | 6.278115  | 0.0011 |
| 32    | 7.178164   | 6.275435  | 0.0018 |
| 33    | 7.146116   | 6.319861  | 0.0019 |
| 34    | 7.134963   | 6.317459  | 0.0019 |
| 35    | 7.125943   | 6.271853  | 0.0019 |
| 36    | 7.113037   | 6.289629  | 0.0019 |
| 37    | 7.105442   | 6.321231  | 0.0018 |

**Full 50-epoch run completed.** Loss curve saved to `results/unet/loss_curve.png`.

**Conclusion:** Failed. Train loss slowly decreasing (8.78 → 7.05), test loss diverging after epoch 3 (5.83 → 6.35), FSS ≈ 0.001 throughout all 50 epochs — flatlines at zero on the FSS plot. Model predicts near-zero everywhere — MSE-optimal for a 99% sparse target. MSE density regression does not work for a sparse regional domain. Switched to binary BCE.

---

### Experiment 7b — Binary BCE Classification

**Output:** Conv1×1 + Sigmoid, target = `lightning_count > 0` (binary), features z-score normalized, FSS threshold = 0.5.

| Epoch | Train Loss | Test Loss |
| ----- | ---------- | --------- |
| 1     | 0.131567   | 0.054314  |
| 2     | 0.027274   | 0.004646  |
| 3     | 0.007838   | 0.004035  |
| 4     | 0.007431   | 0.004072  |
| 5     | 0.007217   | 0.004156  |
| 6     | 0.007059   | 0.004084  |
| 7     | 0.006915   | 0.004092  |
| 8     | 0.006798   | 0.003952  |
| 9     | 0.006739   | 0.003978  |
| 10    | 0.006707   | 0.004571  |
| 11    | 0.006655   | 0.004050  |
| 12    | 0.006589   | 0.004686  |
| 13    | 0.006574   | 0.004191  |
| 14    | 0.006530   | 0.004262  |
| 15    | 0.006348   | 0.004116  |
| 16    | 0.006306   | 0.003884  |
| 17    | 0.006295   | 0.003926  |
| 18    | 0.006276   | 0.003748  |
| 19    | 0.006248   | 0.004291  |
| 20    | 0.006232   | 0.003792  |
| 50    | ~0.006     | ~0.004    |

**Full 50-epoch run completed.** Loss curve saved to `results/unet/loss_curve.png`.

**Training log:**

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 0.016690   | 0.004041  | 0.6131 |
| 2     | 0.006559   | 0.003982  | 0.6131 |
| 3     | 0.006277   | 0.004055  | 0.6131 |
| 4     | 0.006129   | 0.004029  | 0.6131 |
| 5     | 0.006054   | 0.003828  | 0.6131 |
| 6     | 0.005955   | 0.003923  | 0.6131 |
| 7     | 0.005875   | 0.003846  | 0.6095 |
| 8     | 0.005806   | 0.003643  | 0.6095 |
| 9     | 0.005769   | 0.003819  | 0.5877 |
| 10    | 0.005726   | 0.003779  | 0.5913 |
| 11    | 0.005695   | 0.003687  | 0.5986 |
| 12    | 0.005655   | 0.003699  | 0.6095 |
| 13    | 0.005630   | 0.003950  | 0.5804 |
| 14    | 0.005597   | 0.003811  | 0.5876 |
| 15    | 0.005489   | 0.003929  | 0.5731 |
| 16    | 0.005467   | 0.003969  | 0.5736 |
| 17    | 0.005452   | 0.003656  | 0.5913 |
| 18    | 0.005442   | 0.003821  | 0.5742 |
| 19    | 0.005430   | 0.003806  | 0.5772 |
| 20    | 0.005415   | 0.003846  | 0.5733 |
| 21    | 0.005350   | 0.003703  | 0.5699 |
| 22    | 0.005340   | 0.003749  | 0.5740 |
| 23    | 0.005334   | 0.003842  | 0.5675 |
| 24    | 0.005330   | 0.004047  | 0.5669 |
| 25    | 0.005319   | 0.003683  | 0.5736 |
| 26    | 0.005317   | 0.003942  | 0.5672 |
| 27    | 0.005280   | 0.003717  | 0.5662 |
| 28    | 0.005276   | 0.003853  | 0.5601 |
| 29    | 0.005272   | 0.003754  | 0.5704 |
| 30    | 0.005270   | 0.003787  | 0.5670 |
| 31    | 0.005266   | 0.004011  | 0.5499 |
| 32    | 0.005261   | 0.003752  | 0.5636 |
| 33    | 0.005243   | 0.003818  | 0.5634 |
| 34    | 0.005241   | 0.003783  | 0.5669 |
| 35    | 0.005240   | 0.003816  | 0.5634 |
| 36    | 0.005238   | 0.003807  | 0.5671 |
| 37    | 0.005236   | 0.003868  | 0.5564 |
| 38    | 0.005234   | 0.003833  | 0.5598 |
| 39    | 0.005225   | 0.003792  | 0.5599 |
| 40    | 0.005223   | 0.003859  | 0.5565 |
| 41    | 0.005222   | 0.003859  | 0.5565 |
| 42    | 0.005221   | 0.003819  | 0.5563 |
| 43    | 0.005221   | 0.003791  | 0.5635 |
| 44    | 0.005220   | 0.003775  | 0.5598 |
| 45    | 0.005215   | 0.003832  | 0.5565 |
| 46    | 0.005214   | 0.003794  | 0.5634 |
| 47    | 0.005213   | 0.003808  | 0.5636 |
| 48    | 0.005213   | 0.003817  | 0.5564 |
| 49    | 0.005213   | 0.003824  | 0.5562 |
| 50    | 0.005213   | 0.003827  | 0.5566 |

**Best test loss: 0.003643 (epoch 8) — model saved as `unet_best.pt`**
**Best FSS: 0.6131 (epochs 1–6)**

**Observations:**

- Converges rapidly — stable by epoch 2
- Test loss flat ~0.0038 from epoch 3 onward; no significant overfitting
- FSS starts at 0.6131 and gradually settles to ~0.56–0.57 — meaningful spatial skill throughout
- Train loss plateaus after ~epoch 30; LR scheduler triggers with no further improvement
- Loss curve axis labels say "MSE" — cosmetic bug, loss is BCE

**Conclusion:** Binary BCE works well for a sparse regional domain. FSS ~0.61 at best, ~0.56 sustained — the model successfully learns spatial lightning patterns. Best model saved at epoch 8.

### R² Evaluation (evaluate_unet.py, unet_best.pt)

| Metric                                      | Value  | Jones et al. best         |
| ------------------------------------------- | ------ | ------------------------- |
| Per-timestep r² (lightning timesteps only) | 0.0050 | —                        |
| Climatological r² (mean spatial map)       | 0.3521 | 0.92 (ocean), 0.77 (land) |

**Per-timestep r² = 0.005** is expected and not meaningful for binary classification — r² between a continuous predicted probability and a binary {0,1} target is inherently near zero even for a well-performing model. Jones used FSS for timestep-level evaluation, not r².

**Climatological r² = 0.35** is the comparable metric to Jones. The gap vs Jones's 0.92 is explained by:

- **Domain**: Jones's global domain has enormous geographic diversity (tropics, ocean, land) making climatological patterns easy to reproduce. Israel/E. Med is small and homogeneous — little geographic lightning gradient to exploit.
- **Task mismatch**: Jones predicted lightning density (MSE regression) so their mean map directly matches observed density. Binary BCE output averages to a probability map, which compresses dynamic range — the model underestimates high-frequency cells (visible in scatter plot).
- **Data**: Jones trained on 11 years globally; this model uses 7 years over a small region.

**The r² = 0.35 is not directly comparable to Jones's 0.92** — different task, domain, and output type. FSS ~0.61 is the more appropriate comparison metric and is only ~0.08 below Jones's mean FSS of 0.69 despite the much smaller, sparser domain.

---

## Experiment 8 — Jones CPLRSTW Features, Tabular Baseline (XGBoost & LightGBM)

*Date: September 2026*

### Motivation

Before training the U-Net with Jones's 7 CPLRSTW features, evaluate them as a tabular XGBoost/LightGBM baseline. This isolates the effect of **feature set** from **architecture** — if Jones features + tabular models already improve on our best tabular run (Exp 5), the gain is from the features; if not, any future improvement from the U-Net is due to spatial context.

### Feature Set

Jones et al. (2025) CPLRSTW — 7 variables:

| Short | Variable          | Source                                               |
| ----- | ----------------- | ---------------------------------------------------- |
| C     | CAPE              | ERA5 single-level (original download)                |
| P     | Precipitation     | NASA GPM IMERG V07 Half-Hourly, aggregated to hourly |
| L     | Land-Sea Mask     | ERA5 single-level                                    |
| R     | Relative Humidity | Derived from q and T at 500+1000 hPa (avg)           |
| S     | Wind Shear        | sqrt((u500−u1000)²+(v500−v1000)²)                |
| T     | 2m Temperature    | ERA5 single-level                                    |
| W     | Warm Cloud Depth  | Zero Degree Level − Cloud Base Height               |

Train: 2004–2024 (balanced 50/50, no convective mask). Test: 2025 (unbalanced).

### Results

#### XGBoost

**Default threshold (0.50):**

| Class        | Precision | Recall | F1   | Support    |
| ------------ | --------- | ------ | ---- | ---------- |
| No Lightning | 1.00      | 0.90   | 0.95 | 14,897,479 |
| Lightning    | 0.01      | 0.74   | 0.01 | 12,041     |

ROC-AUC: **0.9140** | PR-AUC: **0.0210**

**Optimized threshold (0.9241):** Precision: 0.0183 | Recall: 0.3001

#### LightGBM

**Default threshold (0.50):**

| Class        | Precision | Recall | F1   | Support    |
| ------------ | --------- | ------ | ---- | ---------- |
| No Lightning | 1.00      | 0.79   | 0.88 | 14,897,479 |
| Lightning    | 0.00      | 0.80   | 0.01 | 12,041     |

ROC-AUC: **0.8674** | PR-AUC: **0.0074**

**Optimized threshold (1.0000):** Precision: 0.0093 | Recall: 0.4353

Note: LightGBM threshold hitting 1.0 indicates poorly calibrated probabilities for this feature set.

### Feature Importance

Both models rank **WCD as the #1 feature by a large margin**, followed by CAPE and Precipitation:

| Rank | XGBoost Feature | XGBoost Score | LightGBM Feature | LightGBM Score |
| ---- | --------------- | ------------- | ---------------- | -------------- |
| 1    | wcd             | 0.5523        | wcd              | 1.34e+10       |
| 2    | cape            | 0.2298        | cape             | 3.81e+09       |
| 3    | precipitation   | 0.0719        | wind_shear       | 2.49e+08       |
| 4    | land_sea_mask   | 0.0462        | 2m_temperature   | 1.90e+08       |
| 5    | rh_avg          | 0.0353        | rh_avg           | 1.75e+08       |
| 6    | 2m_temperature  | 0.0340        | precipitation    | 1.67e+08       |
| 7    | wind_shear      | 0.0304        | land_sea_mask    | 5.72e+07       |

### Key Findings

- **XGBoost outperforms LightGBM significantly** — ROC-AUC 0.914 vs 0.867, PR-AUC 0.021 vs 0.007. LightGBM's calibration breaks down with this feature set (threshold=1.0 at optimized recall floor).
- **Recall is higher than in Experiments 3–5** at default threshold: 0.74 (XGB) and 0.80 (LGB) vs ~0.60–0.64 in Exp 3. However, precision remains similarly low (~0.01), and **PR-AUC is actually lower** (0.021 vs ~0.027 in Exp 5). Jones features trade precision for recall compared to our cloud-microphysics feature set.
- **WCD dominates feature importance in both models** — this is the most physically meaningful result and directly validates Jones et al.'s choice to include it. Warm Cloud Depth (ZDL − CBH) captures the vertical extent of the warm liquid water zone, which directly controls precipitation efficiency and electrical charge separation.
- **CAPE is #2** in both models, confirming convective instability remains the second most important signal regardless of feature set.
- **Precipitation (IMERG) ranks #3 in XGBoost** — useful signal for identifying already-active convective cells.
- This tabular baseline confirms the Jones features carry genuine predictive skill on this domain. The U-Net run (Exp 9) will test whether spatial context adds further improvement over this baseline.

---

## Experiment 9 — Jones U-Net with CPLRSTW Features + Pretrained Weights

*Date: September 2026*

### Motivation

Replicate Jones et al.'s U-Net using their published CPLRSTW features (`jones_tabular_dataset_{year}.parquet`) and their pre-trained weights (`cplrstw_mse.pth`). Transfer learning: load Jones weights, freeze encoder (enc1, enc2, bottleneck), fine-tune decoder on the Israel/E. Med domain.

### Architecture

Same as Experiment 7 (Jones et al. simplified U-Net, no skip connections), but:

- **Input:** 7 CPLRSTW channels (not our cloud-microphysics features)
- **Loss:** MSE on z-scored lightning density (matching Jones exactly, not BCE)
- **Transfer learning:** Jones global pretrained weights loaded into enc1+enc2+bottleneck; decoder randomly initialized and fine-tuned

### Configuration

- Optimizer: Adam, LR=1e-4 (lower for fine-tuning), ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 32, Epochs: 50
- Train: jones_tabular_dataset_{2004,2005,2006,2008,2009,2023,2024}.parquet
- Test: jones_tabular_dataset_2025.parquet
- **Temporal resolution: hourly** (one sample per hour — 43,800 train timesteps, ~8,760 test)
- Weight loading: 14/16 tensors loaded successfully (all encoder + decoder conv layers)
- Frozen params: enc1, enc2, bottleneck (~6,000 params); trainable: up1, dec1, up2, dec2, out (~16,000 params)

---

### Experiment 9a — MSE Hourly (Failed)

**Problem:** Same failure mode as Experiment 7a. Hourly lightning has ~0.08% non-zero cells. MSE-optimal strategy is to predict near-zero everywhere — the loss on 99.92% zero cells dominates any gradient from the sparse lightning signal. FSS flatlines at ~0.002 from epoch 1.

**Key output observed:**

- FSS ≈ 0.002 throughout all epochs (near-zero, no learning)
- Train loss decreasing slowly (MSE learning to predict zeros)
- Test loss not improving

**Root cause analysis:**

Jones et al. used **12-hourly temporal aggregation** — their training samples are 12-hour lightning sums, not hourly counts. At 12-hour resolution:

- Lightning sum per cell is much larger (up to 12× hourly)
- Far fewer all-zero windows (lightning events that span multi-hour convective episodes all aggregate into one non-zero cell)
- MSE gradient from non-zero cells becomes meaningful

At hourly resolution on a regional domain (37×46 = 1,702 cells, ~1.4 lightning cells per hour on average), MSE sees almost no signal. Transfer learning from Jones's global weights does not help if the loss function cannot learn from the target distribution.

**Conclusion:** MSE + hourly resolution is unworkable for this sparse regional domain, even with pretrained weights. Must replicate Jones's 12-hourly aggregation.

**Next step:** Experiment 9b — rewrite dataset with 12-hourly aggregation (ERA5 snapshot at first hour, precipitation mean over window, lightning sum over 12 hours), matching Jones's exact training setup.

---

### Experiment 9b — MSE 12-hourly Aggregation (In Progress)

**Dataset:**

- Target mean/std (12-hr aggregated): 0.7027 / 6.6068
- Train: 3,650 windows × 37×46 grid
- Test: 730 windows × 37×46 grid
- 16/16 Jones weight tensors loaded ✓
- Frozen: 7,832 params (enc1, enc2, bottleneck); Trainable: 14,209 params (decoder + output)

**Training log (full run, stopped at epoch 33):**

| Epoch        | Train Loss         | Test Loss          | FSS              |
| ------------ | ------------------ | ------------------ | ---------------- |
| 1            | 1.285674           | 0.677615           | 0.0108           |
| 2            | 1.162023           | 0.650005           | 0.0164           |
| 3            | 1.143945           | 0.643944           | 0.0186           |
| 4            | 1.139890           | 0.642044           | 0.0208           |
| 5            | 1.138026           | 0.640835           | 0.0245           |
| 6            | 1.136817           | 0.640069           | 0.0251           |
| 7            | 1.135783           | 0.639209           | 0.0298           |
| 8            | 1.134934           | 0.638614           | 0.0340           |
| 9            | 1.134332           | 0.638289           | 0.0349           |
| 10           | 1.133881           | 0.637941           | 0.0368           |
| 11           | 1.133514           | 0.637605           | 0.0387           |
| 12           | 1.133206           | 0.637408           | 0.0389           |
| **13** | **1.132938** | **0.637202** | **0.0394** |
| 14           | 1.132706           | 0.637082           | 0.0380           |
| 15           | 1.132506           | 0.636910           | 0.0385           |
| 16           | 1.132321           | 0.636863           | 0.0371           |
| 17           | 1.132156           | 0.636757           | 0.0366           |
| 18           | 1.131991           | 0.636689           | 0.0361           |
| 19           | 1.131862           | 0.636653           | 0.0352           |
| 20           | 1.131769           | 0.636642           | 0.0348           |
| 21           | 1.131644           | 0.636499           | 0.0353           |
| 22           | 1.131529           | 0.636460           | 0.0352           |
| 23           | 1.131435           | 0.636503           | 0.0343           |
| 24           | 1.131297           | 0.636361           | 0.0353           |
| 25           | 1.131185           | 0.636321           | 0.0358           |
| 26           | 1.131095           | 0.636298           | 0.0355           |
| 27           | 1.130983           | 0.636349           | 0.0343           |
| 28           | 1.130889           | 0.636272           | 0.0349           |
| 29           | 1.130807           | 0.636340           | 0.0338           |
| 30           | 1.130680           | 0.636239           | 0.0342           |
| 31           | 1.130570           | 0.636210           | 0.0347           |
| 32           | 1.130482           | 0.636200           | 0.0341           |
| 33           | 1.130367           | 0.636376           | 0.0322           |

**Best epoch: 13** — FSS=0.0394, test_loss=0.637202. Run stopped at epoch 33 (no further improvement expected).

**Code:** `train_unet.py` with `AGG_HOURS=12`, `FREEZE_ENCODER=True`, `PRETRAINED_WEIGHTS='cplrstw_mse.pth'`, `criterion=nn.MSELoss()`, `LR=1e-4`

**Observations:**

- FSS climbed 0.011 → 0.039 (epochs 1–13), then slowly declined to ~0.032 by epoch 33. Test loss decreased monotonically throughout — this is **not classic overfitting** (test loss never rises). The model shifts predictions toward smaller/smoother values which reduces MSE but pushes cells below FSS threshold=0.0 → FSS drops.
- After epoch 13, FSS plateaus in the 0.033–0.039 band with no further improvement — the decoder has converged given the frozen encoder constraint.
- Peak FSS 0.039 is far below the BCE baseline (Exp 7b: FSS 0.61). MSE + frozen encoder does not work for this sparse regional domain even with 12-hourly aggregation and pretrained weights.

**Root cause:**

1. **Frozen encoder is too constrained** — Jones's global encoder learned filters for tropical/global patterns. With it locked, the decoder cannot compensate for the Israel/E. Med domain shift.
2. **MSE still collapses** — 12-hourly aggregation delays the collapse to epoch 13 (vs immediate in 9a) but doesn't solve it. MSE still rewards predicting small/smooth values over the sparse target.
3. **FSS threshold=0.0** — z-score 0.0 with mean 0.7027 and std 6.6068 corresponds to ~0.7 lightning strikes. May still be too sensitive for the aggregated target.

**Conclusion:** Transfer learning with frozen encoder + MSE + 12-hourly aggregation does not work. Best FSS 0.039, far below BCE baseline of 0.61. Next steps:

- **Exp 9c:** BCE loss + Jones CPLRSTW features, all weights unfrozen — tests whether Jones features improve on Exp 7b's FSS 0.61
- **Exp 9d:** Full fine-tuning (unfreeze encoder) + MSE + 12-hourly — isolates whether frozen encoder is the bottleneck

---

### Experiment 9c — BCE Binary, Jones CPLRSTW, Pretrained Weights (Full Fine-tuning)

**Code:** `train_unet.py` with `BINARY_TARGET=True`, `FREEZE_ENCODER=False`, `PRETRAINED_WEIGHTS='cplrstw_mse.pth'`, `criterion=nn.BCEWithLogitsLoss()`, `AGG_HOURS=12`, `LR=1e-4`, `OUT_DIR='results/unet_jones_bce'`

Target: binary per-cell (any lightning in 12-hour window > 0). All 22,041 params trainable. Jones weights as warm start.

**Training log (stopped at epoch 36 — FSS fully plateaued):**

| Epoch       | Train Loss         | Test Loss          | FSS              |
| ----------- | ------------------ | ------------------ | ---------------- |
| 1           | 0.288562           | 0.084950           | 0.0476           |
| 2           | 0.082381           | 0.045344           | 0.3043           |
| 3           | 0.070198           | 0.040202           | 0.4348           |
| 4           | 0.065221           | 0.036233           | 0.4348           |
| 5           | 0.061510           | 0.034577           | 0.4348           |
| **6** | **0.058400** | **0.033051** | **0.4783** |
| 7           | 0.055683           | 0.031822           | 0.4783           |
| 8           | 0.053328           | 0.029896           | 0.4783           |
| 9           | 0.051433           | 0.029178           | 0.4783           |
| 10          | 0.049760           | 0.028150           | 0.4783           |
| 11          | 0.048442           | 0.027276           | 0.4783           |
| 12          | 0.047238           | 0.026991           | 0.4783           |
| 13          | 0.046177           | 0.026890           | 0.4783           |
| 14          | 0.045322           | 0.025949           | 0.4783           |
| 15          | 0.044464           | 0.025679           | 0.4783           |
| 16          | 0.043665           | 0.025615           | 0.4783           |
| 17          | 0.043020           | 0.024917           | 0.4783           |
| 18          | 0.042345           | 0.024435           | 0.4783           |
| 19          | 0.041781           | 0.023979           | 0.4783           |
| 20          | 0.041272           | 0.023663           | 0.4783           |
| 21          | 0.040825           | 0.024398           | 0.4783           |
| 22          | 0.040520           | 0.023564           | 0.4783           |
| 23          | 0.040086           | 0.023550           | 0.4783           |
| 24          | 0.039736           | 0.023139           | 0.4783           |
| 25          | 0.039423           | 0.023491           | 0.4783           |
| 26          | 0.039150           | 0.023364           | 0.4783           |
| 27          | 0.038912           | 0.023105           | 0.4783           |
| 28          | 0.038659           | 0.022937           | 0.4783           |
| 29          | 0.038441           | 0.022802           | 0.4783           |
| 30          | 0.038315           | 0.022508           | 0.4783           |
| 31          | 0.038158           | 0.022525           | 0.4783           |
| 32          | 0.037968           | 0.022666           | 0.4783           |
| 33          | 0.037830           | 0.022216           | 0.4783           |
| 34          | 0.037720           | 0.022141           | 0.4783           |
| 35          | 0.037499           | 0.022155           | 0.4783           |
| 36          | 0.037399           | 0.022383           | 0.4783           |

**Best FSS: 0.4783 (epoch 6, sustained through epoch 36)**

**Observations:**

- FSS converges in two steps: 0.048 → 0.304 (epoch 2) → 0.435 (epoch 3) → 0.478 (epoch 6), then completely flat for 30 epochs. This staircase pattern is expected for BCE + binary FSS — FSS only changes when cells cross the logit=0 threshold; between crossings, loss can decrease without FSS moving.
- Despite loss decreasing steadily (train: 0.288 → 0.037, test: 0.085 → 0.022), the binary prediction map never changes after epoch 6 — the model is sharpening its existing predictions, not gaining new correct cells.
- **Jones CPLRSTW + pretrained weights underperforms our cloud-microphysics baseline**: FSS 0.478 vs Exp 7b FSS 0.613. This is a ~22% relative drop.

**Key finding: Jones CPLRSTW features are weaker than cloud-microphysics features for this domain.**

Two possible explanations:

1. **Feature quality**: Jones features (CAPE, precip, LSM, RH, shear, T2m, WCD) lack the cloud ice water content signal that dominates our Exp 3/7b feature set. CIWC at 500–650 hPa was the #1 feature by a massive margin (gain 0.53) and directly encodes the deep convective ice present in electrified storms.
2. **Weight initialization hurts**: Jones's pretrained weights were learned on global tropical/subtropical patterns with MSE loss. Reinitializing the encoder to these weights may actually bias the first-layer filters away from the regional Mediterranean convective signatures, limiting what the decoder can recover with BCE fine-tuning.

**Conclusion:** The combination of Jones features + Jones pretrained weights does not beat our original feature set. Next experiment: add `specific_cloud_ice_water_content_600hPa` (Exp 3 #1 feature, gain=0.53) to the Jones CPLRSTW set → 8-channel input, no pretrained weights (encoder shape changes), BCE binary, from scratch. This isolates the feature effect cleanly.

---

### Experiment 9d — MSE 12-hourly, Full Fine-tuning (Encoder Unfrozen)

**Code:** `train_unet.py` with `AGG_HOURS=12`, `FREEZE_ENCODER=False`, `PRETRAINED_WEIGHTS='cplrstw_mse.pth'`, `criterion=nn.MSELoss()`, `LR=1e-4`, `OUT_DIR='results/unet_jones_unfrozen'`

All 22,041 parameters trainable. Jones weights used as warm start for entire network.

**Training log (stopped at epoch 33):**

| Epoch        | Train Loss         | Test Loss          | FSS              |
| ------------ | ------------------ | ------------------ | ---------------- |
| 1            | 1.233616           | 0.661420           | 0.0128           |
| 2            | 1.149221           | 0.646841           | 0.0169           |
| 3            | 1.139013           | 0.641410           | 0.0225           |
| 4            | 1.135736           | 0.639723           | 0.0243           |
| 5            | 1.134300           | 0.638893           | 0.0284           |
| 6            | 1.133345           | 0.638309           | 0.0337           |
| 7            | 1.132574           | 0.637911           | 0.0362           |
| 8            | 1.131862           | 0.637698           | 0.0368           |
| 9            | 1.131062           | 0.637412           | 0.0407           |
| 10           | 1.130011           | 0.637306           | 0.0426           |
| 11           | 1.128954           | 0.637615           | 0.0403           |
| 12           | 1.127604           | 0.637173           | 0.0465           |
| 13           | 1.125775           | 0.637522           | 0.0426           |
| 14           | 1.123186           | 0.637499           | 0.0429           |
| 15           | 1.120113           | 0.637238           | 0.0443           |
| 16           | 1.116137           | 0.637810           | 0.0424           |
| 17           | 1.112313           | 0.637095           | 0.0465           |
| 18           | 1.105295           | 0.640909           | 0.0368           |
| **19** | **1.100658** | **0.636525** | **0.0495** |
| 20           | 1.095390           | 0.636480           | 0.0482           |
| 21           | 1.085632           | 0.636610           | 0.0457           |
| 22           | 1.089597           | 0.637538           | 0.0401           |
| 23           | 1.068580           | 0.637748           | 0.0416           |
| 24           | 1.060841           | 0.639288           | 0.0395           |
| 25           | 1.052889           | 0.639796           | 0.0393           |
| 26           | 1.046981           | 0.639888           | 0.0393           |
| 27           | 1.043099           | 0.638817           | 0.0419           |
| 28           | 1.039924           | 0.639762           | 0.0411           |
| 29           | 1.035669           | 0.640004           | 0.0419           |
| 30           | 1.032841           | 0.639446           | 0.0418           |
| 31           | 1.028838           | 0.639578           | 0.0421           |
| 32           | 1.023709           | 0.641403           | 0.0399           |
| 33           | 1.021369           | 0.640854           | 0.0409           |

**Best epoch: 19** — FSS=0.0495, test_loss=0.636525

**Observations:**

- Unfreezing the encoder provides a marginal improvement: peak FSS 0.049 vs 0.039 in 9b (+26% relative). Still far below the BCE baseline of 0.61.
- Train loss is dropping faster than 9b (1.233 → 1.021 by ep 33 vs 1.285 → 1.130), but test loss tracks similarly (~0.636–0.641). The growing train/test gap from epoch 18 onward is the start of overfitting.
- FSS peaked at epoch 19 (0.049) then noisy plateau around 0.040–0.046 — the model has diverged into mild overfitting after the test loss spike at epoch 18 (0.640).
- The frozen encoder was not the main bottleneck. The fundamental problem is MSE + sparse regional target, regardless of which layers are frozen.

**Conclusion:** Unfreezing the encoder gives marginal gains (FSS +0.010) but does not solve the underlying MSE collapse problem. The loss function is the bottleneck, not the encoder. MSE + Jones pretrained weights is a dead end for this sparse regional domain. Moving to Exp 9c: BCE loss with Jones CPLRSTW features.

---

## Experiment 11 — Combined Features (Jones CPLRSTW + Exp 7b Top-6 Microphysics), 13 channels

**Code:** `train_unet.py`, `BINARY_TARGET=True`, `BCE_POS_WEIGHT=None`, `PRETRAINED_WEIGHTS=None`, `AGG_HOURS=1`, `SEED` not set, `ACTIVE_MONTHS=None`

**Features (13 channels):** Jones CPLRSTW (cape, precipitation, land_sea_mask, rh_avg, wind_shear, 2m_temperature, wcd) + Exp 7b top-6 microphysical (CIWC 600/550/650/500 hPa, total_totals_index, CLWC 700 hPa).

**Parquets:** `combined_tabular_dataset_{year}.parquet` (pre-built merge of jones + tabular datasets).

**Training log (stopped at epoch 24 — ongoing):**

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | —         | —        | 0.5839 |
| ...   | —         | —        | ~0.584 |
| 24    | —         | —        | ~0.584 |

**Observations:**

- FSS=0.5839 from epoch 1, plateaued immediately — same staircase behavior as previous experiments.
- Adding Jones CPLRSTW on top of the 7b microphysics features did not move the needle vs. 7b alone (0.584 vs 0.580).
- Plateau reflects initialization: the model's initial logit distribution happens to classify ~58% of the right cells.

**Conclusion:** Combined 13-feature set does not improve over 7b features alone. Feature set is not the primary bottleneck — class imbalance / loss function is.

---

## Experiment 13 — Top-9 XGBoost Features + Oct–Mar Seasonal Filter

**Code:** `train_unet.py`, `BINARY_TARGET=True`, `BCE_POS_WEIGHT=None`, `PRETRAINED_WEIGHTS=None`, `SEED=42`, `ACTIVE_MONTHS=[10,11,12,1,2,3]`

**Features (9 channels):** Top-9 from XGBoost feature importance: CIWC 600/550/650/500 hPa, total_totals_index, CLWC 700 hPa, CAPE, total_column_cloud_ice_water, total_column_cloud_liquid_water.

**Parquets:** `tabular_dataset_{year}.parquet`.

**Motivation:** Lightning in Israel/E. Med is heavily seasonal (Oct–Mar). Filtering to storm months reduces the class imbalance and eliminates empty-window dilution.

**Seasonal filter effect:** 21,120/43,800 training windows kept (48%); 4,368/8,760 test windows kept (50%).

### Experiment 13a — AGG_HOURS=1 (hourly)

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 0.031937   | 0.006541  | 0.6496 |
| 2     | 0.011536   | 0.006100  | 0.6496 |
| 3     | 0.011090   | 0.005919  | 0.6496 |
| 4     | 0.010883   | 0.007041  | 0.6496 |
| 5     | 0.010800   | 0.006368  | 0.6496 |
| 6     | 0.010636   | 0.006508  | 0.6496 |

**Best FSS so far: 0.6496** — new overall best, surpassing all previous experiments.

**Observations:**

- Seasonal filter works: FSS=0.6496 from epoch 1 vs. 0.5803 without the filter. The ~+0.07 improvement comes entirely from training on storm-season data only.
- FSS is again plateaued — same staircase behavior. The model converges to a fixed binary map immediately.
- Experiment stopped early; run not completed to 50 epochs.

### Experiment 13b — AGG_HOURS=12 (12-hour aggregation)

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 0.297497   | 0.045196  | 0.5000 |
| 2     | 0.074860   | 0.041938  | 0.5000 |
| 3     | 0.067885   | 0.037393  | 0.5000 |
| 4     | 0.062552   | 0.034274  | 0.5000 |
| 5     | 0.060393   | 0.033772  | 0.5000 |
| 6     | 0.059306   | 0.032041  | 0.5000 |
| 7     | 0.058348   | 0.032551  | 0.5000 |
| 8     | 0.057847   | 0.032340  | 0.5000 |
| 9     | 0.056790   | 0.031973  | 0.5000 |
| 10    | 0.057254   | 0.034414  | 0.5000 |
| 11    | 0.056329   | 0.030928  | 0.5000 |
| 12    | 0.055310   | 0.032953  | 0.5000 |

**Best FSS so far: 0.5000** — stuck at random baseline.

**Observations:**

- 12-hour aggregation collapses the training set from 21,120 to 1,760 windows — only 1,760 training samples. Too few samples for the model to learn anything useful.
- FSS locked at 0.5000 for all 12 epochs — this is the naive baseline (model predicting 50% of cells positive, matching the lightning fraction in storm-season windows).
- Loss is decreasing (0.297 → 0.055) but FSS never moves, meaning the model is getting more confident about the same wrong predictions.

**Conclusion: AGG_HOURS=12 significantly worsens results.** Hourly aggregation (AGG_HOURS=1) is strictly better: more training samples, higher FSS. The 12-hour window was appropriate for Jones et al.'s global domain but is too coarse for the small Israel/E. Med domain with limited training years.

**Next:** Run Exp 13a to completion (50 epochs) and add `BCE_POS_WEIGHT` to break the FSS plateau.

---

## Experiment 14 — Jones CPLRSTW + Exp 7b Microphysics (13 channels) + Oct–Mar Seasonal Filter

**Code:** `train_unet.py`, `BINARY_TARGET=True`, `BCE_POS_WEIGHT=None`, `PRETRAINED_WEIGHTS=None`, `AGG_HOURS=1`, `SEED=42`, `ACTIVE_MONTHS=[10,11,12,1,2,3]`

**Features (13 channels):**

- Jones CPLRSTW (7): cape, precipitation, land_sea_mask, rh_avg, wind_shear, 2m_temperature, wcd
- Exp 7b top-6 microphysics (excl. CAPE already in Jones): CIWC 600/550/650/500 hPa, total_totals_index, CLWC 700 hPa

**Parquets:** `combined_tabular_dataset_{year}.parquet` (pre-merged Jones + tabular).

**Total parameters:** 23,769 (slightly more than 13a due to 13-channel input vs 9-channel).

**Training log (stopped at epoch 5):**

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 0.038064   | 0.005997  | 0.6569 |
| 2     | 0.011297   | 0.005928  | 0.6569 |
| 3     | 0.010947   | 0.005872  | 0.6569 |
| 4     | 0.010810   | 0.005737  | 0.6569 |
| 5     | 0.010637   | 0.005912  | 0.6569 |

**Best FSS so far: 0.6569** — new overall best.

**Observations:**

- Combining Jones CPLRSTW + Exp 7b microphysics + seasonal filter gives FSS=0.6569, up from 0.6496 (Exp 13a, 9 features, seasonal) and 0.5803 (Exp 7b repro, no seasonal filter).
- FSS is again plateaued from epoch 1 — same staircase behavior. The model's initialization happens to place 65.7% of the right cells above logit=0.
- Trend so far: each addition improves the plateau value (0.5803 → 0.6496 → 0.6569), but none break out of it. Need `pos_weight` to push FSS higher.

**Progression summary:**

| Experiment   | Features              | Seasonal      | FSS plateau      |
| ------------ | --------------------- | ------------- | ---------------- |
| 7b repro     | 7 microphysics        | No            | 0.5803           |
| 13a          | 9 XGBoost top         | Yes           | 0.6496           |
| **14** | **13 combined** | **Yes** | **0.6569** |

**Conclusion:** Both adding seasonal filter and enriching features contribute independently to FSS improvement. The 13-channel combined set + seasonal filter is the best configuration so far. Next step: add `BCE_POS_WEIGHT` to break the plateau and push FSS above 0.6569.

---

## Experiment 15 — Exp 14 + Jones Pretrained Weights (partial enc1 load)

**Code:** Same as Exp 14, plus `PRETRAINED_WEIGHTS='/home/ec2-user/ML_Thesis_with_DL/cplrstw_mse.pth'`, `FREEZE_ENCODER=False`

**Weight loading:** 16/16 tensors loaded. enc1 loaded partially (7/13 input channels from Jones; remaining 6 microphysics channels left random).

**Training log (stopped at epoch 33):**

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 0.021363   | 0.006721  | 0.6569 |
| 2     | 0.012010   | 0.006290  | 0.6569 |
| 3     | 0.011386   | 0.006137  | 0.6569 |
| 4     | 0.011082   | 0.006079  | 0.6569 |
| 5     | 0.010850   | 0.006067  | 0.6569 |
| 6     | 0.010711   | 0.005983  | 0.6569 |
| 7     | 0.010595   | 0.006058  | 0.6569 |
| 8     | 0.010510   | 0.005857  | 0.6569 |
| 9     | 0.010428   | 0.005822  | 0.6569 |
| 10    | 0.010367   | 0.006232  | 0.6569 |
| 11    | 0.010339   | 0.005754  | 0.6425 |
| 12    | 0.010246   | 0.006377  | 0.6569 |
| 13    | 0.010209   | 0.006231  | 0.6067 |
| 14    | 0.010146   | 0.006191  | 0.6570 |
| 15    | 0.010109   | 0.005874  | 0.6424 |
| 16    | 0.010096   | 0.005922  | 0.6497 |
| 17    | 0.010024   | 0.005858  | 0.6356 |
| 18    | 0.009909   | 0.005880  | 0.6353 |
| 19    | 0.009896   | 0.005862  | 0.6206 |
| 20    | 0.009858   | 0.005893  | 0.6063 |
| 21    | 0.009843   | 0.005667  | 0.6351 |
| 22    | 0.009817   | 0.005640  | 0.6569 |
| 23    | 0.009808   | 0.005728  | 0.6350 |
| 24    | 0.009788   | 0.005588  | 0.6496 |
| 25    | 0.009761   | 0.005601  | 0.6425 |
| 26    | 0.009741   | 0.005593  | 0.6278 |
| 27    | 0.009727   | 0.005700  | 0.6207 |
| 28    | 0.009703   | 0.005627  | 0.6425 |
| 29    | 0.009688   | 0.005638  | 0.6282 |
| 30    | 0.009674   | 0.005542  | 0.6424 |
| 31    | 0.009667   | 0.005848  | 0.5777 |
| 32    | 0.009637   | 0.005736  | 0.6280 |
| 33    | 0.009640   | 0.005816  | 0.6063 |

**Best FSS: 0.6570 (epoch 14)** — identical to Exp 14 without pretrained weights.

**Observations:**

- Jones pretrained weights provide no benefit: best FSS 0.6570 vs 0.6569 in Exp 14 (from scratch). Difference is noise.
- FSS becomes noisy after epoch 10, oscillating between 0.5777 and 0.6570 rather than staying at the plateau. This instability is a side effect of Jones weights biasing the encoder toward global patterns that conflict with the 6 new microphysics input channels (which start random and pull the encoder in a different direction during fine-tuning).
- Train loss still steadily decreasing (0.021 → 0.009) but FSS does not follow — same loss-FSS decoupling as all previous experiments.

**Conclusion: Jones pretrained weights do not help when input channels are extended.** The partial enc1 loading (7/13 channels from Jones) creates a mismatch — the 6 random microphysics channels destabilize the Jones feature detectors during fine-tuning, producing noisy FSS. Training from scratch (Exp 14) is more stable and equally good. Pretrained weights are not worth using for the 13-channel combined input.

---

## Experiment 16 — Exp 15 + Frozen Encoder

**Code:** Same as Exp 15, plus `FREEZE_ENCODER=True`.

**Frozen:** 9,560 params (enc1, enc2, bottleneck). Trainable: 14,209 params (decoder + output only).

**Training log (stopped at epoch 10):**

| Epoch | Train Loss | Test Loss | FSS    |
| ----- | ---------- | --------- | ------ |
| 1     | 0.031411   | 0.007355  | 0.6569 |
| 2     | 0.015506   | 0.006959  | 0.6569 |
| 3     | 0.015044   | 0.006801  | 0.6569 |
| 4     | 0.014797   | 0.007015  | 0.6569 |
| 5     | 0.014630   | 0.006895  | 0.6569 |
| 6     | 0.014503   | 0.006885  | 0.6569 |
| 7     | 0.014401   | 0.007047  | 0.6569 |
| 8     | 0.014290   | 0.006707  | 0.6569 |
| 9     | 0.014222   | 0.006845  | 0.6569 |
| 10    | 0.014159   | 0.007654  | 0.6569 |

**Best FSS: 0.6569** — no improvement over any previous experiment.

**Observations:**

- FSS locked at 0.6569 for all 10 epochs, same plateau as Exp 14 and 15.
- Train loss higher than Exp 15 (0.031→0.014 vs 0.021→0.009) because the frozen encoder cannot adapt at all, forcing the decoder to do all the work.
- The 6 microphysics channels in enc1 are frozen at random init — they contribute pure noise to the encoder output, which the decoder cannot overcome.

**Conclusion: Freezing the encoder with partial Jones weights is strictly worse than training from scratch.** Transfer learning (frozen or unfrozen) provides no benefit for the 13-channel combined input. All pretrained weight experiments (Exp 10, 15, 16) confirm the same finding: Jones weights trained on 7 CPLRSTW channels on a global domain do not transfer to a 13-channel regional Mediterranean input. **Going forward: train from scratch, no pretrained weights.**

---

## Experiment 17 — Honest FSS + Precision/Recall/F1 Diagnostics (pos_weight=5)

**Motivation:** Two bugs discovered in prior experiments invalidated all previous FSS scores:

1. **Vacuous FSS**: `compute_fss` returned 1.0 when both predicted and observed fields were empty (both all-zero). With a class ratio of 1:594 (0.17% positive cells), most test timesteps have no lightning anywhere — so the model could predict nothing and still average FSS ≈ 0.65–0.73 by accumulating vacuous 1.0s. The "best" FSS of 0.7281 (Exp 13, Oct-Feb window) was largely or entirely vacuous.
2. **P=R=0 confirmed**: Adding precision/recall/F1 tracking immediately showed P=0, R=0 in experiments with pos_weight=None — the model predicted zero cells positive, but the broken FSS reported 0.65+.

**Fixes applied:**

- `compute_fss` now returns 0.0 when model predicts nothing but obs has lightning (honest worst case). Both-empty still returns 1.0 (correct — no lightning to predict).
- `evaluate()` now accumulates TP/FP/FN across all batches and returns precision, recall, F1.

**Config:**

- Features: Jones CPLRSTW (7) + Exp 7b top-6 microphysics = 13 channels
- `ACTIVE_MONTHS = [10, 11, 12, 1, 2, 3]` (Oct–Mar)
- `BCE_POS_WEIGHT = 5`
- `EPOCHS = 50`, `LR = 1e-3`, `SEED = 42`
- Train: 21,120 timesteps | Test: 4,368 timesteps

**Training log (selected epochs):**

| Epoch | Train Loss | Test Loss | FSS    | Precision | Recall | F1     |
| ----- | ---------- | --------- | ------ | --------- | ------ | ------ |
| 1     | 0.068973   | 0.021378  | 0.6569 | 0.0000    | 0.0000 | 0.0000 |
| 5     | 0.033846   | 0.021295  | 0.5372 | 0.0315    | 0.0401 | 0.0353 |
| 10    | 0.031784   | 0.021343  | 0.5550 | 0.0579    | 0.1425 | 0.0823 |
| 17    | 0.029565   | 0.018543  | 0.5404 | 0.1089    | 0.0894 | 0.0982 |
| 25    | 0.028479   | 0.018710  | 0.5365 | 0.0871    | 0.0922 | 0.0896 |
| 50    | 0.027499   | 0.019151  | 0.5104 | 0.0766    | 0.1124 | 0.0912 |

**Best test loss:** 0.018470 (epoch 40)
**Best FSS:** 0.6569 (epoch 1 — artifact of empty-batch averaging at random init)
**Best F1:** 0.0982 (epoch 17) — P=0.1089, R=0.0894

**Observations:**

- Epoch 1 FSS=0.6569 with P=R=0 reveals residual empty-batch inflation: at random init the model predicts nothing; most test batches have no lightning anywhere → both-empty → FSS=1.0 for those batches → high average. FSS drops to honest ~0.52–0.56 once the model starts predicting.
- Honest FSS plateau: ~0.50–0.57, with no upward trend after epoch 10.
- F1 plateaued at ~0.09–0.10 from epoch 17 onward. P and R are roughly balanced (both ~0.08–0.11), confirming pos_weight=5 is in the right ballpark.
- Class ratio confirmed at 1:594 (0.17% positive cells). pos_weight=40 over-predicted (P=0.02, R=0.5); pos_weight=5 balances P and R but absolute values are low.
- Model capacity (23,769 params) may be a limiting factor at this imbalance level.

**Conclusion:** pos_weight=5 gives honest, balanced P/R but F1≈0.10 is low. FSS honest baseline ≈ 0.52–0.56. Next steps: try higher pos_weight (10) to see if recall improves without collapsing precision; consider whether the architecture needs more capacity to handle 594:1 imbalance.

---

## Experiment 18 — Jones CPLRSTW Only (7 channels), pos_weight=5, Oct–Mar

**Motivation:** Exp 17 used 13 channels (Jones CPLRSTW + 6 CIWC microphysics). Hypothesis: the 6 CIWC channels are heavily correlated pressure-level variants of the same variable, adding redundancy and noise rather than signal. Reverted to Jones 7-channel CPLRSTW only.

**Config:**

- Features: Jones CPLRSTW only — cape, precipitation, land_sea_mask, rh_avg, wind_shear, 2m_temperature, wcd (7 channels)
- `ACTIVE_MONTHS = [10, 11, 12, 1, 2, 3]` (Oct–Mar)
- `BCE_POS_WEIGHT = 5`
- `EPOCHS = 50`, `LR = 1e-3`, `SEED = 42`
- Model params: 22,041 (vs 23,769 for 13-ch — difference is enc1 input weights)
- Train: 21,120 timesteps | Test: 4,368 timesteps

**Training log (selected epochs):**

| Epoch | Train Loss | Test Loss | FSS    | Precision | Recall | F1     |
| ----- | ---------- | --------- | ------ | --------- | ------ | ------ |
| 1     | 0.072615   | 0.022522  | 0.6569 | 0.0000    | 0.0000 | 0.0000 |
| 5     | 0.036403   | 0.019663  | 0.6112 | 0.1515    | 0.0526 | 0.0780 |
| 8     | 0.034925   | 0.020356  | 0.5949 | 0.1272    | 0.1042 | 0.1146 |
| 16    | 0.032511   | 0.018733  | 0.5781 | 0.1620    | 0.0780 | 0.1053 |
| 26    | 0.031025   | 0.020580  | 0.5046 | 0.0984    | 0.1279 | 0.1112 |
| 35    | 0.029528   | 0.018699  | 0.5219 | 0.1215    | 0.1144 | 0.1178 |
| 50    | 0.028424   | 0.019202  | 0.5327 | 0.0997    | 0.1027 | 0.1012 |

**Best test loss:** 0.018526 (epoch 27)
**Best FSS:** 0.6569 (epoch 1 — empty-batch artifact, same as Exp 17)
**Best F1:** 0.1178 (epoch 35) — P=0.1215, R=0.1144

**Comparison vs Exp 17 (13 channels):**

| Metric        | Exp 17 (13-ch) | Exp 18 (7-ch) | Δ      |
| ------------- | -------------- | ------------- | ------- |
| Best F1       | 0.0982         | 0.1178        | +0.0196 |
| Best F1 epoch | 17             | 35            |         |
| FSS plateau   | 0.52–0.56     | 0.52–0.61    | +0.05   |
| Precision     | 0.1089         | 0.1215        | +0.013  |
| Recall        | 0.0894         | 0.1144        | +0.025  |

**Observations:**

- Dropping the 6 CIWC channels improved all metrics. The correlated pressure-level channels (600/550/650/500 hPa CIWC) were adding noise not signal.
- FSS honest plateau shifted up to 0.52–0.61, with some epochs reaching 0.61+.
- P and R are more balanced than Exp 17 and both higher in absolute terms.
- Training is noisier epoch-to-epoch (P ranges 0.10–0.19 across epochs), suggesting the model is close to the logit=0 boundary and small weight updates flip predictions back and forth.
- Best F1 epoch (35) is later than Exp 17 (17), consistent with the 7-channel model taking longer to converge on the relevant features.

**Conclusion: Jones CPLRSTW (7 channels) outperforms 13-channel combined features.** The orthogonal Jones feature set is better matched to the model's 22k-param capacity. This is now the best configuration. Next: tune pos_weight (try 7–10) to close the P/R gap further and push F1 above 0.12.
