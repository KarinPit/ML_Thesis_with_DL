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
*Pending — training in progress.*

---

## Next Steps
- [ ] Complete Experiment 3 training and evaluate results
- [ ] Consider dropping stratospheric features (>100hPa) based on feature importance analysis
- [ ] Test lagged prediction (T-1, T-2 atmospheric features → T lightning)
- [ ] Meet with Vlad to discuss results and U-Net next steps
