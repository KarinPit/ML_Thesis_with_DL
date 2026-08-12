#!/bin/bash
set -e  # stop on first error

echo "=== Step 1: Build tabular datasets ==="
python build_tabular_dataset.py

echo "=== Step 2: Balance datasets ==="
python balance_dataset.py

echo "=== Step 3: Unite parquets ==="
python unite_parquets.py

echo "=== Step 4: Train LightGBM + XGBoost ==="
python train_lightgbm_xgboost.py

echo "=== Step 5: Plot lag comparison ==="
python plot_lag_comparison.py

echo "=== Pipeline complete ==="
