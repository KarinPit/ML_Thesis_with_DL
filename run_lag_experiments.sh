#!/bin/bash
# run_lag_experiments.sh
# ----------------------
# Runs the full pipeline for lag=1 through lag=6.
# lag=0 (synchronous) is already done — results in metrics_train2004_test2025.json
#
# For each lag:
#   1. build_tabular_dataset.py  — ERA5 at T-k, lightning at T
#   2. balance_dataset.py        — 50/50 undersample
#   3. unite_parquets.py         — combine all years
#   4. train_lightgbm_xgboost.py — train & evaluate, saves metrics JSON
#
# Usage:
#   chmod +x run_lag_experiments.sh
#   ./run_lag_experiments.sh
#
# Each lag takes roughly as long as one full pipeline run (~2-3 hours).
# Total estimated time: 12-18 hours. Run with caffeinate or in a tmux session.

PYTHON=/home/ec2-user/ML_Thesis_with_DL/thesis/bin/python

for LAG in 1 2 3 4 5 6; do
    echo ""
    echo "============================================================"
    echo "  Starting LAG = $LAG  ($(date))"
    echo "============================================================"

    # Step 1: build tabular datasets with this lag
    echo "[LAG=$LAG] Step 1/4: Building tabular datasets..."
    sed -i "s/^LAG = .*/LAG = $LAG/" build_tabular_dataset.py
    $PYTHON build_tabular_dataset.py
    if [ $? -ne 0 ]; then echo "ERROR in build_tabular_dataset.py for lag=$LAG"; exit 1; fi

    # Step 2: balance each year
    echo "[LAG=$LAG] Step 2/4: Balancing datasets..."
    sed -i "s/^    LAG = .*/    LAG = $LAG/" balance_dataset.py
    $PYTHON balance_dataset.py
    if [ $? -ne 0 ]; then echo "ERROR in balance_dataset.py for lag=$LAG"; exit 1; fi

    # Step 3: unite all years
    echo "[LAG=$LAG] Step 3/4: Uniting years..."
    sed -i "s/^LAG      = .*/LAG      = $LAG/" unite_parquets.py
    $PYTHON unite_parquets.py
    if [ $? -ne 0 ]; then echo "ERROR in unite_parquets.py for lag=$LAG"; exit 1; fi

    # Step 4: train and evaluate
    echo "[LAG=$LAG] Step 4/4: Training models..."
    sed -i "s/^    LAG = .*/    LAG = $LAG/" train_lightgbm_xgboost.py
    $PYTHON train_lightgbm_xgboost.py
    if [ $? -ne 0 ]; then echo "ERROR in train_lightgbm_xgboost.py for lag=$LAG"; exit 1; fi

    # Cleanup: delete large unbalanced parquets for this lag — only keep balanced ones
    echo "[LAG=$LAG] Cleaning up unbalanced parquets..."
    for YEAR in 2004 2005 2006 2008 2009 2023 2024; do
        rm -f data/tabular_dataset_${YEAR}_lag${LAG}.parquet
    done
    # also delete the combined balanced (large) — only the per-year balanced ones are needed
    # to re-run unite if needed; the combined is re-creatable
    rm -f data/tabular_dataset_2004_2005_2006_2008_2009_2023_2024_lag${LAG}_balanced.parquet

    echo "[LAG=$LAG] Done! ($(date))"
done

# Reset all scripts back to LAG=0
sed -i "s/^LAG = .*/LAG = 0/" build_tabular_dataset.py
sed -i "s/^    LAG = .*/    LAG = 0/" balance_dataset.py
sed -i "s/^LAG      = .*/LAG      = 0/" unite_parquets.py
sed -i "s/^    LAG = .*/    LAG = 0/" train_lightgbm_xgboost.py

echo ""
echo "All lag experiments complete! Run plot_lag_results.py to visualise."
