"""
unite_parquets.py
-----------------
Concatenate balanced parquet files from multiple years into a single
training parquet, sorted by time.

Edit TRAIN_YEARS to include whichever years are ready.
ERA5 for LPATS years (2004-2009) must be downloaded and the full pipeline
(lpats_to_era5_grid → calculate_lpi → build_tabular_dataset → balance_dataset)
must have run before adding those years here.
"""

import os
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = 'data'
# LAG      = 1

# All years to combine into the training set.
TRAIN_YEARS = [
    2004, 2005, 2006, 2008, 2009,  # LPATS
    2023, 2024,                     # ILDN
]

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    lags = [1, 2, 3, 4, 5, 6]

    for lag in lags:
        lag_str = f"_lag{lag}" if lag > 0 else ""
        parts   = []
        missing = []

        for year in TRAIN_YEARS:
            path = os.path.join(DATA_DIR, f'tabular_dataset_{year}{lag_str}_balanced.parquet')
            if not os.path.exists(path):
                print(f"  ⚠  Missing: {path} — skipping year {year}")
                missing.append(year)
                continue
            df_yr = pd.read_parquet(path)
            print(f"  {year}: {len(df_yr):,} rows")
            parts.append(df_yr)

        if not parts:
            raise RuntimeError("No parquet files found — nothing to combine.")

        years_used = [y for y in TRAIN_YEARS if y not in missing]
        years_str  = '_'.join(str(y) for y in years_used)
        out_path   = os.path.join(DATA_DIR, f'tabular_dataset_{years_str}{lag_str}_balanced.parquet')

        df = pd.concat(parts).sort_values('time').reset_index(drop=True)
        print(f"\nCombined: {len(df):,} rows across years {years_used}")
        df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")
