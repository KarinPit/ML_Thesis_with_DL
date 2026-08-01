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

# All years to combine into the training set.
# Comment out years whose balanced parquets don't exist yet.
TRAIN_YEARS = [
    # LPATS years (partial coverage — uncomment as ERA5 downloads complete)
    2004,   # Sep–Dec only
    2005,   # Jan–Nov (missing May)
    2006,   # Jan–Aug only
    2008,   # Sep–Dec only
    2009,   # Jan–Sep only

    # Modern ILDN years (full year)
    2023,
    2024,
]
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    parts = []
    missing = []

    for year in TRAIN_YEARS:
        path = os.path.join(DATA_DIR, f'tabular_dataset_{year}_balanced.parquet')
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
    out_path   = os.path.join(DATA_DIR, f'tabular_dataset_{years_str}_balanced.parquet')

    df = pd.concat(parts).sort_values('time').reset_index(drop=True)
    print(f"\nCombined: {len(df):,} rows across years {years_used}")
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
