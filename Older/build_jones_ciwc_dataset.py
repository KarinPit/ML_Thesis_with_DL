"""
build_combined_dataset.py
--------------------------
One-time script: merges jones_tabular_dataset_{year}.parquet with the
Exp 7b microphysical features from tabular_dataset_{year}.parquet and
saves combined_tabular_dataset_{year}.parquet.

Combined feature set (13 channels):
  Jones CPLRSTW (7):
    cape, precipitation, land_sea_mask, rh_avg, wind_shear,
    2m_temperature, wcd
  Exp 7b top-6 (excl. cape which is already in Jones):
    specific_cloud_ice_water_content_600hPa  (rank 1, gain 0.531)
    specific_cloud_ice_water_content_550hPa  (rank 2, gain 0.150)
    specific_cloud_ice_water_content_650hPa  (rank 3, gain 0.039)
    total_totals_index                        (rank 4, gain 0.026)
    specific_cloud_ice_water_content_500hPa  (rank 5, gain 0.023)
    specific_cloud_liquid_water_content_700hPa (rank 6, gain 0.022)

Run once on EC2, then point train_unet.py at the new parquets.

Usage:
    python build_combined_dataset.py
"""

import os
import pandas as pd

YEARS     = [2004, 2005, 2006, 2008, 2009, 2023, 2024, 2025]
DATA_DIR  = 'data'
AUX_COLS  = [
    'specific_cloud_ice_water_content_600hPa',   # rank 1 (gain 0.531)
    'specific_cloud_ice_water_content_550hPa',   # rank 2 (gain 0.150)
    'specific_cloud_ice_water_content_650hPa',   # rank 3 (gain 0.039)
    'total_totals_index',                         # rank 4 (gain 0.026)
    'specific_cloud_ice_water_content_500hPa',   # rank 5 (gain 0.023)
    'specific_cloud_liquid_water_content_700hPa', # rank 6 (gain 0.022)
]
SORT_KEYS = ['time', 'lat', 'lon']

for year in YEARS:
    jones_path = os.path.join(DATA_DIR, f'jones_tabular_dataset_{year}.parquet')
    aux_path   = os.path.join(DATA_DIR, f'tabular_dataset_{year}.parquet')
    out_path   = os.path.join(DATA_DIR, f'combined_tabular_dataset_{year}.parquet')

    if not os.path.exists(jones_path):
        print(f"  SKIP {year}: missing jones parquet")
        continue
    if not os.path.exists(aux_path):
        print(f"  SKIP {year}: missing tabular parquet")
        continue
    if os.path.exists(out_path):
        print(f"  SKIP {year}: output already exists ({out_path})")
        continue

    print(f"Processing {year}...")
    jones = pd.read_parquet(jones_path).sort_values(SORT_KEYS).reset_index(drop=True)
    aux   = pd.read_parquet(aux_path, columns=AUX_COLS + SORT_KEYS) \
              .sort_values(SORT_KEYS).reset_index(drop=True)

    assert len(jones) == len(aux), \
        f"Row count mismatch: jones={len(jones)}, aux={len(aux)}"

    for col in AUX_COLS:
        jones[col] = aux[col].values
    jones.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}  ({len(jones):,} rows)")
    print(f"  Columns: {list(jones.columns)}")

print("Done.")
