"""
build_jones_tabular_dataset.py
-------------------------------
Builds flat tabular parquet files using Jones et al. (2025) CPLRSTW features:

  C — CAPE                     ← era5_single_level_{year}.nc   (original ERA5)
  P — Precipitation             ← imerg_hourly_{year}.nc        (NASA IMERG)
  L — Land-Sea Mask (LSM)       ← jones_single_level_{year}.nc
  R — Relative Humidity (RH)    ← jones_pressure_level_{year}.nc (avg 500+1000 hPa)
  S — Wind Shear                ← jones_pressure_level_{year}.nc (500-1000 hPa)
  T — 2m Temperature (T2M)      ← jones_single_level_{year}.nc
  W — Warm Cloud Depth (WCD)    ← jones_single_level_{year}.nc

Output: data/jones_tabular_dataset_{year}.parquet
Columns: time, lat, lon, cape, precipitation, land_sea_mask, rh_avg,
         wind_shear, 2m_temperature, wcd, lightning_count

Usage:
    python build_jones_tabular_dataset.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 100   # timesteps per chunk

# ── Feature column names in output parquet ─────────────────────────────────────
JONES_FEATURES = [
    'cape',            # C
    'precipitation',   # P
    'land_sea_mask',   # L
    'rh_avg',          # R
    'wind_shear',      # S
    '2m_temperature',  # T
    'wcd',             # W
]


def build_jones_tabular_dataset(
    era5_single_path,      # original ERA5 single level (for CAPE)
    jones_single_path,     # jones_single_level_{year}.nc (T2M, CBH, ZDL, LSM, WCD)
    jones_pressure_path,   # jones_pressure_level_{year}.nc (wind_shear, rh_avg)
    imerg_path,            # imerg_hourly_{year}.nc (precipitation)
    lightning_path,        # ildn/lpats_on_era5_grid_{year}.nc
    out_dir='data',
):
    print(f"\nLoading datasets...")
    ds_era5   = xr.open_dataset(era5_single_path,   chunks={'time': CHUNK_SIZE})
    ds_jones_s = xr.open_dataset(jones_single_path,  chunks={'time': CHUNK_SIZE})
    ds_jones_p = xr.open_dataset(jones_pressure_path, chunks={'time': CHUNK_SIZE})
    ds_imerg  = xr.open_dataset(imerg_path,          chunks={'time': CHUNK_SIZE})
    ds_light  = xr.open_dataset(lightning_path,      chunks={'time': CHUNK_SIZE})

    # ── Align all datasets to common time axis (intersection) ─────────────────
    # IMERG and ERA5 may have slightly different time coverage
    times = ds_jones_s.time.values
    lats  = ds_jones_s.latitude.values
    lons  = ds_jones_s.longitude.values

    # Align IMERG to Jones time axis (reindex, fill missing with 0)
    ds_imerg = ds_imerg.reindex(time=times, method='nearest', tolerance='1h').fillna(0.0)
    ds_era5  = ds_era5.reindex(time=times,  method='nearest', tolerance='1h')
    ds_light = ds_light.reindex(time=times, method='nearest', tolerance='1h').fillna(0)

    n_hours, n_lat, n_lon = len(times), len(lats), len(lons)
    print(f"Grid: {n_hours} hours × {n_lat} lat × {n_lon} lon = {n_hours * n_lat * n_lon:,} rows")

    # Output filename
    start_year = pd.Timestamp(times[0]).year
    end_year   = pd.Timestamp(times[-1]).year
    ts       = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    out_path = f'{out_dir}/jones_tabular_dataset_{ts}.parquet'

    writer = None

    for t_start in range(0, n_hours, CHUNK_SIZE):
        t_end = min(t_start + CHUNK_SIZE, n_hours)
        sl    = slice(t_start, t_end)
        n_chunk = t_end - t_start

        print(f"  Processing hours {t_start}–{t_end} / {n_hours}...", end='\r')

        t_idx, lat_idx, lon_idx = np.meshgrid(
            np.arange(n_chunk), np.arange(n_lat), np.arange(n_lon), indexing='ij'
        )

        chunk = {
            'time': times[t_start:t_end][t_idx.ravel()],
            'lat':  lats[lat_idx.ravel()],
            'lon':  lons[lon_idx.ravel()],
        }

        # C — CAPE (from original ERA5 single level)
        chunk['cape'] = (
            ds_era5['convective_available_potential_energy']
            .isel(time=sl).compute().values.ravel()
        )

        # P — Precipitation (IMERG)
        chunk['precipitation'] = (
            ds_imerg['precipitation'].isel(time=sl).compute().values.ravel()
        )

        # L, T, W — LSM, T2M, WCD (from Jones single level)
        js = ds_jones_s.isel(time=sl).compute()
        chunk['land_sea_mask']   = js['land_sea_mask'].values.ravel()
        chunk['2m_temperature']  = js['2m_temperature'].values.ravel()
        chunk['wcd']             = js['wcd'].values.ravel()

        # R, S — RH avg, wind shear (from Jones pressure level — already derived, no level dim)
        jp = ds_jones_p.isel(time=sl).compute()
        chunk['rh_avg']     = jp['rh_avg'].values.ravel()
        chunk['wind_shear'] = jp['wind_shear'].values.ravel()

        # Target — lightning count
        chunk['lightning_count'] = (
            ds_light['lightning_count'].isel(time=sl).compute().values.ravel()
        )

        df_chunk = pd.DataFrame(chunk)

        if df_chunk.empty:
            continue

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
            print(f"\nColumns: {list(df_chunk.columns)}")
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"\nSaved → {out_path}")
    return out_path


if __name__ == "__main__":
    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024, 2025]

    for year in LPATS_YEARS:
        build_jones_tabular_dataset(
            era5_single_path   = f'data/era5_single_level_{year}.nc',
            jones_single_path  = f'data/jones_single_level_{year}.nc',
            jones_pressure_path= f'data/jones_pressure_level_{year}.nc',
            imerg_path         = f'data/imerg_hourly_{year}.nc',
            lightning_path     = f'data/lpats_on_era5_grid_{year}.nc',
        )

    for year in ILDN_YEARS:
        build_jones_tabular_dataset(
            era5_single_path   = f'data/era5_single_level_{year}.nc',
            jones_single_path  = f'data/jones_single_level_{year}.nc',
            jones_pressure_path= f'data/jones_pressure_level_{year}.nc',
            imerg_path         = f'data/imerg_hourly_{year}.nc',
            lightning_path     = f'data/ildn_on_era5_grid_{year}.nc',
        )
