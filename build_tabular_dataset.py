import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

def build_tabular_dataset(era5_single_path, era5_pressure_path, lightning_path, out_dir='data'):
    # Load NetCDF files
    ds_single    = xr.open_dataset(era5_single_path)
    ds_pressure  = xr.open_dataset(era5_pressure_path)
    ds_lightning = xr.open_dataset(lightning_path)
    
    # shared coordinates
    times  = ds_single.time.values
    lats   = ds_single.latitude.values
    lons   = ds_single.longitude.values
    levels = ds_pressure.level.values  # the 37 pressure levels in hPa

    n_hours, n_lat, n_lon = len(times), len(lats), len(lons)
    print(f"Grid: {n_hours} hours × {n_lat} lat × {n_lon} lon = {n_hours * n_lat * n_lon} rows")

    # Build index arrays
    # Every combination of (hour, lat, lon) becomes one row
    t_idx, lat_idx, lon_idx = np.meshgrid(
        np.arange(n_hours), np.arange(n_lat), np.arange(n_lon), indexing='ij'
    )

    chunks = {
        'time': times[t_idx.ravel()],
        'lat':  lats[lat_idx.ravel()],
        'lon':  lons[lon_idx.ravel()],
    }

    # single-level variables (one column each)
    for var in ds_single.data_vars:
        chunks[var] = ds_single[var].values.ravel()
        print(f"  Added single-level: {var}")

    # pressure-level variables (one column per variable per level)
    for var in ds_pressure.data_vars:
        arr = ds_pressure[var].values  # shape: (n_hours, n_levels, n_lat, n_lon)
        for lev_idx, level in enumerate(levels):
            chunks[f"{var}_{int(level)}hPa"] = arr[:, lev_idx, :, :].ravel()
        print(f"  Added pressure-level: {var} × {len(levels)} levels")

    # lightning target column
    chunks['lightning_count'] = ds_lightning['lightning_count'].values.ravel()

    df = pd.DataFrame(chunks)

    print(f"\nFinal table shape: {df.shape}")
    print(f"Columns: {len(df.columns)} ({len(ds_single.data_vars)} single-level + "
          f"{len(ds_pressure.data_vars) * len(levels)} pressure-level + 3 index + 1 target)")

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'{out_dir}/tabular_dataset_{ts}.parquet'
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return df, out_path

if __name__ == "__main__":
    build_tabular_dataset(
        era5_single_path='data/era5_single_level_sample.nc',
        era5_pressure_path='data/era5_pressure_level_sample.nc',
        lightning_path='data/ildn_on_era5_grid.nc',
    )
