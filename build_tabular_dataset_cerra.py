import os
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 50  # number of time steps per chunk (3-hourly, so 50 = ~6 days)


def build_tabular_dataset_cerra(cerra_pressure_path, lightning_path,
                                out_dir='data', lag=0):
    """
    Build a flat tabular parquet from CERRA pressure-level data + lightning.

    Differences from ERA5 version:
    - No single-level file (CERRA pressure file only)
    - No LPI (vertical velocity not downloaded)
    - No convective mask (requires w + ciwc, not available)
    - Time dim is 'valid_time', pressure dim is 'pressure_level'
    - Lat/lon are 2D arrays (y, x) on Lambert Conformal grid
    - 3-hourly time steps

    Parameters
    ----------
    cerra_pressure_path : str   path to cerra_pressure_level_{year}.nc
    lightning_path      : str   path to lpats/ildn_on_cerra_grid_{year}.nc
    out_dir             : str   output directory
    lag                 : int   number of 3-hour steps to shift features back
                                lag=0 → features at T predict lightning at T
                                lag=1 → features at T-3h predict lightning at T
    """
    print(f"\nLoading CERRA: {cerra_pressure_path}")
    ds_pressure  = xr.open_dataset(cerra_pressure_path, chunks={'valid_time': CHUNK_SIZE},
                                   mask_and_scale=False)
    ds_lightning = xr.open_dataset(lightning_path,      chunks={'time': CHUNK_SIZE})

    # ── Coordinates ───────────────────────────────────────────────────────────
    times  = pd.to_datetime(ds_pressure['valid_time'].values)
    levels = ds_pressure['pressure_level'].values       # e.g. [700, 600, 500] hPa

    # 2D lat/lon on CERRA Lambert grid
    lat2d = ds_pressure.coords['latitude'].values       # shape (n_y, n_x)
    lon2d = ds_pressure.coords['longitude'].values

    # if lat2d is all NaN, borrow from reference year
    if np.all(np.isnan(lat2d)):
        ref_path = os.path.join(os.path.dirname(cerra_pressure_path),
                                'cerra_pressure_level_2006.nc')
        print(f"  WARNING: lat/lon all NaN — borrowing grid from {ref_path}")
        ds_ref = xr.open_dataset(ref_path, mask_and_scale=False)
        lat2d  = ds_ref.coords['latitude'].values
        lon2d  = ds_ref.coords['longitude'].values
        ds_ref.close()

    n_y, n_x = lat2d.shape
    n_times   = len(times)
    n_cells   = n_y * n_x

    print(f"Grid: {n_times} steps × {n_y}×{n_x} = {n_times * n_cells:,} rows")
    print(f"Pressure levels: {levels}")

    # flat lat/lon repeated for every time step in a chunk
    flat_lat = lat2d.ravel()   # (n_cells,)
    flat_lon = lon2d.ravel()

    # ── Output path ───────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    start_year = times[0].year
    end_year   = times[-1].year
    ts      = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    lag_str = f"_lag{lag}" if lag > 0 else ""
    out_path = f'{out_dir}/tabular_dataset_cerra_{ts}{lag_str}.parquet'

    if lag > 0:
        print(f"Lag mode: CERRA features at T-{lag} steps, lightning at T.")

    writer = None

    for t_start in range(lag, n_times, CHUNK_SIZE):
        t_end = min(t_start + CHUNK_SIZE, n_times)

        lightning_slice = slice(t_start,       t_end)
        cerra_slice     = slice(t_start - lag, t_end - lag)

        chunk_times = times[t_start:t_end]
        n_chunk = len(chunk_times)

        print(f"  Steps {t_start}–{t_end} / {n_times}...", end='\r')

        # repeat each time stamp for every cell
        time_col = np.repeat(chunk_times, n_cells)
        lat_col  = np.tile(flat_lat, n_chunk)
        lon_col  = np.tile(flat_lon, n_chunk)

        chunk = {
            'time': time_col,
            'lat':  lat_col,
            'lon':  lon_col,
        }

        # ── Pressure-level features ──────────────────────────────────────────
        ds_p = ds_pressure.isel(valid_time=cerra_slice).compute()
        for var in ds_p.data_vars:
            arr = ds_p[var].values  # shape: (n_chunk, n_levels, n_y, n_x)
            if arr.ndim == 4:
                for lev_idx, level in enumerate(levels):
                    chunk[f"{var}_{int(level)}hPa"] = arr[:, lev_idx, :, :].ravel()
            elif arr.ndim == 3:
                # no pressure dimension (e.g. surface variable in same file)
                chunk[var] = arr.ravel()

        # ── Lightning target ─────────────────────────────────────────────────
        chunk['lightning_count'] = (
            ds_lightning['lightning_count']
            .isel(time=lightning_slice)
            .compute()
            .values
            .ravel()
        )

        df_chunk = pd.DataFrame(chunk)

        if df_chunk.empty:
            continue

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
            print(f"\nColumns ({len(df_chunk.columns)}): {list(df_chunk.columns)}")
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"\nSaved → {out_path}")
    return out_path


if __name__ == "__main__":
    # ── Configuration ─────────────────────────────────────────────────────────
    # lag is in units of 3-hour steps (CERRA timestep)
    # lag=0 → T→T, lag=1 → T-3h, lag=2 → T-6h, lag=4 → T-12h, lag=8 → T-24h
    LAGS = [0]   # equivalent to 0,3,6,12,24,48 hours
    # LAGS = [0, 1, 2, 4, 8, 16]   # equivalent to 0,3,6,12,24,48 hours

    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024, 2025]
    # ─────────────────────────────────────────────────────────────────────────

    for LAG in LAGS:
        print(f"\n{'='*60}\nBuilding CERRA tabular datasets — LAG={LAG} steps "
              f"({LAG*3}h)\n{'='*60}")

        for year in LPATS_YEARS:
            cerra_path     = f'data/cerra_pressure_level_{year}.nc'
            lightning_path = f'data/lpats_on_cerra_grid_{year}.nc'
            if not os.path.exists(cerra_path) or not os.path.exists(lightning_path):
                print(f"Skipping LPATS {year} — file(s) missing")
                continue
            build_tabular_dataset_cerra(
                cerra_pressure_path=cerra_path,
                lightning_path=lightning_path,
                lag=LAG,
            )

        for year in ILDN_YEARS:
            cerra_path     = f'data/cerra_pressure_level_{year}.nc'
            lightning_path = f'data/ildn_on_cerra_grid_{year}.nc'
            if not os.path.exists(cerra_path) or not os.path.exists(lightning_path):
                print(f"Skipping ILDN {year} — file(s) missing")
                continue
            build_tabular_dataset_cerra(
                cerra_pressure_path=cerra_path,
                lightning_path=lightning_path,
                lag=LAG,
            )