import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 100  # number of time steps processed at a time


def build_tabular_dataset(era5_single_path, era5_pressure_path, lightning_path, out_dir='data', lpi_path=None, lag=0):
    """
    Build a flat tabular parquet from ERA5 + lightning + optional LPI.

    Parameters
    ----------
    lag : int
        Number of hours to shift ERA5 features back relative to lightning.
        lag=0  → synchronous (T → T): diagnostic, current default.
        lag=1  → ERA5 features at T-1 predict lightning at T.
        lag=k  → ERA5 features at T-k predict lightning at T.
        Output filename includes the lag: tabular_dataset_{year}_lag{k}.parquet
    """
    # Load NetCDF files lazily — nothing is read into RAM yet
    ds_single    = xr.open_dataset(era5_single_path,   chunks={'time': CHUNK_SIZE})
    ds_pressure  = xr.open_dataset(era5_pressure_path, chunks={'time': CHUNK_SIZE})
    ds_lightning = xr.open_dataset(lightning_path,     chunks={'time': CHUNK_SIZE})
    ds_lpi       = xr.open_dataset(lpi_path,           chunks={'time': CHUNK_SIZE}) if lpi_path else None

    # shared coordinates
    times  = ds_single.time.values
    lats   = ds_single.latitude.values
    lons   = ds_single.longitude.values
    levels = ds_pressure.level.values  # 37 pressure levels in hPa

    n_hours, n_lat, n_lon = len(times), len(lats), len(lons)
    print(f"Grid: {n_hours} hours × {n_lat} lat × {n_lon} lon = {n_hours * n_lat * n_lon} rows")

    # output path — include lag in filename so lag=0 and lag=1 don't overwrite each other
    start_year = pd.Timestamp(times[0]).year
    end_year   = pd.Timestamp(times[-1]).year
    ts       = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    lag_str  = f"_lag{lag}" if lag > 0 else ""
    out_path = f'{out_dir}/tabular_dataset_{ts}{lag_str}.parquet'

    if lag > 0:
        print(f"Lag mode: ERA5 features at T-{lag}, lightning at T. "
              f"Skipping first {lag} hours.")

    writer = None  # ParquetWriter opened on first chunk

    # with lag, ERA5 must start lag hours earlier than lightning
    # → loop lightning indices from lag..n_hours, ERA5 indices from 0..n_hours-lag
    for t_start in range(lag, n_hours, CHUNK_SIZE):
        t_end   = min(t_start + CHUNK_SIZE, n_hours)

        # lightning index range: t_start..t_end  (time T)
        # ERA5 index range:      t_start-lag..t_end-lag  (time T-k)
        lightning_slice = slice(t_start, t_end)
        era5_slice      = slice(t_start - lag, t_end - lag)

        chunk_times = times[t_start:t_end]  # label rows with lightning time T
        n_chunk = len(chunk_times)

        print(f"  Processing hours {t_start}–{t_end} / {n_hours}...", end='\r')

        # index arrays for this chunk
        t_idx, lat_idx, lon_idx = np.meshgrid(
            np.arange(n_chunk), np.arange(n_lat), np.arange(n_lon), indexing='ij'
        )

        chunk = {
            'time': chunk_times[t_idx.ravel()],
            'lat':  lats[lat_idx.ravel()],
            'lon':  lons[lon_idx.ravel()],
        }

        # single-level variables — ERA5 at T-lag
        ds_s = ds_single.isel(time=era5_slice).compute()
        for var in ds_s.data_vars:
            chunk[var] = ds_s[var].values.ravel()

        # pressure-level variables — ERA5 at T-lag
        ds_p = ds_pressure.isel(time=era5_slice).compute()
        for var in ds_p.data_vars:
            arr = ds_p[var].values  # shape: (n_chunk, n_levels, n_lat, n_lon)
            for lev_idx, level in enumerate(levels):
                chunk[f"{var}_{int(level)}hPa"] = arr[:, lev_idx, :, :].ravel()

        # proxy LPI (optional) — also at T-lag
        if ds_lpi is not None:
            chunk['proxy_lpi'] = ds_lpi['proxy_lpi'].isel(time=era5_slice).compute().values.ravel()

        # lightning target — always at T
        chunk['lightning_count'] = ds_lightning['lightning_count'].isel(time=lightning_slice).compute().values.ravel()

        # write chunk to parquet incrementally
        df_chunk = pd.DataFrame(chunk)
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
            print(f"\nColumns: {len(df_chunk.columns)}")
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"\nSaved to {out_path}")
    return None, out_path


if __name__ == "__main__":
    # ── Configuration ────────────────────────────────────────────────────────
    # lag=0  → synchronous T→T (default, current experiments)
    # lag=1  → ERA5 at T-1 predicts lightning at T  (1-hour forecast)
    # lag=3  → ERA5 at T-3 predicts lightning at T  (3-hour forecast)
    LAG = 0

    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024]
    # ─────────────────────────────────────────────────────────────────────────

    for year in LPATS_YEARS:
        build_tabular_dataset(
            era5_single_path=f'data/era5_single_level_{year}.nc',
            era5_pressure_path=f'data/era5_pressure_level_{year}.nc',
            lightning_path=f'data/lpats_on_era5_grid_{year}.nc',
            lpi_path=f'data/proxy_lpi_{year}.nc',
            lag=LAG,
        )

    for year in ILDN_YEARS:
        build_tabular_dataset(
            era5_single_path=f'data/era5_single_level_{year}.nc',
            era5_pressure_path=f'data/era5_pressure_level_{year}.nc',
            lightning_path=f'data/ildn_on_era5_grid_{year}.nc',
            lpi_path=f'data/proxy_lpi_{year}.nc',
            lag=LAG,
        )
