import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 100  # number of time steps processed at a time


def build_tabular_dataset(era5_single_path, era5_pressure_path, lightning_path, out_dir='data', lpi_path=None):
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

    # output path
    start_year = pd.Timestamp(times[0]).year
    end_year   = pd.Timestamp(times[-1]).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    out_path = f'{out_dir}/tabular_dataset_{ts}.parquet'

    writer = None  # ParquetWriter opened on first chunk

    for t_start in range(0, n_hours, CHUNK_SIZE):
        t_end   = min(t_start + CHUNK_SIZE, n_hours)
        t_slice = slice(t_start, t_end)
        chunk_times = times[t_start:t_end]
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

        # single-level variables — load only this time chunk into RAM
        ds_s = ds_single.isel(time=t_slice).compute()
        for var in ds_s.data_vars:
            chunk[var] = ds_s[var].values.ravel()

        # pressure-level variables — load only this time chunk into RAM
        ds_p = ds_pressure.isel(time=t_slice).compute()
        for var in ds_p.data_vars:
            arr = ds_p[var].values  # shape: (n_chunk, n_levels, n_lat, n_lon)
            for lev_idx, level in enumerate(levels):
                chunk[f"{var}_{int(level)}hPa"] = arr[:, lev_idx, :, :].ravel()

        # proxy LPI (optional)
        if ds_lpi is not None:
            chunk['proxy_lpi'] = ds_lpi['proxy_lpi'].isel(time=t_slice).compute().values.ravel()

        # lightning target
        chunk['lightning_count'] = ds_lightning['lightning_count'].isel(time=t_slice).compute().values.ravel()

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
    build_tabular_dataset(
        era5_single_path='data/era5_single_level_2023.nc',
        era5_pressure_path='data/era5_pressure_level_2023.nc',
        lightning_path='data/ildn_on_era5_grid_2023.nc',
        lpi_path='data/proxy_lpi_2023.nc',
    )
