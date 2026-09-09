"""
lightning_to_cerra_grid.py
--------------------------
Co-register LPATS and ILDN lightning strikes onto the CERRA Lambert conformal
conic grid using a BallTree nearest-neighbour lookup.

Key differences from lpats_to_era5_grid.py:
  - CERRA grid is NOT regular lat/lon → need BallTree, not histogram2d
  - 3-hourly time steps (not hourly)
  - Output files: lpats_on_cerra_grid_{year}.nc / ildn_on_cerra_grid_{year}.nc

Usage:
    python lightning_to_cerra_grid.py
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import xlrd
from sklearn.neighbors import BallTree


# ── LPATS XLS parsing (identical to lpats_to_era5_grid.py) ───────────────────

def _excel_serial_to_datetime(date_serial: float, time_frac: float) -> datetime.datetime:
    base = datetime.date(1899, 12, 30)
    date = base + datetime.timedelta(days=int(date_serial))
    total_sec = time_frac * 86400.0
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    return datetime.datetime(date.year, date.month, date.day, min(h, 23), m, s)


def _read_xls(fpath: str) -> pd.DataFrame:
    wb = xlrd.open_workbook(fpath)
    sh = wb.sheets()[0]
    rows_out = []
    for r in range(sh.nrows):
        row = [sh.cell_value(r, c) for c in range(sh.ncols)]
        if any(isinstance(row[i], str) and row[i].strip()
               for i in range(min(2, len(row)))):
            continue
        parsed = None
        if len(row) >= 6 and isinstance(row[1], float) and row[1] > 38000:
            parsed = (row[1], row[2], row[3], row[4],
                      row[5] if len(row) > 5 else None)
        elif len(row) >= 5 and isinstance(row[0], float) and row[0] > 38000:
            parsed = (row[0], row[1], row[2], row[3],
                      row[4] if len(row) > 4 else None)
        if parsed is None:
            continue
        date_s, time_s, lat, lon, amp = parsed
        if not (isinstance(lat, float) and 20.0 <= lat <= 40.0):
            continue
        if not (isinstance(lon, float) and 25.0 <= lon <= 50.0):
            continue
        try:
            dt = _excel_serial_to_datetime(date_s, time_s)
            rows_out.append({'datetime': dt, 'lat': lat, 'lon': lon, 'amp_kA': amp})
        except Exception:
            continue
    return pd.DataFrame(rows_out)


def load_all_lpats(xls_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(xls_dir, '*.xls')))
    if not files:
        raise FileNotFoundError(f"No .xls files found in {xls_dir}")
    parts = []
    for fpath in files:
        df = _read_xls(fpath)
        if len(df):
            parts.append(df)
        print(f"  {os.path.basename(fpath):<40} {len(df):>7,} rows")
    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = (combined
                .drop_duplicates(subset=['datetime', 'lat', 'lon'])
                .sort_values('datetime')
                .reset_index(drop=True))
    print(f"\nTotal: {before:,} → {len(combined):,} after dedup")
    return combined


# ── Core gridding function ────────────────────────────────────────────────────

def build_lightning_cerra_grid(
    df_strikes: pd.DataFrame,
    cerra_pressure_path: str,
    year: int,
    source: str,
    out_dir: str = 'data',
) -> str:
    """
    Grid lightning strikes for a given year onto the CERRA grid.

    Parameters
    ----------
    df_strikes          : DataFrame with columns [datetime, lat, lon]
    cerra_pressure_path : path to the CERRA pressure-level NetCDF for this year
                          (used to get the grid coordinates and time axis)
    year                : calendar year to process
    source              : 'lpats' or 'ildn' (used in output filename)
    out_dir             : output directory

    Returns
    -------
    str : path to the saved NetCDF file
    """
    print(f"\n{'='*60}")
    print(f"Gridding {source.upper()} lightning onto CERRA grid — year {year}")
    print(f"{'='*60}")

    # ── Load CERRA grid ───────────────────────────────────────────────────────
    ds = xr.open_dataset(cerra_pressure_path)
    cerra_times = pd.to_datetime(ds.valid_time.values)

    # CERRA NetCDF4 has 2D latitude/longitude auxiliary arrays
    # These are named 'latitude' and 'longitude' (2D, shape y × x)
    # Some files use 'lat'/'lon' — handle both
    if 'latitude' in ds.coords:
        lat2d = ds.coords['latitude'].values
        lon2d = ds.coords['longitude'].values
    elif 'lat' in ds.coords:
        lat2d = ds.coords['lat'].values
        lon2d = ds.coords['lon'].values
    else:
        # Try as data variables
        lat2d = ds.coords['latitude'].values if 'latitude' in ds else ds.coords['lat'].values
        lon2d = ds.coords['longitude'].values if 'longitude' in ds else ds.coords['lon'].values

    n_y, n_x = lat2d.shape
    n_cells   = n_y * n_x
    n_times   = len(cerra_times)

    print(f"CERRA grid: {n_y} × {n_x} = {n_cells:,} cells")
    print(f"CERRA times: {n_times} steps ({cerra_times[0]} → {cerra_times[-1]})")

    # ── Build BallTree over CERRA cell centres ────────────────────────────────
    # BallTree with haversine metric expects radians
    flat_lats = lat2d.ravel()
    flat_lons = lon2d.ravel()
    tree_coords = np.deg2rad(np.column_stack([flat_lats, flat_lons]))
    tree = BallTree(tree_coords, metric='haversine')
    print(f"BallTree built over {n_cells:,} CERRA cells.")

    # ── Filter strikes to year and domain ────────────────────────────────────
    lat_min, lat_max = float(flat_lats.min()), float(flat_lats.max())
    lon_min, lon_max = float(flat_lons.min()), float(flat_lons.max())
    time_min = cerra_times.min()
    time_max = cerra_times.max() + pd.Timedelta(hours=3)

    df = df_strikes[df_strikes['datetime'].dt.year == year].copy()
    df = df[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
        (df['datetime'] >= time_min) & (df['datetime'] < time_max)
    ].copy()
    print(f"Strikes in domain & time range: {len(df):,}")

    if len(df) == 0:
        raise ValueError(f"No {source.upper()} strikes found for {year} in CERRA domain")

    # ── Floor each strike to the nearest 3-hour CERRA timestep ───────────────
    # CERRA analysis times: 00, 03, 06, 09, 12, 15, 18, 21 UTC
    df['cerra_time'] = df['datetime'].dt.floor('3h')

    # ── Query BallTree: find nearest CERRA cell for each strike ──────────────
    print("Running BallTree nearest-neighbour lookup...")
    strike_coords = np.deg2rad(df[['lat', 'lon']].values)
    _, indices = tree.query(strike_coords, k=1)
    df['cell_flat_idx'] = indices.ravel()   # flat index into (n_y × n_x)
    df['y_idx'] = df['cell_flat_idx'] // n_x
    df['x_idx'] = df['cell_flat_idx'] % n_x

    # ── Build count array (time × y × x) ─────────────────────────────────────
    print("Accumulating strike counts per CERRA cell per 3-hour step...")
    time_index = {t: i for i, t in enumerate(cerra_times)}
    counts = np.zeros((n_times, n_y, n_x), dtype=np.int32)

    for _, row in df.iterrows():
        t_idx = time_index.get(row['cerra_time'])
        if t_idx is None:
            continue
        counts[t_idx, int(row['y_idx']), int(row['x_idx'])] += 1

    n_lightning_steps = (counts.sum(axis=(1, 2)) > 0).sum()
    print(f"3-hour steps with ≥1 strike: {n_lightning_steps:,} / {n_times:,}")
    print(f"Total strikes gridded: {counts.sum():,}")

    # ── Save as NetCDF ────────────────────────────────────────────────────────
    # Use same x/y dimensions as CERRA so files can be merged easily
    if 'x' in ds.dims and 'y' in ds.dims:
        x_coords = ds['x'].values
        y_coords = ds['y'].values
        dims = ['time', 'y', 'x']
        coords = {
            'time':      cerra_times,
            'y':         y_coords,
            'x':         x_coords,
            'latitude':  (['y', 'x'], lat2d),
            'longitude': (['y', 'x'], lon2d),
        }
    else:
        # fallback: use integer indices
        dims = ['time', 'y', 'x']
        coords = {
            'time':      cerra_times,
            'y':         np.arange(n_y),
            'x':         np.arange(n_x),
            'latitude':  (['y', 'x'], lat2d),
            'longitude': (['y', 'x'], lon2d),
        }

    da = xr.DataArray(
        counts,
        coords=coords,
        dims=dims,
        name='lightning_count',
        attrs={
            'units':  f'count per CERRA cell per 3h',
            'source': source.upper(),
        },
    )

    out_path = os.path.join(out_dir, f'{source}_on_cerra_grid_{year}.nc')
    da.to_netcdf(out_path)
    print(f"Saved to {out_path}")
    ds.close()
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import netCDF4 as nc
    XLS_DIR = 'data/lpats_xls'

    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024, 2025]

    # # ── LPATS ─────────────────────────────────────────────────────────────────
    # print("Loading all LPATS XLS files...")
    # df_lpats = load_all_lpats(XLS_DIR)

    # for year in LPATS_YEARS:
    #     cerra_path = f'data/cerra_pressure_level_{year}.nc'
    #     ds_raw = nc.Dataset(cerra_path)
    #     if not os.path.exists(cerra_path):
    #         print(f"Skipping LPATS {year} — CERRA not downloaded ({cerra_path})")
    #         continue
    #     build_lightning_cerra_grid(
    #         df_strikes=df_lpats,
    #         cerra_pressure_path=cerra_path,
    #         year=year,
    #         source='lpats',
    #     )

    # ── ILDN ──────────────────────────────────────────────────────────────────
    for year in ILDN_YEARS:
        cerra_path = f'data/cerra_pressure_level_{year}.nc'
        ildn_path  = f'data/{year}_for_yoav_including_cloud.txt'

        if not os.path.exists(cerra_path):
            print(f"Skipping ILDN {year} — CERRA not downloaded ({cerra_path})")
            continue
        if not os.path.exists(ildn_path):
            print(f"Skipping ILDN {year} — ILDN file not found ({ildn_path})")
            continue

        # Load ILDN (whitespace-separated text file)
        df_ildn = pd.read_csv(ildn_path, sep=r'\s+', header=None, engine='python')
        df_ildn.columns = ['date', 'time', 'lat', 'lon', 'peak_current_KA',
                           'multiplicity', 'sens_num', 'rise_time', 'fall_time',
                           'type', 'semi_minor_ellipse_km', 'semi_major_ellipse_km',
                           'ellipse_angle']
        df_ildn = df_ildn.iloc[:, :4].copy()
        df_ildn['datetime'] = pd.to_datetime(
            df_ildn['date'].astype(str) + ' ' + df_ildn['time'].astype(str)
        )
        build_lightning_cerra_grid(
            df_strikes=df_ildn,
            cerra_pressure_path=cerra_path,
            year=year,
            source='ildn',
        )
