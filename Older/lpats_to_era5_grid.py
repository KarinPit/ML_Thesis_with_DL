"""
lpats_to_era5_grid.py
---------------------
Convert LPATS lightning XLS files (Israel Electric Corporation / mentor data)
into ERA5-gridded lightning count NetCDF files — same format as
ildn_on_era5_grid_{year}.nc produced by ildn_to_era5_grid.py.

Handles two XLS layouts that appear in the dataset:
  Layout A (with header + units row):
      Number | Date | GMT | Latitude | Longitude | Current[kA]
  Layout B (no header, data from row 0):
      Date | GMT | Latitude | Longitude | Current[kA]

Usage (standalone):
    python lpats_to_era5_grid.py

Or import and call build_lpats_lightning_grid() directly.
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import xlrd


# ── XLS parsing ──────────────────────────────────────────────────────────────

def _excel_serial_to_datetime(date_serial: float, time_frac: float) -> datetime.datetime:
    """Convert Excel date serial + fractional-day time to a Python datetime."""
    base = datetime.date(1899, 12, 30)
    date = base + datetime.timedelta(days=int(date_serial))
    total_sec = time_frac * 86400.0
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    return datetime.datetime(date.year, date.month, date.day,
                             min(h, 23), m, s)


def _read_xls(fpath: str) -> pd.DataFrame:
    """
    Read one LPATS XLS file and return a DataFrame with columns:
        datetime (UTC), lat, lon, amp_kA
    Handles Layout A (has header/units rows) and Layout B (no header).
    """
    wb = xlrd.open_workbook(fpath)
    sh = wb.sheets()[0]

    rows_out = []
    for r in range(sh.nrows):
        row = [sh.cell_value(r, c) for c in range(sh.ncols)]

        # skip header / units rows — any string in the first two cells
        if any(isinstance(row[i], str) and row[i].strip()
               for i in range(min(2, len(row)))):
            continue

        # Layout A: [strike_num, date_serial, time_frac, lat, lon, amp]
        #   date_serial is in col 1 → value > 38000 (year ≥ 2004)
        # Layout B: [date_serial, time_frac, lat, lon, amp]
        #   date_serial is in col 0
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

        # basic sanity checks on coordinates
        if not (isinstance(lat, float) and 20.0 <= lat <= 40.0):
            continue
        if not (isinstance(lon, float) and 25.0 <= lon <= 50.0):
            continue

        try:
            dt = _excel_serial_to_datetime(date_s, time_s)
            rows_out.append({'datetime': dt, 'lat': lat,
                             'lon': lon, 'amp_kA': amp})
        except Exception:
            continue

    return pd.DataFrame(rows_out)


def load_all_lpats(xls_dir: str) -> pd.DataFrame:
    """
    Read all .xls files in xls_dir, concatenate, deduplicate on
    (datetime, lat, lon), and return a clean DataFrame sorted by datetime.
    """
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
    combined = combined.drop_duplicates(
        subset=['datetime', 'lat', 'lon']
    ).sort_values('datetime').reset_index(drop=True)
    print(f"\nTotal: {before:,} rows → {len(combined):,} after dedup "
          f"(removed {before - len(combined):,} duplicates)")
    return combined


# ── Gridding ─────────────────────────────────────────────────────────────────

def build_lpats_lightning_grid(
    xls_dir: str,
    era5_single_path: str,
    year: int,
    out_dir: str = 'data',
) -> str:
    """
    Grid LPATS lightning strikes for a given year onto the ERA5 grid.

    Parameters
    ----------
    xls_dir        : directory containing all LPATS .xls files
    era5_single_path : path to the ERA5 single-level NetCDF for this year
                       (used to get the grid coordinates and time axis)
    year           : calendar year to process (e.g. 2005)
    out_dir        : output directory

    Returns
    -------
    str : path to the saved NetCDF file (lpats_on_era5_grid_{year}.nc)
    """
    print(f"\n{'='*60}")
    print(f"Gridding LPATS data for year {year}")
    print(f"{'='*60}")

    # load all XLS strikes once, filter to the requested year
    print("Reading LPATS XLS files...")
    df_all = load_all_lpats(xls_dir)
    df = df_all[df_all['datetime'].dt.year == year].copy()
    print(f"Strikes for {year}: {len(df):,}")
    if len(df) == 0:
        raise ValueError(f"No LPATS strikes found for year {year}")

    # load ERA5 grid
    ds_single = xr.open_dataset(era5_single_path)
    era5_lats  = ds_single.latitude.values
    era5_lons  = ds_single.longitude.values
    era5_hours = pd.to_datetime(ds_single.time.values)

    # build histogram bin edges (cell centers ± half-step)
    lat_step = lon_step = 0.25
    lat_edges = np.sort(
        np.append(era5_lats + lat_step / 2, era5_lats[-1] - lat_step / 2)
    )
    lon_edges = np.append(
        era5_lons - lon_step / 2, era5_lons[-1] + lon_step / 2
    )

    # filter strikes to domain and ERA5 time window
    LAT_MIN, LAT_MAX = float(era5_lats.min()), float(era5_lats.max())
    LON_MIN, LON_MAX = float(era5_lons.min()), float(era5_lons.max())
    TIME_MIN = era5_hours.min()
    TIME_MAX = era5_hours.max() + pd.Timedelta(hours=1)

    df = df[
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
        (df['datetime'] >= TIME_MIN) & (df['datetime'] < TIME_MAX)
    ].copy()
    print(f"Strikes in domain & ERA5 time range: {len(df):,}")

    # floor to hourly bucket
    df['hour'] = df['datetime'].dt.floor('h')

    # build 3-D count array (time × lat × lon)
    n_lat, n_lon = len(era5_lats), len(era5_lons)
    counts = np.zeros((len(era5_hours), n_lat, n_lon), dtype=np.int32)

    hours_with_strikes = df.groupby('hour')
    for i, era5_hour in enumerate(era5_hours):
        if era5_hour not in hours_with_strikes.groups:
            continue
        df_h = hours_with_strikes.get_group(era5_hour)
        c, _, _ = np.histogram2d(
            df_h['lon'].values, df_h['lat'].values,
            bins=[lon_edges, lat_edges]
        )
        counts[i] = c.T[::-1]  # transpose + flip to match ERA5 N→S order

    n_lightning_hours = (counts.sum(axis=(1, 2)) > 0).sum()
    print(f"Hours with ≥1 strike: {n_lightning_hours:,} / {len(era5_hours):,}")

    # wrap as xarray DataArray co-registered to ERA5
    lightning_da = xr.DataArray(
        counts,
        coords={
            'time':      era5_hours,
            'latitude':  era5_lats,
            'longitude': era5_lons,
        },
        dims=['time', 'latitude', 'longitude'],
        name='lightning_count',
        attrs={
            'units':  'count per ERA5 cell per hour',
            'source': 'LPATS (IEC)',
        },
    )

    out_path = os.path.join(out_dir, f'lpats_on_era5_grid_{year}.nc')
    lightning_da.to_netcdf(out_path)
    print(f"Saved to {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Directory containing all the mentor's LPATS .xls files
    XLS_DIR = 'data/lpats_xls'

    # Years to process and their ERA5 single-level files
    # (ERA5 files must already exist — run download_era5_arco.py first)
    YEARS = [2004, 2005, 2006, 2008, 2009]

    for year in YEARS:
        era5_path = f'data/era5_single_level_{year}.nc'
        if not os.path.exists(era5_path):
            print(f"Skipping {year} — ERA5 not yet downloaded ({era5_path})")
            continue
        build_lpats_lightning_grid(
            xls_dir=XLS_DIR,
            era5_single_path=era5_path,
            year=year,
        )
