"""
process_wwln.py
---------------
Converts raw WWLN .loc files into hourly gridded NetCDF files on the ERA5 0.25° grid.

Input:  data/WWLN/{year}/{Mon}{year}/A{YYYYMMDD}.loc
Output: data/wwln_on_era5_grid_{year}.nc
        Dims:  time (hourly) × lat × lon
        Var:   lightning_count (int16)

WWLN .loc format (comma-separated):
    date, time, lat, lon, residual, num_stations
    e.g.: 2013/04/01,00:00:00.891805, 29.3548, -96.7936, 6.6, 7

Usage:
    python process_wwln.py              # all years 2013-2022
    python process_wwln.py --year 2013  # single year
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR  = 'data'
WWLN_DIR  = os.path.join(DATA_DIR, 'WWLN')
YEARS     = list(range(2013, 2025))

# ERA5 grid bounds (from jones_single_level files)

LAT_MIN, LAT_MAX = 30.970, 43.435
LON_MIN, LON_MAX = -9.142, 39.292
GRID_RES = 0.25


def make_grid():
    """Return lat/lon arrays matching the ERA5 grid."""
    lats = np.arange(LAT_MAX, LAT_MIN - GRID_RES/2, -GRID_RES).astype('float32')
    lons = np.arange(LON_MIN, LON_MAX + GRID_RES/2,  GRID_RES).astype('float32')
    return lats, lons


def process_year(year: int):
    lats, lons = make_grid()
    n_lat, n_lon = len(lats), len(lons)

    # Hourly time axis for the full year
    times = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left')
    n_t   = len(times)

    print(f"\n{'='*50}")
    print(f"  Processing WWLN year {year}")
    print(f"  Grid: {n_t} hours × {n_lat} lat × {n_lon} lon")
    print(f"{'='*50}")

    # Accumulator
    counts = np.zeros((n_t, n_lat, n_lon), dtype='int16')

    # Build time → index map (floor to hour)
    time_index = {t: i for i, t in enumerate(times)}

    # Find all .loc files for this year
    pattern = os.path.join(WWLN_DIR, str(year), f'*{year}', f'A{year}????.loc')
    loc_files = sorted(glob.glob(pattern))

    if not loc_files:
        print(f"  WARNING: no .loc files found at {pattern}")
        return None

    print(f"  Found {len(loc_files)} .loc files")

    total_strokes = 0
    domain_strokes = 0

    for fpath in loc_files:
        try:
            # Parse: date,time,lat,lon,residual,num_stations
            df = pd.read_csv(fpath, header=None,
                             names=['date','time','lat','lon','residual','nstations'],
                             skipinitialspace=True)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue

        total_strokes += len(df)

        # Combine date+time into datetime, floor to hour
        df['datetime'] = pd.to_datetime(
            df['date'].str.strip() + ' ' + df['time'].str.strip().str[:8],
            format='%Y/%m/%d %H:%M:%S',
            errors='coerce'
        ).dt.floor('h')

        # Filter to domain
        mask = (
            (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
            (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
            df['datetime'].notna()
        )
        df = df[mask]
        domain_strokes += len(df)

        if df.empty:
            continue

        # Snap to nearest grid cell
        df['lat_idx'] = np.round((LAT_MAX - df['lat']) / GRID_RES).astype(int).clip(0, n_lat - 1)
        df['lon_idx'] = np.round((df['lon'] - LON_MIN) / GRID_RES).astype(int).clip(0, n_lon - 1)
        df['t_idx']   = df['datetime'].map(time_index)

        # Drop rows where time is outside the year (shouldn't happen, but safety)
        df = df.dropna(subset=['t_idx'])
        df['t_idx'] = df['t_idx'].astype(int)

        # Accumulate counts
        np.add.at(counts, (df['t_idx'].values, df['lat_idx'].values, df['lon_idx'].values), 1)

    print(f"  Total strokes in files:  {total_strokes:,}")
    print(f"  Strokes in domain:       {domain_strokes:,}")
    print(f"  Non-zero grid cells:     {np.count_nonzero(counts):,}")

    # Build xarray Dataset
    ds = xr.Dataset(
        {
            'lightning_count': xr.DataArray(
                counts,
                dims=['time', 'lat', 'lon'],
                attrs={
                    'units':     'count',
                    'long_name': 'WWLN lightning stroke count per grid cell per hour',
                    'source':    'World Wide Lightning Location Network (WWLN)',
                }
            )
        },
        coords={
            'time': times,
            'lat':  lats,
            'lon':  lons,
        }
    )
    ds.attrs['year']        = str(year)
    ds.attrs['grid']        = f'{GRID_RES}deg ERA5 grid, lat {LAT_MIN}–{LAT_MAX}, lon {LON_MIN}–{LON_MAX}'
    ds.attrs['created_by']  = 'process_wwln.py'

    out_path = os.path.join(DATA_DIR, f'wwln_on_era5_grid_{year}.nc')
    ds.to_netcdf(out_path)
    print(f"  Saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=None,
                        help='Process a single year (default: all 2013–2022)')
    args = parser.parse_args()

    years_to_run = [args.year] if args.year else YEARS

    for year in years_to_run:
        process_year(year)

    print("\nAll done!")