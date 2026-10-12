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

Also stores `data_available` (time): 1 for hours covered by a .loc file, 0 for days with no
file. Hours without data are NOT the same as hours without lightning; build_jones_netcdf.py
turns them into NaN.

Usage:
    python process_wwln.py              # all years 2013-2025
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
YEARS     = list(range(2013, 2026))   # 2013–2025 inclusive

# ERA5 grid bounds (used only as fallback if no reference file is found)
LAT_MIN, LAT_MAX = 28.0, 43.435
LON_MIN, LON_MAX = -9.142, 39.292
GRID_RES = 0.25


def make_grid():
    """
    Return lat/lon arrays matching the ACTUAL ERA5 0.25° grid.

    ERA5 grid points sit at exact multiples of 0.25°.
    The domain bounds (LAT_MAX=43.435, LON_MIN=-9.142 etc.) are NOT on those
    multiples, so we snap them inward to the nearest 0.25° grid point:
      LAT_MAX 43.435 → floor to 43.25   (don't overshoot northward)
      LAT_MIN 28.0   → ceil  to 28.0    (already on grid)
      LON_MIN -9.142 → ceil  to -9.0    (don't overshoot westward)
      LON_MAX 39.292 → floor to 39.25   (don't overshoot eastward)
    This gives exactly the same lat/lon values that ARCO-ERA5 returns for
    these bounds, so WWLN and ERA5 share a perfectly aligned grid.
    """
    lat_max_snap = np.floor(LAT_MAX / GRID_RES) * GRID_RES   # 43.435 → 43.25
    lat_min_snap = np.ceil( LAT_MIN / GRID_RES) * GRID_RES   # 28.0   → 28.0
    lon_min_snap = np.ceil( LON_MIN / GRID_RES) * GRID_RES   # -9.142 → -9.0
    lon_max_snap = np.floor(LON_MAX / GRID_RES) * GRID_RES   # 39.292 → 39.25

    lats = np.arange(lat_max_snap, lat_min_snap - GRID_RES/2, -GRID_RES).astype('float32')
    lons = np.arange(lon_min_snap, lon_max_snap + GRID_RES/2,  GRID_RES).astype('float32')
    print(f"  Grid (snapped to ERA5 0.25° points): "
          f"lat {lats[-1]:.2f}–{lats[0]:.2f} ({len(lats)} pts), "
          f"lon {lons[0]:.2f}–{lons[-1]:.2f} ({len(lons)} pts)")
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
    available = np.zeros(n_t, dtype='int8')      # 1 = hour covered by a .loc file that was read

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
    bad_rows = 0

    # A stroke is kept only if it falls inside one of the grid cells (grid point ± half a cell).
    # Previously strokes just outside the outermost cells were clipped onto the edge row/column.
    half = GRID_RES / 2
    lat_lo, lat_hi = float(lats.min()) - half, float(lats.max()) + half
    lon_lo, lon_hi = float(lons.min()) - half, float(lons.max()) + half

    for fpath in loc_files:
        try:
            # Parse: date,time,lat,lon,residual,num_stations
            df = pd.read_csv(fpath, header=None,
                             names=['date','time','lat','lon','residual','nstations'],
                             skipinitialspace=True)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue

        # This day has data (even if no strokes fall in the domain) → mark its 24 hours available
        day = pd.Timestamp(os.path.basename(fpath)[1:9])          # A20131001.loc → 2013-10-01
        i0 = time_index.get(day)
        if i0 is not None:
            available[i0:i0 + 24] = 1

        total_strokes += len(df)

        # Combine date+time into datetime, floor to hour
        df['datetime'] = pd.to_datetime(
            df['date'].str.strip() + ' ' + df['time'].str.strip().str[:8],
            format='%Y/%m/%d %H:%M:%S',
            errors='coerce'
        ).dt.floor('h')

        bad_rows += int(df['datetime'].isna().sum())

        # Filter to strokes inside the grid cells
        mask = (
            (df['lat'] >= lat_lo) & (df['lat'] < lat_hi) &
            (df['lon'] >= lon_lo) & (df['lon'] < lon_hi) &
            df['datetime'].notna()
        )
        df = df[mask]
        domain_strokes += len(df)

        if df.empty:
            continue

        # Snap to nearest grid cell (use actual ERA5 grid origin, not LAT_MAX/LON_MIN)
        df['lat_idx'] = np.round((lats[0] - df['lat']) / GRID_RES).astype(int).clip(0, n_lat - 1)
        df['lon_idx'] = np.round((df['lon'] - lons[0]) / GRID_RES).astype(int).clip(0, n_lon - 1)
        df['t_idx']   = df['datetime'].map(time_index)

        # Drop rows where time is outside the year (shouldn't happen, but safety)
        df = df.dropna(subset=['t_idx'])
        df['t_idx'] = df['t_idx'].astype(int)

        # Accumulate counts
        np.add.at(counts, (df['t_idx'].values, df['lat_idx'].values, df['lon_idx'].values), 1)

    print(f"  Total strokes in files:  {total_strokes:,}")
    print(f"  Strokes in domain:       {domain_strokes:,}")
    print(f"  Non-zero grid cells:     {np.count_nonzero(counts):,}")
    if bad_rows:
        print(f"  WARNING: {bad_rows:,} rows with unreadable date/time were skipped")
    season = times.month >= 9
    missing_days = pd.DatetimeIndex(times[season & (available == 0)]).normalize().unique()
    if len(missing_days):
        print(f"  WARNING: no .loc file for {len(missing_days)} day(s) in Sep–Dec "
              f"(stored as data_available=0): {[d.strftime('%Y-%m-%d') for d in missing_days[:10]]}"
              + (' ...' if len(missing_days) > 10 else ''))

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
            ),
            'data_available': xr.DataArray(
                available,
                dims=['time'],
                attrs={'long_name': '1 = WWLN .loc file read for this hour, 0 = no data',
                       'note': 'lightning_count is 0 for hours without data; treat those as missing'}
            ),
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
                        help='Process a single year (default: all 2013–2025)')
    args = parser.parse_args()

    years_to_run = [args.year] if args.year else YEARS

    for year in years_to_run:
        process_year(year)

    print("\nAll done!")