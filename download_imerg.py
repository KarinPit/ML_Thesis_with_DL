"""
download_imerg.py
-----------------
Downloads NASA GPM IMERG Final Precipitation L3 Half-Hourly 0.1° V07 data
for the Israel/Eastern Mediterranean domain, aggregates to hourly resolution,
regrids to the ERA5 0.25° grid, and saves as NetCDF.

This is the precipitation source used by Jones et al. (2025) — NOT ERA5 precipitation.

Prerequisites:
  1. Create a free NASA Earthdata account: https://urs.earthdata.nasa.gov
  2. Install earthaccess: pip install earthaccess h5py scipy --break-system-packages

First-time setup:
  Run `earthaccess.login(strategy="interactive")` once to save credentials locally.
  After that, subsequent runs authenticate automatically.

Output:
  data/imerg_hourly_{ts}.nc  — hourly precipitation (mm/hr) on 0.25° ERA5 grid
                               (missing hours/pixels are NaN, never 0)

Regridding (--regrid):
  conservative (default) — area-weighted average of the 0.1° IMERG cells inside each 0.25° cell
                           (first-order conservative, as in Jones et al.)
  bilinear               — the original method (point interpolation). Use this if you re-download
                           only some years and want them consistent with files made earlier.

Usage:
    python download_imerg.py                                          # all years, skip existing files
    python download_imerg.py --years 2025 --overwrite                 # re-download one year
    python download_imerg.py --years 2025 --overwrite --regrid bilinear
"""

import os
import argparse
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import h5py
from scipy.interpolate import RegularGridInterpolator
import earthaccess

# ── Domain (match ERA5 grid exactly) ──────────────────────────────────────────
LAT_MIN, LAT_MAX = 28.0, 43.435   # extended south to include Negev / Eilat (~29.5°N)
LON_MIN, LON_MAX = -9.142, 39.292

# ERA5 0.25° target grid for this domain
ERA5_LATS = np.arange(np.ceil(LAT_MIN * 4) / 4, LAT_MAX + 0.001, 0.25)
ERA5_LONS = np.arange(np.ceil(LON_MIN * 4) / 4, LON_MAX + 0.001, 0.25)

# ── IMERG product ──────────────────────────────────────────────────────────────
IMERG_SHORT_NAME = 'GPM_3IMERGHH'
IMERG_VERSION    = '07'
PRECIP_VAR       = 'precipitation'   # V07: 'precipitation'; V06: 'precipitationCal'
REGRID           = 'conservative'    # 'conservative' (area average) or 'bilinear' (original)


def inspect_imerg_file(filepath):
    """Print HDF5 structure of one IMERG file — use for debugging."""
    def print_tree(name, obj):
        print(f"  {name}: {obj}")
    with h5py.File(filepath, 'r') as f:
        print(f"\n=== IMERG file structure: {os.path.basename(filepath)} ===")
        f.visititems(print_tree)
        if 'Grid' in f:
            print(f"\nGrid keys: {list(f['Grid'].keys())}")
            if PRECIP_VAR in f['Grid']:
                print(f"  {PRECIP_VAR} shape: {f['Grid'][PRECIP_VAR].shape}")
            if 'time' in f['Grid']:
                print(f"  time: {f['Grid']['time'][:]}")
                print(f"  time attrs: {dict(f['Grid']['time'].attrs)}")


def read_imerg_granule(filepath):
    """Read one IMERG HDF5 half-hourly file → (time, lat, lon, precip_mm_hr)."""
    with h5py.File(filepath, 'r') as f:
        # V07: precipitation shape is (1, lon, lat)
        grid   = f['Grid']
        var    = PRECIP_VAR if PRECIP_VAR in grid else 'precipitationCal'
        precip = grid[var][0]          # (lon, lat) after dropping time dim
        lons   = f['Grid/lon'][:]      # shape (3600,)
        lats   = f['Grid/lat'][:]      # shape (1800,)

        # Time: IMERG V07 uses GPS epoch = 1980-01-06 00:00:00 UTC
        t_val  = int(f['Grid/time'][0])
        t_unit = f['Grid/time'].attrs.get('units', b'seconds since 1980-01-06 00:00:00 UTC')
        t_unit = t_unit.decode() if isinstance(t_unit, bytes) else t_unit
        # Parse reference date from units string e.g. "seconds since 1980-01-06 00:00:00 UTC"
        ref_str = t_unit.split('since')[-1].strip().replace(' UTC', '')
        ref     = pd.Timestamp(ref_str)
        time    = ref + pd.Timedelta(seconds=t_val)

    # shape after [0]: (lon=3600, lat=1800) — transpose to (lat, lon)
    precip = precip.T   # now (lat=1800, lon=3600)

    precip = np.where(precip < 0, np.nan, precip)   # fill values (-9999.9) = missing → NaN, not 0
    return time, lats, lons, precip


def _cell_edges(centers):
    """Cell edges for a regular 1-D grid of cell centres (ascending)."""
    d = np.diff(centers).mean()
    return np.concatenate([centers - d / 2, [centers[-1] + d / 2]])


def _overlap_weights(src_edges, dst_edges):
    """W[i, j] = fraction of destination cell i covered by source cell j (1-D, ascending edges)."""
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None],  src_edges[None, 1:])
    return np.clip(hi - lo, 0, None) / (dst_edges[1:] - dst_edges[:-1])[:, None]


def crop_and_regrid(lats_src, lons_src, precip, target_lats, target_lons, method=REGRID):
    """Crop to domain and regrid from the 0.1° IMERG grid to the 0.25° ERA5 grid.
    NaN (missing) source pixels are ignored; a target cell with less than half of its area
    covered by valid IMERG pixels becomes NaN."""
    # Crop source to slightly wider than domain (for interpolation boundary)
    lat_mask = (lats_src >= LAT_MIN - 0.2) & (lats_src <= LAT_MAX + 0.2)
    lon_mask = (lons_src >= LON_MIN - 0.2) & (lons_src <= LON_MAX + 0.2)

    lats_c  = lats_src[lat_mask]
    lons_c  = lons_src[lon_mask]
    precip_c = precip[np.ix_(lat_mask, lon_mask)]

    if method == 'bilinear':
        # Original method: bilinear point interpolation onto the ERA5 0.25° grid
        interp = RegularGridInterpolator(
            (lats_c, lons_c), precip_c, method='linear', bounds_error=False, fill_value=np.nan
        )
        grid_lats, grid_lons = np.meshgrid(target_lats, target_lons, indexing='ij')
        regridded = interp(np.stack([grid_lats.ravel(), grid_lons.ravel()], axis=-1))
        return regridded.reshape(len(target_lats), len(target_lons))

    # First-order conservative: area-weighted mean of the IMERG cells overlapping each ERA5 cell.
    # Latitude overlaps are measured in sin(lat), which makes them proportional to true area.
    w_lat = _overlap_weights(np.sin(np.deg2rad(_cell_edges(lats_c))),
                             np.sin(np.deg2rad(_cell_edges(np.asarray(target_lats)))))
    w_lon = _overlap_weights(_cell_edges(lons_c), _cell_edges(np.asarray(target_lons)))
    valid = np.isfinite(precip_c)
    num = w_lat @ np.where(valid, precip_c, 0.0) @ w_lon.T
    cov = w_lat @ valid.astype(float) @ w_lon.T            # fraction of each cell with valid data
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cov >= 0.5, num / cov, np.nan)


def download_imerg(time_range, out_dir='data', tmp_dir='data/tmp_imerg',
                   overwrite=False, regrid=REGRID):
    """
    Download IMERG half-hourly granules, aggregate to hourly, regrid, save NetCDF.
    Every hour of the period is written; hours without IMERG data are NaN.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    start_year = pd.Timestamp(time_range.start).year
    end_year   = pd.Timestamp(time_range.stop).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    out_path = f'{out_dir}/imerg_hourly_{ts}.nc'

    if os.path.exists(out_path) and not overwrite:
        print(f"Output already exists: {out_path}  (use --overwrite to re-download)")
        return out_path

    # A date-only end ('2025-12-31') can be read as midnight, which drops the last day.
    t_end = time_range.stop if 'T' in str(time_range.stop) else f"{time_range.stop}T23:59:59"

    # ── Search for granules ───────────────────────────────────────────────────
    print(f"Searching IMERG granules: {time_range.start} → {time_range.stop}")
    results = earthaccess.search_data(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(time_range.start, t_end),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    print(f"Found {len(results)} granules (2 per hour × ~{len(results)//2} hours)")

    # ── Download granules ─────────────────────────────────────────────────────
    print(f"Downloading to {tmp_dir}/ ...")
    files = earthaccess.download(results, tmp_dir)
    files = sorted(files)
    print(f"Downloaded {len(files)} files")

    # ── Read, crop, regrid all granules ──────────────────────────────────────
    print("Reading and regridding granules...")
    # Inspect first file to verify structure
    if files:
        inspect_imerg_file(files[0])

    records = []   # list of (time, precip_array)
    for fpath in files:
        try:
            t, lats_src, lons_src, precip = read_imerg_granule(fpath)
            p_regrid = crop_and_regrid(lats_src, lons_src, precip, ERA5_LATS, ERA5_LONS, method=regrid)
            records.append((t, p_regrid))
        except Exception as e:
            print(f"  WARNING: could not read {os.path.basename(fpath)}: {e}")

    print(f"Successfully read {len(records)} / {len(files)} granules")
    if not records:
        print(f"ERROR: no IMERG data for {time_range.start} → {time_range.stop}. IMERG Final is "
              f"published ~3.5 months after the fact — this period may not be available yet.")
        return None

    # ── Aggregate two 30-min granules → one hourly value ─────────────────────
    print("Aggregating to hourly...")
    df_times = pd.DatetimeIndex([r[0] for r in records])
    hour_keys = df_times.floor('h')
    unique_hours = sorted(set(hour_keys))

    hourly_precip = []
    hourly_times  = []
    for hr in unique_hours:
        idx = [i for i, h in enumerate(hour_keys) if h == hr]
        stacked = np.stack([records[i][1] for i in idx], axis=0)
        with warnings.catch_warnings():               # all-NaN pixels → NaN, silently
            warnings.simplefilter('ignore', RuntimeWarning)
            hourly_precip.append(np.nanmean(stacked, axis=0))   # mean of 30-min rates → hourly rate
        hourly_times.append(hr)

    precip_arr = np.stack(hourly_precip, axis=0)   # (time, lat, lon)

    # ── Build xarray Dataset and save ────────────────────────────────────────
    ds_out = xr.Dataset(
        {'precipitation': (['time', 'latitude', 'longitude'], precip_arr,
                           {'long_name': 'IMERG calibrated precipitation rate',
                            'units': 'mm/hr'})},
        coords={
            'time':      hourly_times,
            'latitude':  ERA5_LATS,
            'longitude': ERA5_LONS,
        }
    )

    # ── Coverage check: write every hour of the period, NaN where IMERG had nothing ──
    expected = pd.date_range(pd.Timestamp(time_range.start).floor('D'),
                             pd.Timestamp(t_end).floor('h'), freq='h')
    ds_out = ds_out.reindex(time=expected)
    missing = np.isnan(ds_out['precipitation'].values).all(axis=(1, 2))
    if missing.any():
        by_month = pd.DatetimeIndex(expected[missing]).to_period('M').value_counts().sort_index()
        print(f"WARNING: {int(missing.sum())} of {len(expected)} hours have no IMERG data (saved as NaN):")
        for month, n in by_month.items():
            print(f"    {month}: {n} hours missing")
        print("  IMERG Final appears ~3.5 months after the fact; re-run later with --overwrite.")
    ds_out.attrs['regrid'] = regrid
    ds_out.attrs['missing_hours'] = int(missing.sum())

    print(f"Saving to {out_path}...")
    ds_out.to_netcdf(out_path, format='NETCDF4')

    # ── Clean up raw HDF5 files ───────────────────────────────────────────────
    print("Cleaning up raw granule files...")
    for f in files:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nDone! {len(expected)} hourly timesteps saved to {out_path} "
          f"({len(hourly_times)} with data, {int(missing.sum())} missing)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download IMERG Final and regrid to the ERA5 grid.')
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2013, 2026)),
                        help='years to download, Sep–Dec of each (default: 2013–2025)')
    parser.add_argument('--overwrite', action='store_true', help='replace existing output files')
    parser.add_argument('--regrid', choices=['conservative', 'bilinear'], default=REGRID,
                        help='regridding method (default: conservative)')
    args = parser.parse_args()

    # Authenticate once — credentials saved to ~/.netrc, no prompts on subsequent runs
    earthaccess.login(strategy="netrc")

    for year in args.years:
        time_range = slice(f'{year}-09-01', f'{year}-12-31')
        print(f"\n{'='*60}")
        print(f"Processing {time_range.start} → {time_range.stop}  (regrid: {args.regrid})")
        print('='*60)
        download_imerg(time_range=time_range, overwrite=args.overwrite, regrid=args.regrid)
