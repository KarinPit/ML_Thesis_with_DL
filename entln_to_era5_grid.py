import pandas as pd
import numpy as np
import xarray as xr

def build_lightning_grid():
    # load ENTLN lightning data
    df = pd.read_excel('data/ENTLN 2025-2026 season.xlsx')
    df['UTC'] = pd.to_datetime(df['UTC'])

    # load ERA5 grid coordinates
    ds_single = xr.open_dataset('data/era5_single_level_sample.nc')
    era5_lats = ds_single.latitude.values
    era5_lons = ds_single.longitude.values
    era5_hours = pd.to_datetime(ds_single.time.values)

    # build bin edges (histogram2d needs edges, not centers)
    lat_step = lon_step = 0.25
    lat_edges = np.sort(np.append(era5_lats + lat_step / 2,
                                  era5_lats[-1] - lat_step / 2))  # ascending
    lon_edges = np.append(era5_lons - lon_step / 2,
                          era5_lons[-1] + lon_step / 2)

    # filter lightning to domain and time range
    LAT_MIN, LAT_MAX = era5_lats[-1], era5_lats[0]
    LON_MIN, LON_MAX = era5_lons[0], era5_lons[-1]
    TIME_MIN = era5_hours.min()
    TIME_MAX = era5_hours.max() + pd.Timedelta(hours=1)

    df = df[
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
        (df['UTC'] >= TIME_MIN) & (df['UTC'] < TIME_MAX)
    ].copy()
    print(f"Lightning events in domain and time range: {len(df)}")

    # bin into per-hour 2D count maps
    df['hour'] = df['UTC'].dt.floor('h')
    n_lat, n_lon = len(era5_lats), len(era5_lons)
    counts_hourly = np.zeros((len(era5_hours), n_lat, n_lon), dtype=np.int32)

    for i, hour in enumerate(era5_hours):
        df_hour = df[df['hour'] == hour]
        if len(df_hour) == 0:
            continue
        counts_h, _, _ = np.histogram2d(
            df_hour['lon'], df_hour['lat'],
            bins=[lon_edges, lat_edges]
        )
        counts_hourly[i] = counts_h.T[::-1]  # transpose + flip to match ERA5 N→S lat order
        print(f"  {hour}: {int(counts_h.sum())} strikes")

    # wrap into an xarray DataArray co-registered to the ERA5 grid
    lightning_da = xr.DataArray(
        counts_hourly,
        coords={'time': era5_hours, 'latitude': era5_lats, 'longitude': era5_lons},
        dims=['time', 'latitude', 'longitude'],
        name='lightning_count',
        attrs={'units': 'count per ERA5 cell per hour', 'source': 'ENTLN'}
    )

    lightning_da.to_netcdf('data/entln_on_era5_grid.nc')
    print("\nSaved to data/entln_on_era5_grid.nc")

if __name__ == "__main__":
    build_lightning_grid()
