import numpy as np
import xarray as xr
import re


def calculate_proxy_lpi(ds):
    """Calculate proxy Lightning Potential Index (LPI) from ERA5 pressure-level data.

    Adapted from Yair et al. (2010).
    Substitution: specific_cloud_liquid_water_content (clwc) is used in place of
    specific_snow_water_content, as supercooled liquid water droplets are the key
    participant in non-inductive charge separation alongside ice crystals.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain:
        - 'temperature'                       (K)         — pressure levels
        - 'geopotential'                      (m2/s2)     — pressure levels
        - 'vertical_velocity'                 (Pa/s)      — pressure levels
        - 'specific_cloud_ice_water_content'  (kg/kg)     — pressure levels
        - 'specific_cloud_liquid_water_content' (kg/kg)   — pressure levels
        - 'convective_available_potential_energy' (J/kg)  — single level

    Returns
    -------
    xr.DataArray : proxy LPI (time × latitude × longitude)
    """
    g  = 9.80665  # gravity (m/s2)
    Rd = 287.05   # gas constant for dry air (J/kg/K)

    # geopotential → geopotential height (m)
    Z = ds['geopotential'] / g

    # pressure levels: hPa → Pa
    P_pa = ds['level'] * 100.0

    # air density: ρ = P / (Rd × T)
    rho = P_pa / (Rd * ds['temperature'])

    # vertical velocity: Pa/s (omega) → m/s (w)
    # negative omega = upward motion, so multiply by -1
    W = -ds['vertical_velocity'] / (rho * g)

    # charging zone mask: 0°C to −20°C  (273.15 K to 253.15 K)
    in_charging_zone = (ds['temperature'] <= 273.15) & (ds['temperature'] >= 253.15)

    # charging layer depth ΔZ
    Z_masked = Z.where(in_charging_zone)
    Z_0      = Z_masked.max(dim='level')   # highest altitude where T ≈ 0°C
    Z_20     = Z_masked.min(dim='level')   # lowest altitude where T ≈ −20°C
    delta_Z  = Z_0 - Z_20
    delta_Z  = delta_Z.where(delta_Z > 0)  # avoid division by zero

    # microphysical charging efficiency term ε
    # Uses cloud ice (qi) and supercooled liquid water (qs = clwc)
    qi = ds['specific_cloud_ice_water_content'].where(
        ds['specific_cloud_ice_water_content'] > 0, 0.0)
    qs = ds['specific_cloud_liquid_water_content'].where(
        ds['specific_cloud_liquid_water_content'] > 0, 0.0)

    numerator   = qi * qs
    denominator = (np.sqrt(qi) + np.sqrt(qs)) ** 2
    micro_term  = xr.where(denominator > 0, numerator / denominator, 0.0)

    # updraft only (w > 0); lightning requires upward motion
    w_up = W.where(W > 0, 0.0)

    # integrand = w × ε, masked to charging zone only
    integrand_zone = (w_up * micro_term).where(in_charging_zone, 0.0)

    # vertical integration over pressure levels (trapezoid rule)
    integral = integrand_zone.integrate(coord='level')

    # normalize by charging layer depth
    proxy_lpi = integral / delta_Z

    # fill NaNs (no charging zone) with 0
    proxy_lpi = proxy_lpi.fillna(0.0)

    # CAPE mask: ignore non-convective environments
    proxy_lpi = proxy_lpi.where(ds['convective_available_potential_energy'] >= 100.0, 0.0)

    return proxy_lpi


def compute_and_save_lpi(pressure_path, single_path, out_dir='data'):
    """Load ERA5 NetCDF files, compute proxy LPI, and save as NetCDF.

    Parameters
    ----------
    pressure_path : str  path to era5_pressure_level_{ts}.nc
    single_path   : str  path to era5_single_level_{ts}.nc  (for CAPE)
    out_dir       : str  output directory
    """
    print(f"Loading pressure-level data from {pressure_path}...")
    ds_pressure = xr.open_dataset(pressure_path, chunks={'time': 100})

    print(f"Loading single-level data from {single_path}...")
    ds_single = xr.open_dataset(single_path,  chunks={'time': 100})

    # merge CAPE into pressure dataset so calculate_proxy_lpi sees everything in one ds
    ds = ds_pressure.assign(
        convective_available_potential_energy=ds_single['convective_available_potential_energy']
    )

    print("Computing proxy LPI...")
    lpi = calculate_proxy_lpi(ds)
    lpi.name = 'proxy_lpi'

    # derive timestamp from filename years
    match = re.search(r'(\d{4}(?:_\d{4})?)', pressure_path)
    ts = match.group(1) if match else 'unknown'

    out_path = f'{out_dir}/proxy_lpi_{ts}.nc'
    lpi.to_netcdf(out_path)
    print(f"Proxy LPI saved to {out_path}")

    return out_path

def validate_values(file_path):
    lpi = xr.open_dataset(file_path)['proxy_lpi']
    print("Min:", float(lpi.min()))
    print("Max:", float(lpi.max()))
    print("Mean:", float(lpi.mean()))
    print("% zeros:", float((lpi == 0).mean()) * 100)
    print("% NaN:", float(np.isnan(lpi).mean()) * 100)


if __name__ == "__main__":
    year='2024'
    out_path = compute_and_save_lpi(
        pressure_path=f'data/era5_pressure_level_{year}.nc',
        single_path=f'data/era5_single_level_{year}.nc',
    )

    # validate_values(out_path)
