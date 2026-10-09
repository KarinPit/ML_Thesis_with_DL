"""
jones_figures.py
----------------
Clean evaluation figures for the Israel-domain U-Net.
Adapted from Jones et al. (2026) "Sample Figures Script" notebook.

Usage (on EC2):
    cd /home/ec2-user/ML_Thesis_with_DL
    MKL_THREADING_LAYER=GNU python jones_figures.py

Outputs saved to:  data/figures/
"""

import xarray as xr
import numpy as np
import pandas as pd
import os, gc, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.dates import DateFormatter, MonthLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter
from scipy.stats import linregress
from sklearn.metrics import r2_score
import scipy.stats as stats

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR = '/home/ec2-user/ML_Thesis_with_DL/data'
FIG_DIR  = '/home/ec2-user/ML_Thesis_with_DL/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load predictions (binary model, test set 2024-2025) ────────────────────────
print("Loading data...", flush=True)
ds_pred    = xr.open_dataset(os.path.join(DATA_DIR, 'unet_binary_OND_predictions.nc'))
prob_da    = ds_pred['prob']   # (T, lat, lon)  predicted probability 0–1
pred_da    = ds_pred['pred']   # (T, lat, lon)  binary prediction 0/1
test_times = ds_pred['time'].values
lats = ds_pred['lat'].values
lons = ds_pred['lon'].values
n_lat, n_lon = len(lats), len(lons)

# ── Load observations — resample hourly → 12-hourly binary ─────────────────────
ds_obs_h = xr.open_dataset(os.path.join(DATA_DIR, 'jones_israel_all_years.nc'))
ds_obs_h = ds_obs_h.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
obs_12h  = (ds_obs_h['ltg'].resample(time='12h').sum() > 0).astype(float)
obs_da   = obs_12h.sel(time=test_times, method='nearest')

# Land-sea mask
lsm_raw = ds_obs_h['lsm']
lsm_2d  = lsm_raw.isel(time=0).values[:n_lat, :n_lon] if 'time' in lsm_raw.dims \
          else lsm_raw.values[:n_lat, :n_lon]
land_mask = lsm_2d >= 0.5

print(f"Test period : {pd.Timestamp(test_times[0]).date()} → {pd.Timestamp(test_times[-1]).date()}")
print(f"Grid        : {n_lat}×{n_lon}, Timesteps: {len(test_times)}")
print(f"Lat range   : {lats.min():.2f} → {lats.max():.2f}")
print(f"Lon range   : {lons.min():.2f} → {lons.max():.2f}", flush=True)

prob_np = prob_da.values   # (T, lat, lon)  probability
pred_np = pred_da.values   # (T, lat, lon)  binary prediction
obs_np  = obs_da.values    # (T, lat, lon)  binary observation (0/1)

# ── Jones' FSS (verbatim) ───────────────────────────────────────────────────────
def compute_fss(obs, pred, threshold, window_size):
    obs_bin   = (obs  >= threshold).astype(int)
    pred_bin  = (pred >= threshold).astype(int)
    obs_frac  = uniform_filter(obs_bin.astype(float),  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin.astype(float), size=window_size, mode='constant', cval=0)
    num = np.sum((pred_frac - obs_frac) ** 2)
    den = np.sum(pred_frac**2 + obs_frac**2)
    return np.nan if den == 0 else 1 - num / den

THRESHOLD    = 0.5   # probability threshold for FSS
WINDOW_SIZE  = 3     # primary window (used for seasonal plot, metrics)
WINDOW_SIZES = [1, 3, 5]   # for kernel comparison plot

# FSS for multiple window sizes
print("Computing FSS for windows 1×1, 3×3, 5×5...", flush=True)
fss_by_window = {}
for w in WINDOW_SIZES:
    scores = [compute_fss(obs_np[i], prob_np[i], THRESHOLD, w)
              for i in range(len(test_times))]
    fss_by_window[w] = np.array(scores)
    valid = fss_by_window[w][np.isfinite(fss_by_window[w])]
    print(f"  {w}×{w}: mean={np.nanmean(valid):.4f}", flush=True)

# Primary window arrays (3×3) used by other figures
fss_arr   = fss_by_window[WINDOW_SIZE]
fss_valid = fss_arr[np.isfinite(fss_arr)]

# R² on lightning frequency (mean over time = fraction of timesteps with lightning)
obs_freq  = obs_np.mean(axis=0).ravel()
pred_freq = pred_np.mean(axis=0).ravel()
fin = np.isfinite(obs_freq) & np.isfinite(pred_freq)
r2_clim = r2_score(obs_freq[fin], pred_freq[fin])

print(f"Mean FSS (3×3) = {np.nanmean(fss_valid):.4f}   R²(freq) = {r2_clim:.4f}", flush=True)


# ── Helper: degree-formatted tick labels ────────────────────────────────────────
def fmt_deg(x, pos):
    return f"{x:.1f}°"

def add_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=9)
    return cb

def map_axes(ax):
    """Apply lat/lon formatting to a map axis."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)


# ============================================================
# Fig 1 — Observed and Predicted Lightning Climatology Maps
# ============================================================
print("Fig 1: Climatology maps...", flush=True)

obs_mean  = obs_np.mean(axis=0)
pred_mean = pred_np.mean(axis=0)

# Colour scale: mask zeros → transparent; blue (low) → red (high)
pos_vals  = obs_mean[obs_mean > 0]
vmax_clim = float(np.nanpercentile(pos_vals, 98)) if len(pos_vals) > 0 else 0.2
cmap_ltg  = plt.cm.RdYlBu_r   # blue=low, yellow=mid, red=high
cmap_ltg.set_bad(color='white')  # NaN (= zero cells) → white

# Try cartopy, then geopandas, then plain axes
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    HAS_GPD = False
    print("  Using cartopy for borders", flush=True)
except ImportError:
    HAS_CARTOPY = False
    try:
        import geopandas as gpd
        HAS_GPD = True
        print("  cartopy not found — using geopandas for borders", flush=True)
    except ImportError:
        HAS_GPD = False
        print("  Neither cartopy nor geopandas found — plain axes", flush=True)

lon_min, lon_max = float(lons.min()), float(lons.max())
lat_min, lat_max = float(lats.min()), float(lats.max())

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5),
                             subplot_kw={'projection': proj},
                             constrained_layout=True)
    for ax, data, title in zip(axs,
                                [obs_mean, pred_mean],
                                ['Observed (WWLN)', 'CPLRSTW U-Net']):
        data_masked = np.where(data > 0, data, np.nan)
        im = ax.pcolormesh(lons, lats, data_masked, cmap=cmap_ltg,
                           vmin=0, vmax=vmax_clim, shading='auto',
                           transform=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='gray')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, label=f'Lightning frequency\n(fraction of 12h steps)', shrink=0.85)
elif HAS_GPD:
    # Load Natural Earth borders via geopandas
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except Exception:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, data, title in zip(axs,
                                [obs_mean, pred_mean],
                                ['Observed (WWLN)', 'CPLRSTW U-Net']):
        data_masked = np.where(data > 0, data, np.nan)
        im = ax.pcolormesh(lons, lats, data_masked, cmap=cmap_ltg,
                           vmin=0, vmax=vmax_clim, shading='auto')
        world.boundary.plot(ax=ax, linewidth=0.6, color='black', zorder=2)
        add_colorbar(fig, ax, im, 'Lightning frequency\n(fraction of 12h steps)')
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_facecolor('#d0e8f0')   # light blue ocean background
        map_axes(ax)
        ax.set_title(title, fontsize=12)
else:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, data, title in zip(axs,
                                [obs_mean, pred_mean],
                                ['Observed (WWLN)', 'CPLRSTW U-Net']):
        data_masked = np.where(data > 0, data, np.nan)
        im = ax.pcolormesh(lons, lats, data_masked, cmap=cmap_ltg,
                           vmin=0, vmax=vmax_clim, shading='auto')
        add_colorbar(fig, ax, im, 'Lightning frequency\n(fraction of 12h steps)')
        ax.set_facecolor('white')
        map_axes(ax)
        ax.set_title(title, fontsize=12)

fig.suptitle(f'Mean Lightning Frequency — Test Period 2024–2025  (OND season)', fontsize=13)
fig.savefig(os.path.join(FIG_DIR, 'fig1_climatology_maps.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig1_climatology_maps.png", flush=True)


# ============================================================
# Fig 2 — FSS Time Series (monthly mean ± std, clean line)
# ============================================================
print("Fig 2: FSS time series...", flush=True)

time_index = pd.DatetimeIndex(test_times)
colors_w   = {1: 'steelblue', 3: 'darkorange', 5: 'forestgreen'}

fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)

for w in WINDOW_SIZES:
    series       = pd.Series(fss_by_window[w], index=time_index)
    monthly_mean = series.resample('MS').mean()
    mean_val     = np.nanmean(fss_by_window[w][np.isfinite(fss_by_window[w])])

    ax.plot(monthly_mean.index, monthly_mean.values,
            color=colors_w[w], lw=2.5, marker='o', markersize=6,
            label=f'{w}×{w}  (mean={mean_val:.3f})')

ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='FSS = 0.5 (skillful)')
ax.set_ylim(0, 1)
ax.set_xlabel('Date/Time', fontsize=11)
ax.set_ylabel('FSS', fontsize=11)
ax.set_title('FSS Kernel Size Comparison — Monthly Mean ± Std  (OND season, 2024–2025)', fontsize=12)
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(MonthLocator(interval=1))
ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
ax.grid(axis='y', ls='--', alpha=0.4)
fig.savefig(os.path.join(FIG_DIR, 'fig2_fss_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig2_fss_timeseries.png", flush=True)


# ============================================================
# Fig 3 — Difference Map (Modeled − Observed)
# ============================================================
print("Fig 3: Difference map...", flush=True)

diff = pred_mean - obs_mean
vmax_d = float(np.nanpercentile(np.abs(diff), 98))

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': proj},
                           constrained_layout=True)
    im = ax.pcolormesh(lons, lats, diff, cmap='bwr',
                       vmin=-vmax_d, vmax=vmax_d, shading='auto', transform=proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='gray')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    plt.colorbar(im, ax=ax, label='Predicted freq − Observed freq', shrink=0.85)
else:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    im = ax.pcolormesh(lons, lats, diff, cmap='bwr',
                       vmin=-vmax_d, vmax=vmax_d, shading='auto')
    if HAS_GPD:
        world.boundary.plot(ax=ax, linewidth=0.6, color='black', zorder=2)
        ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    add_colorbar(fig, ax, im, 'Predicted freq − Observed freq')
    map_axes(ax)

ax.set_title('Mean Frequency Bias: CPLRSTW U-Net − Observed  (2024–2025)', fontsize=12)
fig.savefig(os.path.join(FIG_DIR, 'fig3_difference_map.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig3_difference_map.png", flush=True)


# ============================================================
# Fig 4 — Scatter: Observed vs Predicted  (log-log density, Jones style)
# ============================================================
print("Fig 4: Scatter plot (log-log density)...", flush=True)

from scipy.stats import gaussian_kde

def density_scatter(ax, xd, yd, panel_label):
    """Log-log contourf density scatter matching Jones et al. style."""
    # Keep only pixels where both obs > 0 and pred > 0
    keep = (xd > 0) & (yd > 0) & np.isfinite(xd) & np.isfinite(yd)
    xd, yd = xd[keep], yd[keep]
    if len(xd) < 10:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return

    r2 = r2_score(np.log10(xd), np.log10(yd))   # R² in log space (matches Jones)

    # KDE in log space (subsample if large for speed)
    MAX_KDE = 5000
    if len(xd) > MAX_KDE:
        idx_s = np.random.choice(len(xd), MAX_KDE, replace=False)
        xk, yk = np.log10(xd[idx_s]), np.log10(yd[idx_s])
    else:
        xk, yk = np.log10(xd), np.log10(yd)

    kde = gaussian_kde(np.vstack([xk, yk]))

    # Evaluation grid in log space
    lo = min(xk.min(), yk.min()) - 0.3
    hi = max(xk.max(), yk.max()) + 0.3
    xi = np.linspace(lo, hi, 80)
    yi = np.linspace(lo, hi, 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # contourf with log-norm density
    lev_min = max(Zi[Zi > 0].min(), Zi.max() * 1e-3)
    levels = np.logspace(np.log10(lev_min), np.log10(Zi.max()), 14)
    cf = ax.contourf(10**Xi, 10**Yi, Zi, levels=levels,
                     cmap='rainbow', norm=mcolors.LogNorm())
    cb = plt.colorbar(cf, ax=ax)
    cb.set_label('Point Density', fontsize=9)

    # 1:1 line
    lim = [10**lo, 10**hi]
    ax.plot(lim, lim, 'k-', lw=1.5, label=f'1:1 Line (R² = {r2:.2f})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(10**lo, 10**hi); ax.set_ylim(10**lo, 10**hi)
    ax.set_xlabel('Observed Lightning', fontsize=10)
    ax.set_ylabel('Modeled Lightning',  fontsize=10)
    ax.legend(fontsize=9, loc='upper left')

datasets = [
    ('All',   obs_freq,
              pred_freq),
    ('Land',  np.where(land_mask.ravel(),  obs_freq, np.nan),
              np.where(land_mask.ravel(),  pred_freq, np.nan)),
    ('Ocean', np.where(~land_mask.ravel(), obs_freq, np.nan),
              np.where(~land_mask.ravel(), pred_freq, np.nan)),
]

fig, axs = plt.subplots(3, 1, figsize=(6, 16), constrained_layout=True)
for ax, (subset, xd, yd) in zip(axs, datasets):
    density_scatter(ax, xd, yd, panel_label='')
    ax.set_title(f'CPLRSTW U-Net: {subset}', fontsize=11, fontweight='bold')

fig.suptitle('Observed vs Modeled Lightning — Test 2024–2025', fontsize=13)
fig.savefig(os.path.join(FIG_DIR, 'fig4_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig4_scatter.png", flush=True)


# ============================================================
# Fig 5 — Reliability Diagram (calibration of predicted probability)
# ============================================================
print("Fig 5: Reliability diagram...", flush=True)

N_BINS = 20   # quantile bins — equal number of predictions per bin
prob_flat = prob_np.ravel()
obs_flat  = obs_np.ravel()
fin_mask  = np.isfinite(prob_flat) & np.isfinite(obs_flat)
prob_flat, obs_flat = prob_flat[fin_mask], obs_flat[fin_mask]

# Sort by predicted probability, then split into N_BINS equal-count quantile bins
sort_idx   = np.argsort(prob_flat)
prob_sorted = prob_flat[sort_idx]
obs_sorted  = obs_flat[sort_idx]
splits = np.array_split(np.arange(len(prob_sorted)), N_BINS)

bin_centers  = np.array([prob_sorted[s].mean() for s in splits])
bin_obs_freq = np.array([obs_sorted[s].mean()  for s in splits])
bin_counts   = np.array([len(s)                for s in splits])

bin_obs_freq = np.array(bin_obs_freq)
bin_counts   = np.array(bin_counts)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True,
                                gridspec_kw={'height_ratios': [3, 1]})

# Reliability curve
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
ax1.plot(bin_centers, bin_obs_freq, 'o-', color='steelblue', lw=2,
         markersize=7, label='U-Net')
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_ylabel('Observed lightning frequency', fontsize=11)
ax1.set_title('Reliability Diagram — Predicted Probability vs Observed Frequency\n(2024–2025 test set)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(ls='--', alpha=0.4)

# Count histogram
ax2.bar(bin_centers, bin_counts, width=0.09, color='steelblue', alpha=0.7)
ax2.set_xlabel('Predicted probability', fontsize=11)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_xlim(0, 1)
ax2.grid(axis='y', ls='--', alpha=0.4)

fig.savefig(os.path.join(FIG_DIR, 'fig5_reliability.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig5_reliability.png", flush=True)


# ============================================================
# Fig 6 — Seasonal Breakdown (FSS by month)
# ============================================================
print("Fig 6: Seasonal FSS...", flush=True)

months = pd.DatetimeIndex(test_times).month
month_fss = {}
for m in range(1, 13):
    idx = np.where(months == m)[0]
    if len(idx) == 0:
        continue
    scores = [compute_fss(obs_np[i], prob_np[i], THRESHOLD, WINDOW_SIZE) for i in idx]
    valid  = [s for s in scores if np.isfinite(s)]
    if valid:
        month_fss[m] = (np.mean(valid), np.std(valid) / np.sqrt(len(valid)))

month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ms = sorted(month_fss.keys())
mv = [month_fss[m][0] for m in ms]
me = [month_fss[m][1] for m in ms]

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
ax.bar([month_labels[m-1] for m in ms], mv, yerr=me, capsize=5,
       color='teal', alpha=0.85)
ax.set_ylabel('Mean FSS', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
ax.set_title('Monthly Mean FSS  (3×3 window, 2024–2025)', fontsize=12)
ax.set_ylim(0, 1)
ax.axhline(np.nanmean(fss_valid), color='crimson', ls='--', lw=1.5,
           label=f'Overall mean = {np.nanmean(fss_valid):.3f}')
ax.legend(fontsize=10)
ax.grid(axis='y', ls='--', alpha=0.5)
fig.savefig(os.path.join(FIG_DIR, 'fig6_seasonal_fss.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig6_seasonal_fss.png", flush=True)


# ============================================================
# Fig 7 — Per-Month Maps: Observed vs Predicted (OND)
# ============================================================
print("Fig 7: Monthly obs vs predicted maps...", flush=True)

OND_MONTHS      = [10, 11, 12]
OND_LABELS      = ['October', 'November', 'December']
months_arr      = pd.DatetimeIndex(test_times).month

# Shared colour scale: use 98th percentile of the per-month mean values (not raw binary)
month_means_all = np.stack([
    obs_np[np.where(pd.DatetimeIndex(test_times).month == m)[0]].mean(axis=0)
    for m in OND_MONTHS
])
pos_means = month_means_all[month_means_all > 0]
vmax_m = float(np.nanpercentile(pos_means, 98)) if len(pos_means) > 0 else 0.1
print(f"  Monthly maps vmax = {vmax_m:.4f}", flush=True)

nrows = len(OND_MONTHS)
if HAS_CARTOPY:
    proj  = ccrs.PlateCarree()
    fig, axs = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows),
                             subplot_kw={'projection': proj},
                             constrained_layout=True)
else:
    fig, axs = plt.subplots(nrows, 2, figsize=(13, 4 * nrows),
                             constrained_layout=True)

for row, (m, mlabel) in enumerate(zip(OND_MONTHS, OND_LABELS)):
    idx_m = np.where(months_arr == m)[0]
    obs_m  = np.where(obs_np[idx_m].mean(axis=0)  > 0, obs_np[idx_m].mean(axis=0),  np.nan)
    pred_m = np.where(pred_np[idx_m].mean(axis=0) > 0, pred_np[idx_m].mean(axis=0), np.nan)

    for col, (data, panel_title) in enumerate(zip(
            [obs_m, pred_m],
            [f'{mlabel} — Observed (WWLN)', f'{mlabel} — CPLRSTW U-Net'])):

        ax = axs[row, col]
        if HAS_CARTOPY:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_ltg,
                               vmin=0, vmax=vmax_m, shading='auto', transform=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
            ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='gray')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        else:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap_ltg,
                               vmin=0, vmax=vmax_m, shading='auto')
            if HAS_GPD:
                world.boundary.plot(ax=ax, linewidth=0.6, color='black', zorder=2)
                ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
            ax.set_facecolor('#d0e8f0')
            map_axes(ax)

        ax.set_title(panel_title, fontsize=11)
        plt.colorbar(im, ax=ax, label='Lightning frequency', shrink=0.85)

fig.suptitle('Monthly Lightning Frequency: Observed vs Predicted  (Test 2024–2025)', fontsize=13)
fig.savefig(os.path.join(FIG_DIR, 'fig7_monthly_maps.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig7_monthly_maps.png", flush=True)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*55}")
print(f"  TEST SET SUMMARY")
print(f"{'='*55}")
print(f"  Period        : {pd.Timestamp(test_times[0]).date()} → {pd.Timestamp(test_times[-1]).date()}")
print(f"  Timesteps     : {len(test_times)} × 12h")
print(f"  Mean FSS      : {np.nanmean(fss_valid):.4f}")
print(f"  Median FSS    : {np.nanmedian(fss_valid):.4f}")
print(f"  R² (clim)     : {r2_clim:.4f}")
print(f"{'='*55}")
print(f"\nFigures saved → {FIG_DIR}", flush=True)
