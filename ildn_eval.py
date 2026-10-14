"""
ildn_eval.py — how well does the WWLLN-trained U-Net predict lightning seen by the Israeli network (ILDN)?

The model is NOT retrained: its predictions (made from ERA5 + IMERG) are compared with ILDN strokes
over a box around Israel. ILDN is a dense local network, so it detects many more strokes than WWLLN;
absolute counts therefore don't match, and the scores focus on WHERE and WHEN lightning happens.
WWLLN itself is scored against ILDN as the reference: the model learned to predict WWLLN, so
"WWLLN vs ILDN" is roughly the best it can be expected to do.

Years: only years the model never trained or validated on — 2024 (test year) and 2023 (excluded;
train_unet.py writes its predictions when EXTRA_PRED_YEARS = [2023]). 2025 has no IMERG for Oct–Dec.

Usage (on EC2, from the repo folder, after train_unet.py):
    python ildn_eval.py                                   # the run currently set in train_unet.py
    python ildn_eval.py unet_12h_poisson_OND_predictions.nc

Needs the ILDN files in the data folder:   data/<year>_for_yoav_including_cloud.txt
Outputs: figures/<run folder>/ildn/  (same run folder as jones_figures.py) — figures + ildn_summary.json
"""
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from run_info import default_pred_file, run_folder, run_label, _load

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/ec2-user/ML_Thesis_with_DL/data'
FIG_ROOT   = '/home/ec2-user/ML_Thesis_with_DL/figures'
ILDN_GLOB  = os.path.join(DATA_DIR, '*_for_yoav_including_cloud.txt')   # one file per year
HOURLY_PATH = os.path.join(DATA_DIR, 'jones_israel_all_years.nc')        # hourly WWLLN, for the time check

BOX        = (30.0, 34.5, 32.0, 36.5)   # lat_min, lat_max, lon_min, lon_max: Israel + nearby sea and land
ILDN_TYPES = 'all'                      # 'all' | 'CG' (ground strokes only) | 'IC' (cloud strokes only)
ILDN_MIN_KA = 0.0                       # drop strokes weaker than this |peak current| (kA); 0 = keep all
ILDN_TIME_OFFSET_H = 0                  # hours to SUBTRACT from ILDN times (2 if the file were Israel winter time)
ACTIVE_MIN = 1                         # an "active" window has ≥ this many ILDN strokes in the box
N_CASES    = 4                          # case-study windows (most ILDN strokes)

# colours (validated categorical slots 1–2; observations in ink)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_ILDN, C_WWLLN, C_UNET = INK, '#2a78d6', '#eb6834'
SEQ = mcolors.LinearSegmentedColormap.from_list(
    'blue_seq', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
SEQ.set_bad('#ffffff')

# ── Predictions: test years + extra held-out years ────────────────────────────
args = [a for a in sys.argv[1:] if not a.startswith('--')]
PRED_FILE = args[0] if args else default_pred_file()
PRED_PATH = PRED_FILE if os.path.isabs(PRED_FILE) else os.path.join(DATA_DIR, PRED_FILE)
if not os.path.exists(PRED_PATH):
    sys.exit(f"{PRED_PATH} not found — run train_unet.py first (or pass a predictions file)")
EXTRA_PATH = PRED_PATH.replace('_OND_predictions.nc', '_OND_predictions_extra.nc')

pred_main = xr.open_dataset(PRED_PATH).load()
A = {k: _load(v) for k, v in pred_main.attrs.items()}
AGG    = int(A.get('agg_hours', 12))
FSS_W  = int(A.get('fss_window', 3))
MONTHS = list(A.get('months', [10, 11, 12]))
parts  = [pred_main]
if os.path.exists(EXTRA_PATH):
    pred_extra = xr.open_dataset(EXTRA_PATH).load()
    if pred_extra.attrs.get('ckpt_trained_at') != pred_main.attrs.get('ckpt_trained_at'):
        sys.exit(f"{os.path.basename(EXTRA_PATH)} comes from a different training run than "
                 f"{os.path.basename(PRED_PATH)} — rerun train_unet.py (RETRAIN = False is enough).")
    parts.append(pred_extra)
else:
    print(f"NOTE: {os.path.basename(EXTRA_PATH)} not found — only test years can be scored. Set "
          f"EXTRA_PRED_YEARS = [2023] in train_unet.py and rerun it (RETRAIN = False is enough).", flush=True)
keep_vars = ['pred_count', 'obs_count']
pred = xr.concat([p[keep_vars] for p in parts], dim='time').sortby('time')
pred_years = sorted(set(pd.DatetimeIndex(pred.time.values).year))

RUN_NAME  = run_folder(pred_main.attrs, PRED_PATH)
RUN_LABEL = run_label(pred_main.attrs)
OUT_DIR   = os.path.join(FIG_ROOT, RUN_NAME, 'ildn')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run: {RUN_LABEL}\n  predictions for {pred_years}\n  output → {OUT_DIR}", flush=True)

lats, lons = pred.lat.values.astype(float), pred.lon.values.astype(float)
dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]

# ── ILDN strokes ──────────────────────────────────────────────────────────────
COLS = ['date', 'time', 'lat', 'lon', 'peak_ka', 'c6', 'n_sensors', 'c8', 'c9', 'type', 'c11', 'c12', 'c13']

def read_ildn(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLS, usecols=['date', 'time', 'lat', 'lon',
                                                                         'peak_ka', 'type'])
    df['t'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    bad = int(df['t'].isna().sum())
    if bad:
        print(f"  {os.path.basename(path)}: {bad} rows with unreadable time skipped", flush=True)
    return df.dropna(subset=['t']).drop(columns=['date', 'time'])

files = sorted(glob.glob(ILDN_GLOB))
if not files:
    sys.exit(f"No ILDN files match {ILDN_GLOB} — copy the <year>_for_yoav_including_cloud.txt files there")
ildn = pd.concat([read_ildn(f) for f in files], ignore_index=True)
ildn['t'] = ildn['t'] - pd.Timedelta(hours=ILDN_TIME_OFFSET_H)
n_all = len(ildn)
ildn = ildn[ildn.t.dt.month.isin(MONTHS)]
if ILDN_TYPES == 'CG':
    ildn = ildn[ildn['type'] == 'G']
elif ILDN_TYPES == 'IC':
    ildn = ildn[ildn['type'] == 'C']
ildn = ildn[ildn['peak_ka'].abs() >= ILDN_MIN_KA]
ildn_years = sorted(set(ildn.t.dt.year))
print(f"ILDN: {n_all:,} strokes in {len(files)} files → {len(ildn):,} in months {MONTHS} "
      f"(types: {ILDN_TYPES}, |I| ≥ {ILDN_MIN_KA:g} kA), years {ildn_years}", flush=True)

YEARS = [y for y in ildn_years if y in pred_years]
for y in ildn_years:
    if y not in pred_years:
        print(f"  {y}: ILDN data but no model predictions (no complete inputs, or not in "
              f"TEST_YEARS / EXTRA_PRED_YEARS) — skipped", flush=True)
if not YEARS:
    sys.exit("No year has both ILDN data and model predictions.")
_pt = pd.DatetimeIndex(pred.time.values)
pred = pred.sel(time=_pt.year.isin(YEARS) & _pt.month.isin(MONTHS))
times = pd.DatetimeIndex(pred.time.values)

# snap to the nearest grid point (same rule as process_wwln.py) and to the model's windows
ildn['i'] = np.round((ildn['lat'] - lats[0]) / dlat).astype(int)
ildn['j'] = np.round((ildn['lon'] - lons[0]) / dlon).astype(int)
ildn = ildn[(ildn.i >= 0) & (ildn.i < len(lats)) & (ildn.j >= 0) & (ildn.j < len(lons))]
season0 = pd.to_datetime(ildn.t.dt.year.astype(str) + f'-{min(MONTHS):02d}-01')
step = pd.Timedelta(hours=AGG)
ildn['win'] = season0 + ((ildn.t - season0) // step) * step
ildn['k'] = times.get_indexer(ildn['win'])
n_outside = int((ildn['k'] < 0).sum())
ildn_w = ildn[ildn.k >= 0]
I_full = np.zeros((len(times), len(lats), len(lons)), dtype='float32')
np.add.at(I_full, (ildn_w.k.values, ildn_w.i.values, ildn_w.j.values), 1)
print(f"  gridded {len(ildn_w):,} strokes onto {len(times)} × {AGG}-h windows "
      f"({n_outside:,} fell in windows the model has no prediction for — incomplete inputs)", flush=True)

# ── Box around Israel ─────────────────────────────────────────────────────────
bi = np.where((lats >= BOX[0]) & (lats <= BOX[1]))[0]
bj = np.where((lons >= BOX[2]) & (lons <= BOX[3]))[0]
blat, blon = lats[bi], lons[bj]
crop = lambda a: a[:, bi][:, :, bj]
I = crop(I_full)
fields = {'U-Net': crop(np.nan_to_num(pred['pred_count'].values)),
          'WWLLN': crop(np.nan_to_num(pred['obs_count'].values))}
print(f"Box lat {blat.min():.2f}–{blat.max():.2f}, lon {blon.min():.2f}–{blon.max():.2f} "
      f"({len(bi)}×{len(bj)} cells); ILDN strokes in box: {int(I.sum()):,}", flush=True)

# baselines from the aggregated cache (if it has the inputs): R14 = CAPE × precipitation (scale-free
# scores don't need its constant) and the training-years monthly climatology of WWLLN
PREP_PATH = os.path.join(DATA_DIR, f'jones_israel_{AGG}h_OND.nc')
lsm_box = None
if os.path.exists(PREP_PATH):
    cache = xr.open_dataset(PREP_PATH)
    cbox = cache.sel(lat=blat, lon=blon, method='nearest')
    if 'lsm' in cbox:
        l = cbox['lsm']
        lsm_box = (l.isel(time=0) if 'time' in l.dims else l).values
    tr_years = A.get('train_years')
    if 'cape' in cbox and 'precipitation' in cbox and isinstance(tr_years, list) and 'ltg' in cache:
        # R14: strokes = c · CAPE · P, with c fitted on the training years over the whole domain
        # (same as jones_figures.py), so it is in WWLLN-equivalent strokes like the U-Net
        cfull = cache.sel(lat=lats, lon=lons, method='nearest')
        cfull = cfull.sel(time=cfull.time.dt.year.isin(tr_years))
        cp_tr = (cfull['cape'].clip(min=0) * cfull['precipitation'].clip(min=0)).fillna(0)
        c_r14 = float(cfull['ltg'].fillna(0).sum()) / max(float(cp_tr.sum()), 1e-12)
        cp = (cbox['cape'].clip(min=0) * cbox['precipitation'].clip(min=0)).sel(time=times, method='nearest')
        fields['R14 (CAPE×P)'] = c_r14 * np.nan_to_num(cp.values)
    if isinstance(tr_years, list) and 'ltg' in cbox:
        ltr = cbox['ltg'].sel(time=cbox.time.dt.year.isin(tr_years)).fillna(0)
        clim = ltr.groupby('time.month').mean('time')
        fields['Climatology'] = np.stack([clim.sel(month=m).values for m in times.month]).astype('float32')
else:
    print(f"  {PREP_PATH} not found — R14 and climatology baselines skipped", flush=True)

# ── Time-zone check: hourly ILDN vs hourly WWLLN in the box at different lags ──
lag_res = None
if os.path.exists(HOURLY_PATH):
    h = xr.open_dataset(HOURLY_PATH)['ltg']
    hb = h.isel(lat=np.where((h.lat >= BOX[0] - 1e-6) & (h.lat <= BOX[1] + 1e-6))[0],
                lon=np.where((h.lon >= BOX[2] - 1e-6) & (h.lon <= BOX[3] + 1e-6))[0])
    hb = hb.sel(time=hb.time.dt.year.isin(YEARS) & hb.time.dt.month.isin(MONTHS))
    w_h = hb.sum(['lat', 'lon'], min_count=1).to_series()
    in_box = ildn[(ildn.lat >= BOX[0]) & (ildn.lat <= BOX[1]) & (ildn.lon >= BOX[2]) & (ildn.lon <= BOX[3])]
    i_h = in_box.groupby(in_box.t.dt.floor('h')).size().reindex(w_h.index, fill_value=0)
    lags = np.arange(-6, 7)
    cors = []
    for L in lags:                                   # corr( WWLLN(t), ILDN(t + L) )
        a, b = w_h.values, i_h.shift(-L).values
        ok = np.isfinite(a) & np.isfinite(b)
        cors.append(pearsonr(np.log1p(a[ok]), np.log1p(b[ok]))[0] if ok.sum() > 10 else np.nan)
    cors = np.array(cors)
    best = int(lags[np.nanargmax(cors)])
    lag_res = {'lags_h': lags.tolist(), 'corr': np.round(cors, 3).tolist(), 'best_lag_h': best}
    r0 = float(cors[lags == 0][0])
    if np.nanmax(cors) < 0.3:
        print(f"WARNING time check: hourly ILDN and WWLLN in the box hardly correlate at any lag "
              f"(best r = {np.nanmax(cors):.2f}) — check the ILDN files and the box", flush=True)
    elif best == 0 or np.nanmax(cors) < r0 + 0.03:
        print(f"Time check: ILDN and WWLLN line up at lag 0 h (r = {r0:.2f}) — times agree", flush=True)
    else:
        print(f"WARNING time check: ILDN lines up best with WWLLN when shifted by {best:+d} h "
              f"(r = {np.nanmax(cors):.2f} vs {cors[lags == 0][0]:.2f} at 0 h). If ILDN is in local time, set "
              f"ILDN_TIME_OFFSET_H = {ILDN_TIME_OFFSET_H + best} and rerun.", flush=True)
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.plot(lags, cors, color=C_WWLLN, lw=2, marker='o', ms=6)
    ax.axvline(0, color=INK2, lw=1, ls=':')
    ax.set_xlabel('lag L (h): correlation of WWLLN(t) with ILDN(t + L)', color=INK2)
    ax.set_ylabel('r (hourly box totals, log1p)', color=INK2)
    ax.set_title(f'Time alignment ILDN vs WWLLN — best lag {best:+d} h', color=INK, fontsize=11)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT_DIR, 'ildn0_time_alignment.png'), dpi=150); plt.close(fig)
else:
    print(f"  {HOURLY_PATH} not found — time-zone check skipped", flush=True)

# ── Scores ─────────────────────────────────────────────────────────────────────
def fss_mean(obs_bin, est_bin):
    vals = []
    for o, p in zip(obs_bin, est_bin):
        of = uniform_filter(o.astype(float), FSS_W, mode='constant')
        pf = uniform_filter(p.astype(float), FSS_W, mode='constant')
        den = np.sum(of ** 2 + pf ** 2)
        if den > 0:
            vals.append(1 - np.sum((of - pf) ** 2) / den)
    return float(np.mean(vals)) if vals else np.nan

def scores(Iobs, est):
    tot_i, tot_e = Iobs.sum(axis=(1, 2)), est.sum(axis=(1, 2))
    active = tot_i >= ACTIVE_MIN
    obs_bin = Iobs >= 1
    f_obs = obs_bin.mean()
    # frequency-matched threshold: the estimate gets the same fraction of "lightning" cells as ILDN
    thr = np.quantile(est, 1 - f_obs) if 0 < f_obs < 1 else np.inf
    est_fm = est >= thr if thr > 0 else est > 0
    ci, ce = Iobs.mean(0).ravel(), est.mean(0).ravel()
    return {
        'windows': int(len(tot_i)), 'active_windows': int(active.sum()),
        'spearman_box_total': float(spearmanr(tot_e, tot_i)[0]),
        'auc_active_window': float(roc_auc_score(active, tot_e)) if 0 < active.sum() < len(active) else np.nan,
        'fss_freq_matched': fss_mean(obs_bin, est_fm),
        'fss_ge1': fss_mean(obs_bin, est >= 1),
        'map_spearman': float(spearmanr(ce, ci)[0]),
        'ratio_to_ildn': float(tot_e.sum() / max(tot_i.sum(), 1e-9)),
    }

yr = times.year.values
groups = {str(y): yr == y for y in YEARS}
if len(YEARS) > 1:
    groups['all'] = np.ones(len(times), bool)
results = {g: {name: scores(I[m], f[m]) for name, f in fields.items()} for g, m in groups.items()}
for g in groups:
    m = groups[g]
    results[g]['_ildn'] = {'strokes_in_box': int(I[m].sum()), 'windows': int(m.sum()),
                           'active_windows': int((I[m].sum(axis=(1, 2)) >= ACTIVE_MIN).sum()),
                           'cell_event_rate': float((I[m] >= 1).mean())}

LABELS = {'spearman_box_total': 'Spearman r, box total per window (when)',
          'auc_active_window':  f'AUC: active window (≥{ACTIVE_MIN} ILDN strokes) vs not',
          'fss_freq_matched':   f'FSS {FSS_W}×{FSS_W}, frequency-matched (where)',
          'fss_ge1':            f'FSS {FSS_W}×{FSS_W}, ≥1 stroke in both',
          'map_spearman':       'Spearman r, mean map (climatological where)',
          'ratio_to_ildn':      'total / ILDN total'}
order = [k for k in ['U-Net', 'R14 (CAPE×P)', 'Climatology', 'WWLLN'] if k in fields]
for g in groups:
    r = results[g]
    print(f"\n── {g}: {r['_ildn']['strokes_in_box']:,} ILDN strokes, {r['_ildn']['windows']} windows "
          f"({r['_ildn']['active_windows']} active) ──")
    print(f"  {'':44s}" + ''.join(f'{k:>14s}' for k in order))
    for key, lab in LABELS.items():
        print(f"  {lab:44s}" + ''.join(f"{r[k][key]:14.3f}" for k in order))

# ── Figures ─────────────────────────────────────────────────────────────────────
ext = [blon.min() - abs(dlon) / 2, blon.max() + abs(dlon) / 2, blat.min() - abs(dlat) / 2, blat.max() + abs(dlat) / 2]

def style(ax):
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)

def draw(ax, field, norm, title):
    im = ax.pcolormesh(blon, blat, np.ma.masked_less_equal(field, 0), cmap=SEQ, norm=norm, shading='nearest')
    if lsm_box is not None:
        ax.contour(blon, blat, lsm_box, levels=[0.5], colors=INK2, linewidths=0.8)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, color=INK)
    style(ax)
    return im

def stamp(fig):
    fig.text(0.995, -0.01, f"{RUN_NAME} · ILDN {ILDN_TYPES}, box {BOX}", ha='right', va='top',
             fontsize=6.5, color='0.45')

ylab = ', '.join(map(str, YEARS))
# Fig 1: mean maps, each divided by its own box mean (patterns comparable despite different detection)
map_names = ['ILDN'] + [k for k in ['WWLLN', 'U-Net', 'R14 (CAPE×P)'] if k in fields]
means = {'ILDN': I.mean(0), **{k: fields[k].mean(0) for k in map_names[1:]}}
rel = {k: v / max(v.mean(), 1e-12) for k, v in means.items()}
pos = np.concatenate([v[v > 0] for v in rel.values()])
norm1 = mcolors.LogNorm(vmin=max(np.percentile(pos, 2), 1e-2), vmax=np.percentile(pos, 99.5))
fig, axs = plt.subplots(1, len(map_names), figsize=(3.4 * len(map_names) + 0.8, 3.9), constrained_layout=True)
for ax, k in zip(np.atleast_1d(axs), map_names):
    r = results['all' if 'all' in results else str(YEARS[0])]
    sub = '' if k == 'ILDN' else f"\nmap r = {r[k]['map_spearman']:.2f}"
    im = draw(ax, rel[k], norm1, f'{k}{sub}')
cb = fig.colorbar(im, ax=axs, shrink=0.85, extend='both')
cb.set_label('mean ÷ box mean (1 = average)', color=INK2, fontsize=9)
fig.suptitle(f'Where: mean lightning, Oct–Dec {ylab} — {RUN_LABEL}', color=INK, fontsize=11)
stamp(fig)
fig.savefig(os.path.join(OUT_DIR, 'ildn1_mean_maps.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# Fig 2: box total per window through time (one log axis; all three are strokes per window)
fig, axs = plt.subplots(len(YEARS), 1, figsize=(11, 3.0 * len(YEARS)), squeeze=False, constrained_layout=True)
for ax, y in zip(axs[:, 0], YEARS):
    m = yr == y
    tt = times[m]
    for name, series, col in [('ILDN', I[m].sum(axis=(1, 2)), C_ILDN),
                              ('WWLLN', fields['WWLLN'][m].sum(axis=(1, 2)), C_WWLLN),
                              ('U-Net', fields['U-Net'][m].sum(axis=(1, 2)), C_UNET)]:
        ax.plot(tt, series + 1, color=col, lw=2 if name != 'ILDN' else 1.6, label=name)
    ax.set_yscale('log')
    ax.set_ylabel(f'strokes / {AGG} h in box (+1)', color=INK2, fontsize=9)
    r = results[str(y)]
    ax.set_title(f"{y}: Spearman r with ILDN — U-Net {r['U-Net']['spearman_box_total']:.2f}, "
                 f"WWLLN {r['WWLLN']['spearman_box_total']:.2f}", fontsize=10, color=INK, loc='left')
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); style(ax)
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)
fig.suptitle(f'When: lightning in the Israel box per {AGG}-h window — {RUN_LABEL}', color=INK, fontsize=11)
stamp(fig)
fig.savefig(os.path.join(OUT_DIR, 'ildn2_timeseries.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# Fig 3: per-window box totals, estimate vs ILDN
tot_i = I.sum(axis=(1, 2))
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
for ax, (name, col) in zip(axs, [('U-Net', C_UNET), ('WWLLN', C_WWLLN)]):
    te = fields[name].sum(axis=(1, 2))
    for y, mk in zip(YEARS, ['o', 's', '^']):
        m = yr == y
        ax.scatter(tot_i[m] + 1, te[m] + 1, s=26, marker=mk, facecolor=col, edgecolor='white', lw=0.6,
                   alpha=0.85, label=str(y))
    ax.set_xscale('log'); ax.set_yscale('log')
    rr = results['all' if 'all' in results else str(YEARS[0])][name]
    ax.set_title(f"{name} vs ILDN — Spearman r = {rr['spearman_box_total']:.2f}, "
                 f"AUC = {rr['auc_active_window']:.2f}", fontsize=10, color=INK)
    ax.set_xlabel(f'ILDN strokes / {AGG} h in box (+1)', color=INK2, fontsize=9)
    ax.set_ylabel(f'{name} strokes / {AGG} h in box (+1)', color=INK2, fontsize=9)
    ax.axvline(ACTIVE_MIN + 1, color=INK2, lw=1, ls=':')
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); style(ax)
    ax.legend(fontsize=8, frameon=False, title='year', title_fontsize=8)
fig.suptitle(f'Each point is one {AGG}-h window (dotted: "active" threshold, {ACTIVE_MIN} ILDN strokes)',
             color=INK, fontsize=11)
stamp(fig)
fig.savefig(os.path.join(OUT_DIR, 'ildn3_scatter.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# Fig 4: case studies — the windows with the most ILDN strokes (each panel on its own scale)
cases = np.argsort(tot_i)[::-1][:N_CASES]
cases = cases[tot_i[cases] > 0]
if len(cases):
    cols = [('ILDN', I), ('WWLLN', fields['WWLLN']), ('U-Net', fields['U-Net'])]
    fig, axs = plt.subplots(len(cases), 3, figsize=(11, 3.2 * len(cases)), squeeze=False, constrained_layout=True)
    for r_, k in enumerate(cases):
        for c_, (name, arr) in enumerate(cols):
            f = arr[k]
            vmax = max(np.percentile(f[f > 0], 99) if (f > 0).any() else 1, 1.01)
            vmin = 1 if name != 'U-Net' else max(vmax / 100, 1e-3)   # model: also show < 1 expected stroke
            im = draw(axs[r_, c_], f, mcolors.LogNorm(vmin=vmin, vmax=vmax),
                      f"{name} — {times[k]:%Y-%m-%d %HZ}\n{f.sum():,.0f} strokes in box")
            cbar = fig.colorbar(im, ax=axs[r_, c_], shrink=0.8)
            cbar.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle(f'Most active ILDN windows (each panel has its own colour scale) — {RUN_LABEL}',
                 color=INK, fontsize=11)
    stamp(fig)
    fig.savefig(os.path.join(OUT_DIR, 'ildn4_cases.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# Fig 5: score table
for g in groups:
    r = results[g]
    rows = [[lab] + [f"{r[k][key]:.2f}" for k in order] for key, lab in LABELS.items()]
    fig, ax = plt.subplots(figsize=(3.2 + 1.45 * len(order), 0.42 * len(rows) + 1.3), constrained_layout=True)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['score vs ILDN'] + [k + (' (reference)' if k == 'WWLLN' else '')
                                                                for k in order],
                   loc='center', cellLoc='center', colWidths=[0.42] + [0.58 / len(order)] * len(order))
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_facecolor('#2c5f8a'); c.set_text_props(color='white', fontweight='bold')
        elif j == 0:
            c.set_text_props(ha='left', color=INK); c.PAD = 0.03
    info = r['_ildn']
    ax.set_title(f"ILDN comparison, {g} — {info['strokes_in_box']:,} ILDN strokes, {info['windows']} windows "
                 f"({info['active_windows']} active)\n{RUN_LABEL}", fontsize=10, color=INK)
    stamp(fig)
    fig.savefig(os.path.join(OUT_DIR, f'ildn5_scores_{g}.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

# ── Summary JSON ────────────────────────────────────────────────────────────────
clean = lambda d: {k: (None if isinstance(v, float) and not np.isfinite(v) else
                       round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
summary = {
    'run_folder': RUN_NAME, 'label': RUN_LABEL, 'predictions': [PRED_PATH] + ([EXTRA_PATH] if len(parts) > 1 else []),
    'settings': {'box': BOX, 'ildn_types': ILDN_TYPES, 'ildn_min_ka': ILDN_MIN_KA,
                 'ildn_time_offset_h': ILDN_TIME_OFFSET_H, 'active_min': ACTIVE_MIN, 'years': YEARS,
                 'agg_hours': AGG, 'fss_window': FSS_W, 'ildn_files': files},
    'time_check': lag_res,
    'scores': {g: {k: clean(v) for k, v in r.items()} for g, r in results.items()},
}
with open(os.path.join(OUT_DIR, 'ildn_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=1, default=str, ensure_ascii=False)
print(f"\nFigures + ildn_summary.json → {OUT_DIR}", flush=True)
