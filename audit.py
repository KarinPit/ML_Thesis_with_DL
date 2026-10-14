"""
audit.py — data integrity, leakage and skill audit for the lightning U-Net.

Run on EC2 from the repo folder. It reads every setting (features, years, loss, paths, model
class) straight from train_unet.py, so it always audits the configuration you actually train.

    python audit.py                 # everything, incl. control experiments (trains a few small models)
    python audit.py --quick         # data + leakage checks + baselines only (no training)
    python audit.py --pred unet_12h_poisson_top7_OND_predictions.nc   # audit a different run
(the loss and window length come from train_unet.py — set LOSS / AGG_HOURS there first)

Nothing is overwritten: checkpoints, caches and prediction files are only read.
Output: a report with PASS / WARN / FAIL lines, a summary at the end, figures in figures/audit/<run name>/.

Parts
  A. Data integrity      ranges, completeness, duplicated years/samples, lightning record,
                         time alignment (lag test) and spatial alignment (shift test)
  B. Leakage             split disjointness, checkpoint provenance, prediction-file provenance,
                         inputs that are suspiciously close to the target
  C. Skill vs baselines  climatology, persistence and CAPE×precip (R14) baselines with the same
                         metrics and calibration; bootstrap confidence interval; per-year scores
  D. Controls            (skipped with --quick) retrain with shuffled targets and with inputs from
                         another day — both must lose their skill; repeat the real model with
                         several seeds; permutation importance of each input
"""
import argparse, ast, hashlib, json, math, os, sys, time, warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ════════════════════════════════════════════════════════════════════════════════
# Settings from train_unet.py
# ════════════════════════════════════════════════════════════════════════════════
def read_train_config(path):
    """Evaluate the simple top-level assignments of train_unet.py without running it."""
    src = open(path).read()
    tree = ast.parse(src)
    cfg = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                pairs = [(tgt, node.value)]
            elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                pairs = list(zip(tgt.elts, node.value.elts))
            else:
                continue
            for t, v in pairs:
                if not isinstance(t, ast.Name):
                    continue
                try:
                    cfg[t.id] = ast.literal_eval(v)
                    continue
                except Exception:
                    pass
                try:   # f-strings / simple expressions that only use earlier settings
                    cfg[t.id] = eval(compile(ast.Expression(v), path, 'eval'), {'__builtins__': {}}, dict(cfg))
                except Exception:
                    pass
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimpleCNN'), None)
    cfg['_model_src'] = ast.get_source_segment(src, cls) if cls else None
    return cfg


parser = argparse.ArgumentParser(description='Audit data, leakage and skill of the lightning U-Net.')
parser.add_argument('--quick', action='store_true', help='skip the control experiments (no training)')
parser.add_argument('--pred', default=None, help='predictions file to audit (default: OUT_PATH of train_unet.py)')
parser.add_argument('--config', default='train_unet.py', help='training script to read settings from')
parser.add_argument('--seeds', type=int, default=3, help='number of seeds for the real-model repeat (default 3)')
parser.add_argument('--max-epochs', type=int, default=60, help='epoch cap for control trainings (default 60)')
parser.add_argument('--shift-days', type=int, default=7, help='time shift for the shifted-input control')
parser.add_argument('--fig-dir', default=None, help='default: figures/audit/<run name>')
args = parser.parse_args()

cfg = read_train_config(args.config)
need = ['DATA_DIR', 'DATA_PATH', 'PREP_PATH', 'CKPT_PATH', 'OUT_PATH', 'LOSS', 'feature_name', 'output_name',
        'LOG_FEATURES', 'VAL_YEARS', 'TEST_YEARS', 'ACTIVE_MONTHS', 'AGG_HOURS',
        'LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX', 'SEED', 'BATCH_SIZE', 'LR', 'PATIENCE']
missing_cfg = [k for k in need if k not in cfg]
if missing_cfg:
    sys.exit(f"Could not read these settings from {args.config}: {missing_cfg}")
EXCLUDE_YEARS = cfg.get('EXCLUDE_YEARS', [])
LOSS          = cfg['LOSS']
FEATURES      = list(cfg['feature_name'])
TARGET        = cfg['output_name'][0]
LOG_FEATURES  = list(cfg['LOG_FEATURES'])
LOG_TARGET    = (LOSS == 'log_mse')
PRED_PATH     = args.pred or cfg['OUT_PATH']
if not os.path.isabs(PRED_PATH) and not os.path.exists(PRED_PATH):
    PRED_PATH = os.path.join(cfg['DATA_DIR'], PRED_PATH)
if args.fig_dir is None:   # one folder per run, so audits of different windows/losses don't overwrite each other
    _run = os.path.basename(PRED_PATH).replace('_OND_predictions.nc', '').replace('.nc', '')
    args.fig_dir = os.path.join('figures', 'audit', _run)
os.makedirs(args.fig_dir, exist_ok=True)
THR, WIN = cfg.get('FSS_THRESHOLD', 1.0), cfg.get('FSS_WINDOW', 3)

# ════════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════════════════════
RESULTS, _section = [], ['']

def section(title):
    _section[0] = title
    print(f"\n{'═' * 100}\n  {title}\n{'═' * 100}", flush=True)

def report(level, msg):
    RESULTS.append((level, _section[0], msg))
    print(f"  [{level}] {msg}", flush=True)

def info(msg):
    print(f"         {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# Metrics (same definitions as train_unet.py / jones_figures.py)
# ════════════════════════════════════════════════════════════════════════════════
def fss_parts(o, p, thr=THR, w=WIN):
    of = uniform_filter((o >= thr).astype(float), size=w, mode='constant', cval=0)
    pf = uniform_filter((p >= thr).astype(float), size=w, mode='constant', cval=0)
    return np.sum((pf - of) ** 2), np.sum(pf ** 2 + of ** 2)

def fss_per_window(obs, pred, thr=THR):
    out = np.full(len(obs), np.nan)
    for i in range(len(obs)):
        num, den = fss_parts(obs[i], pred[i], thr)
        if den > 0:
            out[i] = 1 - num / den
    return out

def metrics(obs, pred):
    obs, pred = np.nan_to_num(obs), np.nan_to_num(pred)
    f1 = fss_per_window(obs, pred)
    parts = np.array([fss_parts(obs[i], pred[i]) for i in range(len(obs))])
    o, p = obs.ravel(), pred.ravel()
    ob, pb = o >= THR, p >= THR
    co, cp = obs.mean(0).ravel(), pred.mean(0).ravel()
    return {
        'mean ratio':      p.mean() / max(o.mean(), 1e-12),
        'event rate o/p':  f"{ob.mean():.3f}/{pb.mean():.3f}",
        'FSS mean':        np.nanmean(f1),
        'FSS pooled':      1 - parts[:, 0].sum() / max(parts[:, 1].sum(), 1e-12),
        'FSS ≥10 strokes': np.nanmean(fss_per_window(obs, pred, thr=10.0)),
        'CSI':             (ob & pb).sum() / max((ob | pb).sum(), 1),
        'R² clim':         r2_score(co, cp),
        'r clim':          pearsonr(co, cp)[0] if cp.std() > 0 else np.nan,
        'R² log1p':        r2_score(np.log1p(o), np.log1p(p)),
        '_fss':            f1,
    }

def print_table(rows):
    keys = ['mean ratio', 'event rate o/p', 'FSS mean', 'FSS pooled', 'FSS ≥10 strokes', 'CSI',
            'R² clim', 'r clim', 'R² log1p']
    df = pd.DataFrame({name: {k: m[k] for k in keys} for name, m in rows.items()}).T
    with pd.option_context('display.width', 250, 'display.max_columns', 20,
                           'display.float_format', '{:.3f}'.format):
        print(df.to_string())

def fit_quantile_map(pred, obs):
    probs = np.unique(np.concatenate([np.linspace(0, 0.9, 200), 1 - np.logspace(-6, -1, 2000)]))
    qp = np.quantile(np.nan_to_num(pred).ravel(), probs)
    qo = np.quantile(np.nan_to_num(obs).ravel(), probs)
    qp = np.maximum.accumulate(qp) + np.arange(len(qp)) * 1e-12
    return qp, qo

def calibrate(pred_val, obs_val, pred_test):
    qp, qo = fit_quantile_map(pred_val, obs_val)
    return np.interp(pred_test, qp, qo).reshape(pred_test.shape)


# ════════════════════════════════════════════════════════════════════════════════
# Load the aggregated data exactly as train_unet.py does
# ════════════════════════════════════════════════════════════════════════════════
print(f"Auditing settings from {args.config}: LOSS={LOSS}, features={FEATURES}")
print(f"  val {cfg['VAL_YEARS']}  test {cfg['TEST_YEARS']}  excluded {EXCLUDE_YEARS}")
if not os.path.exists(cfg['PREP_PATH']):
    sys.exit(f"{cfg['PREP_PATH']} not found — run train_unet.py once first (it builds this file).")
ds = xr.open_dataset(cfg['PREP_PATH']).load()
n_lat, n_lon = (ds.sizes['lat'] // 4) * 4, (ds.sizes['lon'] // 4) * 4
ds = ds.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
years = ds.time.dt.year.values
keep = ~np.isin(years, EXCLUDE_YEARS)
idx_val   = np.where(keep & np.isin(years, cfg['VAL_YEARS']))[0]
idx_test  = np.where(keep & np.isin(years, cfg['TEST_YEARS']))[0]
idx_train = np.where(keep & ~np.isin(years, cfg['VAL_YEARS'] + cfg['TEST_YEARS']))[0]
TRAIN_YEARS = sorted(set(years[idx_train].tolist()))
ltg = np.nan_to_num(ds[TARGET].values.astype('float64'))
times = pd.DatetimeIndex(ds.time.values)


# ════════════════════════════════════════════════════════════════════════════════
# A. DATA INTEGRITY
# ════════════════════════════════════════════════════════════════════════════════
section("A1. Cache consistency with train_unet.py settings")
attrs = ds.attrs
cache_vars = attrs.get('variables')
if cache_vars is None:
    report('WARN', 'cache has no settings recorded (built by an old train_unet.py) — rerun train_unet.py once')
elif cache_vars != str(FEATURES + [TARGET]):
    report('FAIL', f"cache was built for variables {cache_vars}, settings say {FEATURES + [TARGET]}")
else:
    report('PASS', 'cache variables match the training settings')
missing_vars = [v for v in FEATURES + [TARGET] if v not in ds]
if missing_vars:
    print(f"\n  STOP: {cfg['PREP_PATH']} has no {missing_vars}.\n"
          f"  The cached {cfg['AGG_HOURS']}-hour file was built with different inputs than {args.config} now lists\n"
          f"  (cache: {cache_vars or list(ds.data_vars)}).\n"
          f"  Run 'python {args.config}' first with the inputs you want — it rebuilds the cache and\n"
          f"  retrains — then run the audit again.")
    sys.exit(1)
step = pd.Series(np.diff(ds.time.values)).mode()[0]
report('PASS' if step == pd.Timedelta(hours=cfg['AGG_HOURS']) else 'FAIL',
       f"time step {step} (expected {cfg['AGG_HOURS']} h)")
report('PASS' if times.is_monotonic_increasing and not times.duplicated().any() else 'FAIL',
       f"time sorted and unique ({times.duplicated().sum()} duplicates)")
months_found = sorted(set(times.month))
report('PASS' if set(months_found) <= set(cfg['ACTIVE_MONTHS']) else 'FAIL', f"months in cache: {months_found}")
for c in ('lat', 'lon'):
    d = np.diff(ds[c].values)
    report('PASS' if np.allclose(d, d[0], atol=1e-4) else 'FAIL', f"{c} spacing regular ({d.min():.4f}…{d.max():.4f})")

section(f"A2. Physical ranges ({cfg['AGG_HOURS']}-hour values the model sees)")
RANGES = {'cape': (0, 20000), 'precipitation': (0, 200), 'lsm': (0, 1), 'rh': (0, 100), 't2m': (220, 330),
          'total_totals_index': (-60, 80), 'k_index': (-80, 70), 'total_column_cloud_ice_water': (0, 20),
          'total_column_cloud_liquid_water': (0, 20), 'ciwc500': (0, 0.01), 'ciwc1000': (0, 0.01),
          'wcd': (-12000, 8000), 'shear': (0, 150), 'cbh': (0, 20000), TARGET: (0, 1e7)}
for v in FEATURES + [TARGET]:
    a = ds[v].values
    fin = a[np.isfinite(a)]
    lo, hi = RANGES.get(v, (-np.inf, np.inf))
    eps = 1e-3 * (hi - lo) if np.isfinite(hi - lo) else 0
    bad = np.mean((fin < lo - eps) | (fin > hi + eps)) if fin.size else 1.0
    msg = (f"{v:32s} min {fin.min():11.4g}  max {fin.max():11.4g}  NaN {np.isnan(a).mean():.4f}  "
           f"outside [{lo}, {hi}]: {bad:.5f}") if fin.size else f"{v}: all NaN"
    report('PASS' if bad == 0 and fin.size else ('WARN' if bad < 0.001 else 'FAIL'), msg)
frac_nonint = np.mean(np.abs(ltg - np.round(ltg)) > 1e-3)
report('PASS' if frac_nonint == 0 else 'FAIL',
       f"lightning totals are whole stroke counts (non-integer fraction {frac_nonint:.5f})")
if 'lsm' in ds:
    lsm_var = float(ds['lsm'].std('time').max())
    report('PASS' if lsm_var < 1e-4 else 'WARN', f"land-sea mask constant in time (max std {lsm_var:.2e})")

section(f"A3. Completeness per year ({cfg['AGG_HOURS']}-hour windows; zero-everywhere fields = missing data)")
expected_per_year = sum(pd.Period(f'2001-{m:02d}').days_in_month for m in cfg['ACTIVE_MONTHS']) * 24 // cfg['AGG_HOURS']
rows = []
for y in sorted(set(years)):
    sel = years == y
    r = {'year': y, 'windows': int(sel.sum()), 'split': ('excluded' if y in EXCLUDE_YEARS else 'val' if y in cfg['VAL_YEARS']
                                                         else 'test' if y in cfg['TEST_YEARS'] else 'train')}
    for v in FEATURES:
        if v == 'lsm':
            continue
        zero_win = (ds[v].isel(time=np.where(sel)[0]) == 0).all(['lat', 'lon']).mean().item()
        r[v[:14]] = zero_win
    rows.append(r)
comp = pd.DataFrame(rows).set_index('year')
with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.2f}'.format):
    print(comp.to_string())
for y, r in comp.iterrows():
    if r['windows'] < expected_per_year and r['split'] != 'excluded':
        report('WARN', f"{y} ({r['split']}): {r['windows']}/{expected_per_year} windows (some dropped for missing data)")
    zero_cols = [c for c in comp.columns if c not in ('windows', 'split') and r[c] > 0.5]
    if zero_cols and r['split'] != 'excluded':
        report('FAIL', f"{y} ({r['split']}): zero-everywhere in most windows for {zero_cols} → missing data used as input")
if not any(l != 'PASS' for l, s, _ in RESULTS if s.startswith('A3')):
    report('PASS', 'every used year complete, no all-zero input fields')

section("A4. Duplicated data (copy/merge errors)")
def h(a):
    return hashlib.md5(np.ascontiguousarray(np.nan_to_num(a).astype('float32')).tobytes()).hexdigest()
dup_found = False
for v in FEATURES + [TARGET]:
    if v == 'lsm':
        continue
    hashes = {}
    for y in sorted(set(years)):
        sel = np.where(years == y)[0]
        hashes.setdefault(h(ds[v].values[sel]), []).append(y)
    for hh, ys in hashes.items():
        if len(ys) > 1:
            dup_found = True
            report('FAIL', f"'{v}' is identical in years {[int(y) for y in ys]} — a year was copied")
X_all = np.stack([np.nan_to_num(ds[v].values) for v in FEATURES if v != 'lsm'], axis=1)
win_hash = np.array([h(X_all[i]) for i in range(len(X_all))])
split_of = np.array(['train'] * len(years), dtype=object)
split_of[idx_val], split_of[idx_test] = 'val', 'test'
split_of[~keep] = 'excluded'
_, inv, counts = np.unique(win_hash, return_inverse=True, return_counts=True)
dups = np.where(counts[inv] > 1)[0]
cross = [i for i in dups if len(set(split_of[inv == inv[i]])) > 1]
if len(dups):
    dup_found = True
    report('FAIL' if cross else 'WARN',
           f"{len(dups)} windows have exactly identical inputs to another window "
           f"({len(cross)} of them across splits) e.g. {[str(times[i]) for i in dups[:4]]}")
if not dup_found:
    report('PASS', 'no duplicated years and no identical input windows')

section("A5. Lightning record over the years (WWLLN detection efficiency changes with time)")
lr = pd.DataFrame({'year': years, 'mean': ltg.mean(axis=(1, 2)), 'events': (ltg >= THR).mean(axis=(1, 2))})
lr = lr.groupby('year').mean()
lr['split'] = [comp.loc[y, 'split'] for y in lr.index]
print(lr.to_string(float_format=lambda x: f'{x:.4f}'))
used = lr[lr.split != 'excluded']
slope = np.polyfit(used.index, used['mean'], 1)[0] / used['mean'].mean() * 100
info(f"trend in mean strokes per {cfg['AGG_HOURS']} h: {slope:+.1f}% per year")
ratio = lr.loc[lr.split == 'test', 'mean'].mean() / lr.loc[lr.split == 'train', 'mean'].mean()
report('PASS' if 0.67 < ratio < 1.5 else 'WARN',
       f"test-years / train-years mean lightning = {ratio:.2f}"
       + ('' if 0.67 < ratio < 1.5 else ' — large shift (detection efficiency or climate); calibration may compensate'))
if abs(slope) > 5:
    report('WARN', f"lightning counts trend {slope:+.1f}%/yr — WWLLN detection efficiency changes over the record "
                   "(Jones corrects with Hutchins et al. 2012); mention in the thesis")

section("A6. Time & space alignment of lightning vs inputs (hourly file)")
if not os.path.exists(cfg['DATA_PATH']):
    report('WARN', f"{cfg['DATA_PATH']} not found — alignment tests skipped")
else:
    hr = xr.open_dataset(cfg['DATA_PATH'])
    hr = hr.sel(lat=ds.lat, lon=ds.lon, method='nearest')
    align_vars = [v for v in ['precipitation', 'total_column_cloud_ice_water', 'cape'] if v in hr]
    test_year = [y for y in cfg['TEST_YEARS'] if y not in EXCLUDE_YEARS][:1]
    check_years = [TRAIN_YEARS[len(TRAIN_YEARS) // 2]] + test_year
    LAGS = [-24, -12, -6, -3, -2, -1, 0, 1, 2, 3, 6, 12, 24]
    lag_res = {}
    for y in check_years:
        hy = hr.sel(time=(hr.time.dt.year == y) & hr.time.dt.month.isin(cfg['ACTIVE_MONTHS']))
        L = np.log1p(np.nan_to_num(hy[TARGET].values.astype('float32')))
        for v in align_vars:
            A = np.nan_to_num(hy[v].values.astype('float32'))
            A = np.log1p(np.clip(A, 0, None)) if v != 'cape' else A
            cs = []
            for lag in LAGS:
                a = L[max(0, -lag): len(L) - max(0, lag)]
                b = A[max(0, lag): len(A) - max(0, -lag)]
                cs.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            lag_res[(y, v)] = cs
            best = LAGS[int(np.nanargmax(cs))]
            line = '  '.join(f"{l:+d}h:{c:.3f}" for l, c in zip(LAGS, cs))
            print(f"  {y} {v:30s} {line}")
            if v == 'precipitation':
                report('PASS' if abs(best) <= 1 else 'FAIL',
                       f"{y}: lightning best matches precipitation at lag {best:+d} h (expect 0 or ±1)")
            else:
                info(f"{y}: {v} best lag {best:+d} h (ERA5 field; physical lags of a few hours are possible)")
        # spatial shift test with precipitation (same hour)
        if 'precipitation' in align_vars:
            P = np.log1p(np.clip(np.nan_to_num(hy['precipitation'].values.astype('float32')), 0, None))
            k, shifts = 2, {}
            Lc = L[:, k:-k, k:-k]
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    Pc = P[:, k + dy: P.shape[1] - k + dy, k + dx: P.shape[2] - k + dx]
                    shifts[(dy, dx)] = np.corrcoef(Lc.ravel(), Pc.ravel())[0, 1]
            best = max(shifts, key=shifts.get)
            flipped = np.corrcoef(L.ravel(), P[:, ::-1, :].ravel())[0, 1]
            grid = np.array([[shifts[(dy, dx)] for dx in range(-k, k + 1)] for dy in range(-k, k + 1)])
            print(f"  {y} precipitation spatial shift (rows dy=-2..2, cols dx=-2..2):\n"
                  + "\n".join('     ' + ' '.join(f'{c:.3f}' for c in row) for row in grid))
            report('FAIL' if flipped > shifts[(0, 0)] else
                   'PASS' if best == (0, 0) else ('WARN' if max(map(abs, best)) <= 1 else 'FAIL'),
                   f"{y}: lightning lines up with precipitation at shift {best} (expect (0, 0)); "
                   f"N–S flipped correlation {flipped:.3f} vs aligned {shifts[(0, 0)]:.3f}")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        colors = ['#2a78d6', '#eb6834', '#1baf7a']
        fig, axes = plt.subplots(1, len(check_years), figsize=(6 * len(check_years), 3.8), squeeze=False)
        for ax, y in zip(axes[0], check_years):
            for c, v in zip(colors, align_vars):
                if (y, v) in lag_res:
                    ax.plot(LAGS, lag_res[(y, v)], marker='o', ms=4, lw=2, color=c, label=v)
            ax.axvline(0, color='#8a8986', lw=1)
            ax.set_title(f'{y}: correlation of lightning with input at lag', fontsize=10, color='#0b0b0b')
            ax.set_xlabel('lag (hours; + = input later than lightning)', color='#52514e')
            ax.grid(color='#e6e5e1', lw=0.6); ax.legend(fontsize=8, frameon=False)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.fig_dir, 'A6_time_alignment.png'), dpi=130); plt.close(fig)
    except Exception as e:
        info(f"(figure skipped: {e})")


# ════════════════════════════════════════════════════════════════════════════════
# B. LEAKAGE
# ════════════════════════════════════════════════════════════════════════════════
section("B1. Split")
sets = {'train': set(TRAIN_YEARS), 'val': set(cfg['VAL_YEARS']) - set(EXCLUDE_YEARS),
        'test': set(cfg['TEST_YEARS']) - set(EXCLUDE_YEARS)}
info(f"train {sorted(sets['train'])} | val {sorted(sets['val'])} | test {sorted(sets['test'])} | excluded {EXCLUDE_YEARS}")
overlap = (sets['train'] & sets['val']) | (sets['train'] & sets['test']) | (sets['val'] & sets['test'])
report('PASS' if not overlap else 'FAIL', f"train/val/test years are disjoint {'' if not overlap else overlap}")
report('PASS' if len(idx_test) and len(idx_val) and len(idx_train) else 'FAIL',
       f"windows — train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")
gap_days = (times[idx_test].min() - times[idx_train].max()).days if len(idx_test) else None
info(f"closest train→test distance: {abs(gap_days) if gap_days is not None else '?'} days "
     "(seasons are Oct–Dec only, so neighbouring years are ~9 months apart)")

section("B2. Checkpoint provenance (was the model trained on these years only?)")
meta_path = cfg['CKPT_PATH'] + '.json'
ckpt_meta = None
if not os.path.exists(cfg['CKPT_PATH']):
    report('WARN', f"no checkpoint at {cfg['CKPT_PATH']}")
elif not os.path.exists(meta_path):
    report('WARN', "checkpoint has no record of the years it was trained on (made by an older train_unet.py). "
                   "Retrain once (RETRAIN = True) so this can be verified.")
else:
    ckpt_meta = json.load(open(meta_path))
    trained_on = set(ckpt_meta.get('train_years', []))
    bad = trained_on & (sets['test'] | sets['val'])
    report('PASS' if not bad else 'FAIL',
           f"checkpoint trained on {sorted(trained_on)}" + ('' if not bad else f" — includes val/test years {sorted(bad)}!"))
    same = (ckpt_meta.get('features') == FEATURES and ckpt_meta.get('loss') == LOSS)
    report('PASS' if same else 'WARN', f"checkpoint features/loss match current settings ({ckpt_meta.get('trained_at')})")

section("B3. Prediction file provenance")
pred = None
if not os.path.exists(PRED_PATH):
    report('WARN', f"{PRED_PATH} not found — run train_unet.py first; parts B3/C use it")
else:
    pred = xr.open_dataset(PRED_PATH).load()
    pyears = sorted(set(pd.DatetimeIndex(pred.time.values).year))
    report('PASS' if set(pyears) <= sets['test'] else 'FAIL', f"prediction times are all in test years {pyears}")
    pa = pred.attrs
    for k, want in [('val_years', str(cfg['VAL_YEARS'])), ('test_years', str(cfg['TEST_YEARS'])),
                    ('loss', LOSS)]:
        if k in pa and str(pa[k]) != want:
            report('WARN', f"predictions were made with {k}={pa[k]}, settings now say {want} — rerun train_unet.py")
    if 'train_years' in pa:
        ty = set(json.loads(pa['train_years']))
        report('PASS' if not (ty & sets['test']) else 'FAIL', f"predictions come from a model trained on {sorted(ty)}")
    else:
        report('WARN', "prediction file does not record the training years (older train_unet.py) — rerun to verify")
    if ckpt_meta and pa.get('ckpt_trained_at') and pa['ckpt_trained_at'] != ckpt_meta.get('trained_at'):
        report('WARN', 'prediction file was made by a different checkpoint than the one on disk')
    common = np.intersect1d(pred.time.values, ds.time.values)
    if len(common):
        same_obs = np.allclose(np.nan_to_num(pred['obs_count'].sel(time=common).values),
                               ltg[np.searchsorted(ds.time.values, common)])
        report('PASS' if same_obs else 'FAIL', "observed lightning in the prediction file equals the data file")

section("B4. Inputs suspiciously close to the target (a leaked copy of lightning would score ≈ 1)")
report('PASS' if TARGET not in FEATURES else 'FAIL', f"'{TARGET}' is not an input")
rng = np.random.default_rng(0)
tr_flat = np.nan_to_num(ltg[idx_train]).ravel()
samp = rng.choice(tr_flat.size, size=min(2_000_000, tr_flat.size), replace=False)
y_s = tr_flat[samp]
pos = y_s > 0
def _c(f, a, b):
    try:
        return abs(f(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0
    except Exception:
        return 0.0
for v in FEATURES:
    a = np.nan_to_num(ds[v].values[idx_train]).ravel()[samp]
    # three views, because ~90% of pixels are 0 strokes and rank correlations get diluted by those ties:
    # linear, linear vs log-lightning, and rank correlation on lightning pixels only
    scores = {'pearson': _c(pearsonr, a, y_s), 'pearson log': _c(pearsonr, a, np.log1p(y_s)),
              'spearman ltg>0': _c(spearmanr, a[pos], y_s[pos]) if pos.sum() > 100 else 0.0}
    worst = max(scores.values())
    level = 'FAIL' if worst > 0.95 else ('WARN' if worst > 0.8 else 'PASS')
    report(level, f"{v:32s} |r| with lightning (training years): "
                  + ', '.join(f"{k} {val:.2f}" for k, val in scores.items()))
info("Reminder: choose inputs, loss and settings using training/validation years only. Rankings made on "
     "test years (e.g. an old Fig 5) leak test information into design decisions.")


# ════════════════════════════════════════════════════════════════════════════════
# C. SKILL VS BASELINES (all calibrated the same way: quantile mapping on the val year)
# ════════════════════════════════════════════════════════════════════════════════
section("C. Skill vs baselines on the test years")
obs_test, obs_val = ltg[idx_test], ltg[idx_val]
mon = times.month.values
clim = {m: ltg[idx_train][mon[idx_train] == m].mean(0) for m in set(mon)}
clim_val = np.stack([clim[m] for m in mon[idx_val]])
clim_test = np.stack([clim[m] for m in mon[idx_test]])

t_index = {t: i for i, t in enumerate(ds.time.values)}
def previous_window(idx):
    prev = np.full((len(idx),) + ltg.shape[1:], np.nan)
    for j, i in enumerate(idx):
        k = t_index.get(ds.time.values[i] - np.timedelta64(cfg['AGG_HOURS'], 'h'))
        if k is not None:
            prev[j] = ltg[k]
    return prev
pers_test, pers_val = previous_window(idx_test), previous_window(idx_val)
ok_p = np.isfinite(pers_test).all(axis=(1, 2))

rows = {'climatology (raw)': metrics(obs_test, clim_test),
        'climatology (calibrated)': metrics(obs_test, calibrate(clim_val, obs_val, clim_test))}
if 'cape' in ds and 'precipitation' in ds:
    cp = np.nan_to_num(ds['cape'].values) * np.nan_to_num(ds['precipitation'].values)
    c = ltg[idx_train].sum() / max(cp[idx_train].sum(), 1e-12)
    rows['R14 c·CAPE·P (raw)'] = metrics(obs_test, c * cp[idx_test])
    rows['R14 c·CAPE·P (calibrated)'] = metrics(obs_test, calibrate(c * cp[idx_val], obs_val, c * cp[idx_test]))
rows[f"persistence: previous {cfg['AGG_HOURS']} h"] = metrics(obs_test[ok_p], np.nan_to_num(pers_test[ok_p]))
if pred is not None and not set(ds.time.values[idx_test]) <= set(pred.time.values):
    report('WARN', 'prediction file does not cover the current test windows (made with another split) — '
                   'U-Net rows skipped; rerun train_unet.py')
    pred = None
if pred is not None:
    pt = pred.sel(time=ds.time.values[idx_test])
    if 'pred_count_raw' in pt:
        rows['U-Net (raw)'] = metrics(obs_test, pt['pred_count_raw'].values)
    rows['U-Net (as saved)'] = metrics(obs_test, pt['pred_count'].values)
print_table(rows)
info(f"persistence uses observed lightning {cfg['AGG_HOURS']} h earlier, which the U-Net does NOT get; it shows how much "
     "of the score is just 'storms last'. Calibrated rows are fair comparisons with the calibrated U-Net.")

if pred is not None:
    unet = rows['U-Net (as saved)']
    base_name = max([k for k in rows if 'calibrated' in k], key=lambda k: rows[k]['FSS mean'])
    base = rows[base_name]
    report('PASS' if unet['FSS mean'] > base['FSS mean'] else 'WARN',
           f"U-Net FSS {unet['FSS mean']:.3f} vs best calibrated baseline ({base_name}) {base['FSS mean']:.3f}"
           + ('' if unet['FSS mean'] > base['FSS mean'] else
              ' — not a bug, but the U-Net does not beat this simpler method; important for the thesis'))
    # bootstrap over days: CI of the FSS difference
    days = pd.DatetimeIndex(ds.time.values[idx_test]).normalize()
    uniq = days.unique()
    diff = unet['_fss'] - base['_fss']
    rng = np.random.default_rng(1)
    by_day = [np.where(days == d)[0] for d in uniq]
    boots = []
    for _ in range(2000):
        pick = np.concatenate([by_day[i] for i in rng.integers(0, len(uniq), len(uniq))])
        boots.append(np.nanmean(diff[pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    report('PASS' if lo > 0 else 'WARN',
           f"FSS gain over {base_name}: {np.nanmean(diff):+.3f} (95% CI {lo:+.3f} … {hi:+.3f}, day bootstrap)")
    for y in sorted(sets['test']):
        sel = pd.DatetimeIndex(ds.time.values[idx_test]).year == y
        m = metrics(obs_test[sel], pt['pred_count'].values[sel])
        info(f"test year {y}: FSS {m['FSS mean']:.3f}, pooled {m['FSS pooled']:.3f}, mean ratio {m['mean ratio']:.2f}, "
             f"R² clim {m['R² clim']:.3f}")


# ════════════════════════════════════════════════════════════════════════════════
# D. CONTROL EXPERIMENTS (retraining) — skipped with --quick
# ════════════════════════════════════════════════════════════════════════════════
if args.quick:
    section("D. Control experiments — skipped (--quick)")
else:
    import torch, torch.nn as nn, torch.optim as optim
    section("D. Control experiments (each trains a fresh model in memory; nothing is saved)")
    if cfg['_model_src'] is None:
        sys.exit("SimpleCNN class not found in train_unet.py")
    ns = {'nn': nn, 'torch': torch}
    exec(cfg['_model_src'], ns)
    SimpleCNN = ns['SimpleCNN']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    dsm = ds.copy()
    for v in LOG_FEATURES:
        if v in dsm:
            dsm[v] = np.log1p(dsm[v].clip(min=0))
    dsm[TARGET] = np.log1p(dsm[TARGET].clip(min=0)) if LOG_TARGET else dsm[TARGET].clip(min=0).fillna(0)
    Xtr_ds = dsm[FEATURES].isel(time=idx_train)
    xm = Xtr_ds.groupby('time.month').mean(dim=['time', 'lat', 'lon'])
    xs = Xtr_ds.groupby('time.month').std(dim=['time', 'lat', 'lon'])
    ytr_ds = dsm[[TARGET]].isel(time=idx_train)
    ym = ytr_ds.groupby('time.month').mean(dim=['time', 'lat', 'lon'])
    ysd = ytr_ds.groupby('time.month').std(dim=['time', 'lat', 'lon'])

    def to_tensor(d):
        arr = np.nan_to_num(np.moveaxis(np.array(d.to_array()), 0, -1), nan=0.0)
        return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)
    def X_of(idx):
        d = dsm[FEATURES].isel(time=idx)
        return to_tensor(((d.groupby('time.month') - xm).groupby('time.month') / xs).fillna(0))
    def y_of(idx):
        d = dsm[[TARGET]].isel(time=idx)
        if LOSS == 'poisson':
            return to_tensor(d)
        return to_tensor(((d.groupby('time.month') - ym).groupby('time.month') / ysd).fillna(0))
    def to_counts(raw, idx):
        if LOSS == 'poisson':
            return np.exp(np.clip(raw, -20, 12))
        mu = np.array([float(ym[TARGET].sel(month=k)) for k in mon[idx]])[:, None, None]
        sd = np.array([float(ysd[TARGET].sel(month=k)) for k in mon[idx]])[:, None, None]
        t = np.clip(raw * sd + mu, 0, None)            # 'mse': strokes; 'log_mse': log1p(strokes)
        return np.expm1(t) if LOG_TARGET else t

    Xtr, Xva, Xte = X_of(idx_train), X_of(idx_val), X_of(idx_test)
    ytr, yva = y_of(idx_train), y_of(idx_val)

    def make_loss():
        if LOSS == 'poisson':
            pl = nn.PoissonNLLLoss(log_input=True, full=True)
            return lambda o, y: pl(o.clamp(-20, 12), y)
        return nn.MSELoss()

    def train_model(seed, Xa, ya, Xv, yv):
        torch.manual_seed(seed); np.random.seed(seed)
        model = SimpleCNN(len(FEATURES), n_lat, n_lon).to(DEVICE)
        if LOSS == 'poisson':
            with torch.no_grad():
                model.outputs[0].bias.fill_(math.log(max(float(ya.mean()), 1e-3)))
        crit, opt = make_loss(), optim.Adam(model.parameters(), lr=cfg['LR'])
        best, best_state, wait = float('inf'), None, 0
        for ep in range(args.max_epochs):
            model.train()
            perm = torch.randperm(len(Xa))
            for i in range(0, len(Xa), cfg['BATCH_SIZE']):
                b = perm[i:i + cfg['BATCH_SIZE']]
                opt.zero_grad()
                loss = crit(model(Xa[b].to(DEVICE)), ya[b].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('GRAD_CLIP', 1.0))
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = sum(crit(model(Xv[i:i + 64].to(DEVICE)), yv[i:i + 64].to(DEVICE)).item() * len(Xv[i:i + 64])
                         for i in range(0, len(Xv), 64)) / len(Xv)
            if vl < best:
                best, wait = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg['PATIENCE']:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model

    def predict(model, X):
        with torch.no_grad():
            return torch.cat([model(X[i:i + 64].to(DEVICE)).cpu() for i in range(0, len(X), 64)]).numpy().squeeze(1)

    def evaluate(model, Xv=Xva, Xt=Xte, iv=idx_val, it=idx_test, obs_t=obs_test):
        pv, pt_ = to_counts(predict(model, Xv), iv), to_counts(predict(model, Xt), it)
        return metrics(obs_t, calibrate(pv, ltg[iv], pt_)), pv, pt_

    ctrl = {}
    t0 = time.time()
    real = []
    for s in range(args.seeds):
        m_s = train_model(cfg['SEED'] + s, Xtr, ytr, Xva, yva)
        res, _, _ = evaluate(m_s)
        real.append(res)
        info(f"real model, seed {cfg['SEED'] + s}: calibrated FSS {res['FSS mean']:.3f}  ({time.time() - t0:.0f}s)")
        if s == 0:
            model0 = m_s
    ctrl['real (mean of seeds)'] = {k: (np.mean([r[k] for r in real]) if not isinstance(real[0][k], (str, np.ndarray))
                                        else real[0][k]) for k in real[0]}
    fss_seeds = np.array([r['FSS mean'] for r in real])
    report('PASS' if fss_seeds.std() < 0.03 else 'WARN',
           f"seed-to-seed spread of FSS: {fss_seeds.mean():.3f} ± {fss_seeds.std():.3f} (report this ± in the thesis)")
    if pred is not None:
        saved = rows['U-Net (as saved)']['FSS mean']
        report('PASS' if abs(saved - fss_seeds.mean()) < max(0.05, 3 * fss_seeds.std()) else 'WARN',
               f"retrained models reproduce the saved result (saved {saved:.3f} vs retrained {fss_seeds.mean():.3f})")

    # shuffled targets: inputs and targets no longer belong together
    rng = np.random.default_rng(123)
    shuf = train_model(cfg['SEED'], Xtr, ytr[torch.tensor(rng.permutation(len(ytr)))], Xva, yva)
    ctrl['shuffled targets'], _, _ = evaluate(shuf)
    clim_fss = rows['climatology (calibrated)']['FSS mean']
    report('PASS' if ctrl['shuffled targets']['FSS mean'] <= clim_fss + 0.05 else 'FAIL',
           f"shuffled-target model FSS {ctrl['shuffled targets']['FSS mean']:.3f} ≤ climatology {clim_fss:.3f} + 0.05 "
           "(if not, the model gets lightning information from somewhere other than the inputs)")

    # shifted inputs: each target paired with inputs from ~N days earlier (same split, same season).
    # The shift is rounded to a whole number of windows, so it lands on a window start for any AGG_HOURS.
    n_shift = max(1, round(args.shift_days * 24 / cfg['AGG_HOURS']))
    shift = np.timedelta64(n_shift * cfg['AGG_HOURS'], 'h')
    sh_lbl = f"inputs {n_shift * cfg['AGG_HOURS'] / 24:g} days earlier"
    def shifted(idx):
        src = [t_index.get(ds.time.values[i] - shift) for i in idx]
        keep_ = np.array([k is not None and split_of[k] == split_of[i] for k, i in zip(src, idx)], dtype=bool)
        return (np.array([k for k, ok in zip(src, keep_) if ok], dtype=int),
                np.asarray(idx, dtype=int)[keep_])
    s_tr, t_tr = shifted(idx_train); s_va, t_va = shifted(idx_val); s_te, t_te = shifted(idx_test)
    if min(len(s_tr), len(s_va), len(s_te)) == 0:
        report('WARN', f"shifted-input control skipped: no windows {sh_lbl[7:]} inside the same season "
                       f"(try a smaller --shift-days)")
    else:
        sh_model = train_model(cfg['SEED'], X_of(s_tr), y_of(t_tr), X_of(s_va), y_of(t_va))
        ctrl[sh_lbl], _, _ = evaluate(sh_model, X_of(s_va), X_of(s_te), t_va, t_te, ltg[t_te])
        sh_fss, real_fss = ctrl[sh_lbl]['FSS mean'], fss_seeds.mean()
        report('PASS' if sh_fss < real_fss - 0.1 else 'WARN',
               f"with {sh_lbl} FSS drops {real_fss:.3f} → {sh_fss:.3f} "
               "(a large drop means the model reads the actual weather, not just the season/geography)")

    print()
    print_table(ctrl | {'climatology (calibrated)': rows['climatology (calibrated)']})

    # permutation importance on the seed-0 model
    section("D2. Permutation importance (shuffle one input across test windows; bigger drop = more important)")
    base_res, pv0, _ = evaluate(model0)
    imp = {}
    for k, v in enumerate(FEATURES):
        Xp = Xte.clone()
        Xp[:, k] = Xp[torch.tensor(np.random.default_rng(7).permutation(len(Xp))), k]
        pt_ = to_counts(predict(model0, Xp), idx_test)
        imp[v] = base_res['FSS mean'] - metrics(obs_test, calibrate(pv0, obs_val, pt_))['FSS mean']
    for v, d in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {v:32s} FSS drop {d:+.3f}")
    top = max(imp, key=imp.get)
    info(f"most important input: {top}. If 'lsm' alone dominates, the skill is mostly geography.")
    share = imp[top] / max(base_res['FSS mean'] - clim_fss, 1e-6)
    report('WARN' if share > 0.9 else 'PASS',
           f"'{top}' alone accounts for {100 * min(share, 9.99):.0f}% of the skill above climatology"
           + (" — check that it is a genuine weather input and not derived from lightning" if share > 0.9 else ''))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        names = list(ctrl) + ['climatology (calibrated)']
        vals = [ctrl[n]['FSS mean'] if n in ctrl else rows[n]['FSS mean'] for n in names]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.barh(names[::-1], vals[::-1], color='#2a78d6', height=0.55)
        for i, v in enumerate(vals[::-1]):
            ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9, color='#52514e')
        ax.set_xlim(0, 1); ax.set_xlabel('mean FSS (3×3, ≥1 stroke), calibrated', color='#52514e')
        ax.set_title('Control experiments: only the real model should be skilful', fontsize=10, color='#0b0b0b')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.fig_dir, 'D_controls.png'), dpi=130); plt.close(fig)
    except Exception as e:
        info(f"(figure skipped: {e})")


# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 100}\n  SUMMARY\n{'═' * 100}")
for lvl in ('FAIL', 'WARN'):
    items = [(s, m) for l, s, m in RESULTS if l == lvl]
    print(f"  {lvl}: {len(items)}")
    for s, m in items:
        print(f"    - [{s.split('.')[0]}] {m}")
print(f"  PASS: {sum(l == 'PASS' for l, _, _ in RESULTS)}")
print(f"\n  Figures: {args.fig_dir}/")
