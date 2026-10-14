"""
train_unet.py — lightning U-Net on AGG_HOURS-hour windows (default 12 h).

Three loss options (same data, model and split, so they compare cleanly as an ablation):

  LOSS = 'poisson'  (recommended)  model predicts log(expected strokes); Poisson loss on raw counts.
                    The prediction is the MEAN number of strokes, so monthly means match observations.
  LOSS = 'mse'                     Jones et al.: MSE on z-scored stroke counts. MSE_UNNORM chooses how
                    predictions are turned back into strokes ('direct' or Jones's 'jones' re-standardisation).
  LOSS = 'log_mse'                 MSE on z-scored log1p(strokes). Predicts a "typical" value, which is
                    far below the mean for rare, bursty lightning → intensity is underestimated.

AGG_HOURS  any whole number of hours (e.g. 3, 6, 12, 24, 48). Each window length gets its own cached
           data file, checkpoint and predictions file, so runs never overwrite each other.

CALIBRATE = True  additionally applies quantile mapping fitted on the VALIDATION year, so the
                  distribution of predicted intensities matches the observed one (test is never used).
"""
import xarray as xr
import numpy as np
import os
import sys
import json
import hashlib
import datetime
import random
import math
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim

# ── Config ─────────────────────────────────────────────────────────────────────
LOSS      = 'poisson'          # 'poisson' | 'mse' (Jones et al.) | 'log_mse'
MSE_UNNORM = 'direct'          # LOSS='mse' only: 'direct' = invert the z-score with the training mean/std
                               # 'jones'  = Jones et al.: re-standardise the predictions with their own
                               #            mean/std first, then give them the training mean/std.
                               # Only post-processing changes: set RETRAIN = False to reuse the MSE model.
AGG_HOURS = 12                 # window length in hours: any whole number (3, 6, 12, 24, 48 …)
CALIBRATE = True               # quantile-map predictions to the observed distribution (fit on val year)
RETRAIN   = True               # False = reuse the saved checkpoint (e.g. after fixing test-year data only).
                               # Refused if the checkpoint was trained on different years/inputs (no leakage).

DATA_DIR   = '/home/ec2-user/ML_Thesis_with_DL/data'
DATA_PATH  = f'{DATA_DIR}/jones_israel_all_years.nc'            # hourly source file
PREP_PATH  = f'{DATA_DIR}/jones_israel_{AGG_HOURS}h_OND.nc'      # cached aggregated file (one per window length)
RUN_NAME   = f'{AGG_HOURS}h_{LOSS}' + ('_jonesnorm' if LOSS == 'mse' and MSE_UNNORM == 'jones' else '')
CKPT_PATH  = f'{DATA_DIR}/unet_{AGG_HOURS}h_{LOSS}_best.pt'        # same model for both MSE un-normalisations
OUT_PATH   = f'{DATA_DIR}/unet_{RUN_NAME}_OND_predictions.nc'

LAT_MIN, LAT_MAX = 30.0, 43.0
LON_MIN, LON_MAX = 10.0, 39.0

ACTIVE_MONTHS = [10, 11, 12]

# Years left out completely (not used for training, validation or testing):
#   2025 — precipitation is all zeros in the source file (IMERG data gap). After re-downloading
#          IMERG 2025 and rebuilding the data files, remove 2025 from this list.
#   2023 — unusual year (few storms, mostly over land)
EXCLUDE_YEARS = [2023, 2025]
VAL_YEARS     = [2021]
TEST_YEARS    = [2022, 2024]
# everything else (2013–2020) is training

feature_name = ['precipitation', 'total_totals_index', 'ciwc500',
                'total_column_cloud_ice_water', 'cape', 'k_index',
                'total_column_cloud_liquid_water', 'lsm']
output_name  = ['ltg']

LOG_FEATURES = ['cape', 'precipitation', 'total_column_cloud_ice_water',
                'ciwc500', 'total_column_cloud_liquid_water']
LOG_TARGET   = (LOSS == 'log_mse')     # 'poisson' and 'mse' work on raw counts
WCD_MIN      = -2000.0

SEED       = 13
BATCH_SIZE = 16                        # short windows (1–6 h) give many more samples: 32–64 trains faster
EPOCHS     = 100
PATIENCE   = 10
LR         = 1e-3
GRAD_CLIP  = 1.0                       # protects against the occasional huge storm count

FSS_THRESHOLD = 1.0                    # strokes per window
FSS_WINDOW    = 3

if LOSS not in ('poisson', 'mse', 'log_mse'):
    sys.exit(f"LOSS must be 'poisson', 'mse' or 'log_mse' (got {LOSS!r})")
if MSE_UNNORM not in ('direct', 'jones'):
    sys.exit(f"MSE_UNNORM must be 'direct' or 'jones' (got {MSE_UNNORM!r})")
if not (isinstance(AGG_HOURS, int) and AGG_HOURS >= 1):
    sys.exit(f"AGG_HOURS must be a whole number of hours ≥ 1 (got {AGG_HOURS!r})")
if 24 % AGG_HOURS and AGG_HOURS % 24:
    print(f"NOTE: {AGG_HOURS} h does not divide a day evenly, so windows start at different times of day "
          f"(the diurnal cycle is mixed across windows).", flush=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE} | LOSS={LOSS}" + (f" ({MSE_UNNORM} un-normalisation)" if LOSS == 'mse' else '')
      + f" | window {AGG_HOURS} h | CALIBRATE={CALIBRATE}", flush=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── 1. Aggregated dataset (cached; rebuilt automatically when settings or source data change) ──
CACHE_KEY = {
    'agg_hours': AGG_HOURS,
    'months':    ACTIVE_MONTHS,
    'domain':    [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX],
    'variables': feature_name + output_name,
    'wcd_min':   WCD_MIN,
}

def source_stamp():
    """Modification time of the hourly source file (None if it isn't on this machine)."""
    return f"{os.path.getmtime(DATA_PATH):.0f}" if os.path.exists(DATA_PATH) else None

def cache_status():
    """(is_valid, reason). The cache is only reused if it was built with the same settings
    from the same version of the hourly file."""
    if not os.path.exists(PREP_PATH):
        return False, 'no cached file yet'
    with xr.open_dataset(PREP_PATH) as c:
        attrs = dict(c.attrs)
    for k, v in CACHE_KEY.items():
        if attrs.get(k) != str(v):
            return False, f"setting '{k}' differs (cache: {attrs.get(k)}, now: {v})"
    stamp = source_stamp()
    if stamp is not None and attrs.get('source_mtime') != stamp:
        return False, f'{os.path.basename(DATA_PATH)} changed since the cache was built'
    return True, 'settings and source file unchanged'

def build_aggregated():
    print(f"Building {AGG_HOURS}-hourly dataset from hourly file...", flush=True)
    ds = xr.open_dataset(DATA_PATH)
    lat_mask = (ds.lat >= LAT_MIN) & (ds.lat <= LAT_MAX)
    lon_mask = (ds.lon >= LON_MIN) & (ds.lon <= LON_MAX)
    ds = ds.isel(lat=lat_mask.values, lon=lon_mask.values)[feature_name + output_name]

    parts = []
    for yr in np.unique(ds.time.dt.year.values):
        d = ds.sel(time=str(yr))
        d = d.sel(time=d.time.dt.month.isin(ACTIVE_MONTHS)).load()
        if d.sizes['time'] == 0:
            continue
        if 'wcd' in d:
            d['wcd'] = d['wcd'].fillna(0).clip(min=WCD_MIN)

        # Flag variables that are zero everywhere for the whole year: that is missing data
        # stored as 0 (like the 2025 IMERG gap), not real zeros.
        for v in feature_name:
            if v != 'lsm' and float((d[v] == 0).mean()) > 0.999:
                print(f"  WARNING {yr}: '{v}' is zero everywhere — almost certainly missing data!",
                      flush=True)

        # An hour is valid only if every variable has data somewhere in the domain
        # (wcd excluded: it is legitimately NaN where there is no cloud, and was filled above).
        check_vars = [v for v in feature_name + output_name if v != 'wcd']
        hour_ok = xr.concat([d[v].notnull().any(['lat', 'lon']) for v in check_vars], dim='v').all('v')

        rs     = f'{AGG_HOURS}h'
        X      = d[feature_name].resample(time=rs).mean()
        y      = d[output_name].resample(time=rs).sum(min_count=1)
        nhrs   = d['time'].resample(time=rs).count()            # hours present in the window
        nvalid = hour_ok.resample(time=rs).sum()                # hours with all variables present
        keep   = (nhrs == AGG_HOURS) & (nvalid == AGG_HOURS)    # complete windows only
        agg    = xr.merge([X, y]).sel(time=nhrs.time[keep])
        parts.append(agg.astype('float32'))
        n_gap = int(((nhrs == AGG_HOURS) & (nvalid < AGG_HOURS)).sum())
        print(f"  {yr}: {d.sizes['time']} hourly → {agg.sizes['time']} × {AGG_HOURS}h"
              + (f"  ({n_gap} windows dropped: missing data)" if n_gap else ""), flush=True)
    out = xr.concat(parts, dim='time')
    out.attrs.update({k: str(v) for k, v in CACHE_KEY.items()})
    out.attrs['source_mtime'] = source_stamp()
    out.to_netcdf(PREP_PATH)
    return out

valid, reason = cache_status()
if valid:
    print(f"Loading cached {AGG_HOURS}-hourly dataset: {PREP_PATH} ({reason})", flush=True)
    ds = xr.open_dataset(PREP_PATH).load()
else:
    print(f"Rebuilding cache: {reason}", flush=True)
    ds = build_aggregated()

n_lat = (ds.sizes['lat'] // 4) * 4
n_lon = (ds.sizes['lon'] // 4) * 4
ds = ds.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
print(f"Grid {n_lat}×{n_lon}, {ds.sizes['time']} time steps", flush=True)

# ── 2. Year-based split ────────────────────────────────────────────────────────
assert not set(EXCLUDE_YEARS) & set(VAL_YEARS + TEST_YEARS), "excluded year also in val/test"
years     = ds.time.dt.year.values
keep      = ~np.isin(years, EXCLUDE_YEARS)
idx_val   = np.where(keep & np.isin(years, VAL_YEARS))[0]
idx_test  = np.where(keep & np.isin(years, TEST_YEARS))[0]
idx_train = np.where(keep & ~np.isin(years, VAL_YEARS + TEST_YEARS))[0]
print(f"Excluded years: {EXCLUDE_YEARS}", flush=True)
print(f"Splits — train: {len(idx_train)} {sorted(set(years[idx_train].tolist()))}, "
      f"val: {len(idx_val)} {VAL_YEARS}, test: {len(idx_test)} {TEST_YEARS}", flush=True)

obs_counts_val  = ds['ltg'].isel(time=idx_val)
obs_counts_test = ds['ltg'].isel(time=idx_test)

# ── 3. Transforms + normalisation ──────────────────────────────────────────────
ds_model = ds.copy()
for v in LOG_FEATURES:
    ds_model[v] = np.log1p(ds_model[v].clip(min=0))
if LOG_TARGET:
    ds_model['ltg'] = np.log1p(ds_model['ltg'].clip(min=0))
else:
    ds_model['ltg'] = ds_model['ltg'].clip(min=0).fillna(0)

X_tr, X_va, X_te = (ds_model[feature_name].isel(time=i) for i in (idx_train, idx_val, idx_test))
y_tr, y_va       = (ds_model[output_name].isel(time=i)  for i in (idx_train, idx_val))

train_X_mean = X_tr.groupby('time.month').mean(dim=['time', 'lat', 'lon'])
train_X_std  = X_tr.groupby('time.month').std(dim=['time', 'lat', 'lon'])
train_y_mean = y_tr.groupby('time.month').mean(dim=['time', 'lat', 'lon'])
train_y_std  = y_tr.groupby('time.month').std(dim=['time', 'lat', 'lon'])

def zscore_X(d):
    return ((d.groupby('time.month') - train_X_mean).groupby('time.month') / train_X_std).fillna(0)

def zscore_y(d):
    return ((d.groupby('time.month') - train_y_mean).groupby('time.month') / train_y_std).fillna(0)

def to_tensor(d):
    arr = np.moveaxis(np.array(d.to_array()), 0, -1)
    arr = np.nan_to_num(arr, nan=0.0)
    return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)

tr_X_tensor, val_X_tensor, test_X_tensor = (to_tensor(zscore_X(d)) for d in (X_tr, X_va, X_te))
if LOSS == 'poisson':
    tr_y_tensor, val_y_tensor = to_tensor(y_tr), to_tensor(y_va)          # raw counts
else:
    tr_y_tensor, val_y_tensor = to_tensor(zscore_y(y_tr)), to_tensor(zscore_y(y_va))

# ── 4. Model (unchanged) ───────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, zdim, ydim, xdim):
        super(SimpleCNN, self).__init__()
        self.encoder0 = self.encoder_block(zdim, 32)
        self.encoder1 = self.encoder_block(32, 16)
        self.center   = self.conv_block(16, 8)
        self.decoder1 = self.decoder_block(8, 16)
        self.decoder0 = self.decoder_block(16, 32)
        self.outputs  = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1, padding=0))

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def encoder_block(self, in_ch, out_ch):
        return nn.Sequential(self.conv_block(in_ch, out_ch), nn.MaxPool2d(2, 2))

    def decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),        nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        c  = self.center(e1)
        d1 = self.decoder1(c)
        d0 = self.decoder0(d1)
        return self.outputs(d0)     # poisson: log(expected strokes) | log_mse: z-scored log1p

model = SimpleCNN(len(feature_name), n_lat, n_lon).to(DEVICE)

if LOSS == 'poisson':
    # Start from the domain-mean rate so early training isn't spent learning the overall level
    with torch.no_grad():
        model.outputs[0].bias.fill_(math.log(max(float(tr_y_tensor.mean()), 1e-3)))
    # full=True adds the log(y!) term: constant w.r.t. the model (same gradients, same training),
    # but makes the loss a true negative log-likelihood (≥ 0) instead of an arbitrary-offset negative number
    _poisson = nn.PoissonNLLLoss(log_input=True, full=True)
    criterion = lambda out, y: _poisson(out.clamp(-20, 12), y)
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

# ── 5. Training ────────────────────────────────────────────────────────────────
def train(model, train_inputs, train_targets, val_inputs, val_targets):
    best_val_loss, patience_counter, best_epoch = float('inf'), 0, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_inputs))
        tr_loss, n = 0.0, 0
        for i in range(0, len(train_inputs), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(train_inputs[idx].to(DEVICE)), train_targets[idx].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_loss += loss.item() * len(idx); n += len(idx)
        tr_loss /= n

        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_inputs), BATCH_SIZE):
                xb, yb = val_inputs[i:i + BATCH_SIZE].to(DEVICE), val_targets[i:i + BATCH_SIZE].to(DEVICE)
                va_loss += criterion(model(xb), yb).item() * len(xb); n += len(xb)
        va_loss /= n
        print(f'Epoch {epoch+1}/{EPOCHS}, Train loss: {tr_loss:.4f}, Val loss: {va_loss:.4f}', flush=True)

        if va_loss < best_val_loss:
            best_val_loss, patience_counter, best_epoch = va_loss, 0, epoch + 1
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break
    return {'best_epoch': best_epoch, 'epochs_run': epoch + 1, 'best_val_loss': round(best_val_loss, 6)}

# The checkpoint gets a small JSON "passport" with the years and inputs it was trained on, so a
# checkpoint can never silently be reused with a different split (e.g. one that saw today's test years).
TRAIN_YEARS = sorted(set(years[idx_train].tolist()))
CKPT_META_PATH = CKPT_PATH + '.json'
try:
    import inspect
    _model_md5 = hashlib.md5(inspect.getsource(SimpleCNN).encode()).hexdigest()[:8]
except (OSError, TypeError):
    _model_md5 = 'unknown'
run_meta = {'train_years': TRAIN_YEARS, 'val_years': list(VAL_YEARS), 'test_years': list(TEST_YEARS),
            'excluded_years': list(EXCLUDE_YEARS), 'features': list(feature_name), 'loss': LOSS,
            'log_features': list(LOG_FEATURES), 'agg_hours': AGG_HOURS, 'months': list(ACTIVE_MONTHS),
            'hyperparams': {'seed': SEED, 'batch_size': BATCH_SIZE, 'max_epochs': EPOCHS, 'patience': PATIENCE,
                            'lr': LR, 'grad_clip': GRAD_CLIP},
            'model': {'name': type(model).__name__, 'source_md5': _model_md5,
                      'n_params': sum(p.numel() for p in model.parameters())}}

if RETRAIN or not os.path.exists(CKPT_PATH):
    run_meta['training'] = train(model, tr_X_tensor, tr_y_tensor, val_X_tensor, val_y_tensor)
    run_meta['trained_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    with open(CKPT_META_PATH, 'w') as f:
        json.dump(run_meta, f, indent=1)
else:
    if not os.path.exists(CKPT_META_PATH):
        sys.exit(f"RETRAIN=False but {CKPT_PATH} has no record of what it was trained on "
                 f"(made by an older version). Set RETRAIN = True once.")
    saved_meta = json.load(open(CKPT_META_PATH))
    leaked = set(saved_meta['train_years']) & set(VAL_YEARS + TEST_YEARS)
    if leaked:
        sys.exit(f"Refusing to reuse {CKPT_PATH}: it was trained on {sorted(leaked)}, which are now "
                 f"validation/test years (data leakage). Set RETRAIN = True.")
    changed = [k for k in ('train_years', 'features', 'loss', 'log_features', 'agg_hours', 'months')
               if saved_meta.get(k) != run_meta[k]]
    if changed:
        sys.exit(f"Refusing to reuse {CKPT_PATH}: settings changed since it was trained ({changed}). "
                 f"Set RETRAIN = True.")
    run_meta['trained_at'] = saved_meta.get('trained_at')
    # the reused model was trained with the settings in its passport, not necessarily today's
    for k in ('hyperparams', 'model', 'training'):
        run_meta[k] = saved_meta.get(k, 'unknown (checkpoint made by an older version)')
    _saved_md5 = saved_meta['model'].get('source_md5') if isinstance(saved_meta.get('model'), dict) else 'unknown'
    if 'unknown' not in (_saved_md5, _model_md5) and _saved_md5 != _model_md5:
        sys.exit(f"Refusing to reuse {CKPT_PATH}: the SimpleCNN code changed since it was trained. Set RETRAIN = True.")
    print(f"RETRAIN=False → reusing checkpoint trained {run_meta['trained_at']} on {TRAIN_YEARS}", flush=True)

# ── 6. Predictions in stroke counts ────────────────────────────────────────────
print("Loading best checkpoint...", flush=True)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

def predict_raw(X_tensor):
    with torch.no_grad():
        parts = [model(X_tensor[i:i + BATCH_SIZE].to(DEVICE)).cpu()
                 for i in range(0, len(X_tensor), BATCH_SIZE)]
    return torch.cat(parts, dim=0).numpy().squeeze(1)

def to_counts(raw, ref_da):
    """Model output → expected strokes per window."""
    if LOSS == 'poisson':
        return np.exp(np.clip(raw, -20, 12))
    da = xr.DataArray(raw, dims=['time', 'lat', 'lon'],
                      coords={'time': ref_da.time.values, 'lat': ref_da.lat.values, 'lon': ref_da.lon.values})
    if LOSS == 'mse' and MSE_UNNORM == 'jones':
        # Jones et al.: standardise the predictions by their own mean/std over this whole set,
        # then give them the training mean/std of each month
        da = (da - da.mean()) / da.std()
    t = ((da.groupby('time.month') * train_y_std['ltg']).groupby('time.month') + train_y_mean['ltg']).clip(min=0)
    return (np.expm1(t) if LOG_TARGET else t).values

pred_val_raw  = to_counts(predict_raw(val_X_tensor),  obs_counts_val)
pred_test_raw = to_counts(predict_raw(test_X_tensor), obs_counts_test)

# ── 7. Quantile-mapping calibration (fit on VALIDATION year only) ──────────────
def fit_quantile_map(pred, obs):
    # dense probabilities in the upper tail, where all the lightning lives
    probs = np.unique(np.concatenate([np.linspace(0, 0.9, 200), 1 - np.logspace(-6, -1, 2000)]))
    qp = np.quantile(pred.ravel(), probs)
    qo = np.quantile(np.nan_to_num(obs).ravel(), probs)
    qp = np.maximum.accumulate(qp) + np.arange(len(qp)) * 1e-12     # strictly increasing for interp
    return qp, qo

if CALIBRATE:
    qp, qo = fit_quantile_map(pred_val_raw, obs_counts_val.values)
    pred_test = np.interp(pred_test_raw, qp, qo).reshape(pred_test_raw.shape)
    print(f"Quantile mapping fitted on {VAL_YEARS}: "
          f"val mean pred {pred_val_raw.mean():.3f} → obs {np.nanmean(obs_counts_val.values):.3f}", flush=True)
else:
    pred_test = pred_test_raw

# ── 8. Save ────────────────────────────────────────────────────────────────────
dims   = ['time', 'lat', 'lon']
coords = {'time': obs_counts_test.time.values, 'lat': obs_counts_test.lat.values,
          'lon': obs_counts_test.lon.values}
loss_desc = {'poisson': 'Poisson NLL on raw strokes (predicts mean)',
             'mse':     f'MSE on z-scored strokes (Jones et al.; {MSE_UNNORM} un-normalisation)',
             'log_mse': 'MSE on z-scored log1p(strokes)'}[LOSS]
calib_desc = f'quantile mapping fitted on {VAL_YEARS}' if CALIBRATE else 'none'

data_vars = {
    'pred_count':     xr.DataArray(pred_test.astype('float32'), dims=dims, coords=coords,
                                   attrs={'description': f'predicted strokes per {AGG_HOURS} h '
                                                         f'(calibration: {calib_desc})'}),
    'pred_count_raw': xr.DataArray(pred_test_raw.astype('float32'), dims=dims, coords=coords,
                                   attrs={'description': 'model output before calibration'}),
    'obs_count':      obs_counts_test.astype('float32'),
}
if LOSS == 'poisson':
    data_vars['pred_prob'] = xr.DataArray((1 - np.exp(-pred_test_raw)).astype('float32'), dims=dims,
                                          coords=coords,
                                          attrs={'description': 'P(at least 1 stroke) = 1 - exp(-rate)'})
ds_out = xr.Dataset(data_vars, attrs={
    'agg_hours': AGG_HOURS, 'loss': LOSS, 'loss_desc': loss_desc, 'calibration': calib_desc,
    'mse_unnorm': MSE_UNNORM if LOSS == 'mse' else 'n/a',
    'log_target': int(LOG_TARGET), 'val_years': str(VAL_YEARS), 'test_years': str(TEST_YEARS),
    'excluded_years': str(EXCLUDE_YEARS),
    'train_years': json.dumps(TRAIN_YEARS), 'features': json.dumps(list(feature_name)),
    'ckpt_trained_at': str(run_meta.get('trained_at')),
    # full configuration, so the figure scripts can label the run and keep different set-ups apart
    'run_name': RUN_NAME, 'log_features': json.dumps(list(LOG_FEATURES)), 'months': json.dumps(ACTIVE_MONTHS),
    'domain': json.dumps([LAT_MIN, LAT_MAX, LON_MIN, LON_MAX]), 'wcd_min': WCD_MIN,
    'fss_threshold': FSS_THRESHOLD, 'fss_window': FSS_WINDOW,
    'hyperparams': json.dumps(run_meta['hyperparams']), 'model': json.dumps(run_meta['model']),
    'training': json.dumps(run_meta.get('training'))})
ds_out.to_netcdf(OUT_PATH)
print(f"Saved → {OUT_PATH}", flush=True)

# ── 9. Metrics (raw and calibrated) ────────────────────────────────────────────
obs_np = np.nan_to_num(obs_counts_test.values.astype('float32'))

def compute_fss(obs, pred, threshold=FSS_THRESHOLD, window_size=FSS_WINDOW):
    ob = (obs >= threshold).astype(float); pb = (pred >= threshold).astype(float)
    of = uniform_filter(ob, size=window_size, mode='constant', cval=0)
    pf = uniform_filter(pb, size=window_size, mode='constant', cval=0)
    den = np.sum(pf ** 2 + of ** 2)
    return np.nan if den == 0 else 1 - np.sum((pf - of) ** 2) / den

def report(pred, name):
    o, p = obs_np.ravel(), pred.ravel()
    clim_o, clim_p = obs_np.mean(0).ravel(), pred.mean(0).ravel()
    fss = np.array([compute_fss(obs_np[i], pred[i]) for i in range(len(obs_np))])
    print(f"\n  [{name}]")
    print(f"  Mean strokes obs / pred   : {o.mean():.3f} / {p.mean():.3f}  (ratio {p.mean() / max(o.mean(), 1e-9):.2f})")
    print(f"  Event rate obs / pred     : {(o >= FSS_THRESHOLD).mean():.4f} / {(p >= FSS_THRESHOLD).mean():.4f}")
    print(f"  R² per-cell climatology   : {r2_score(clim_o, clim_p):.3f}  (r = {pearsonr(clim_o, clim_p)[0]:.3f})")
    print(f"  R² all pixels (log1p)     : {r2_score(np.log1p(o), np.log1p(p)):.3f}")
    print(f"  Mean FSS (3×3, ≥1)        : {np.nanmean(fss):.3f}")

print(f"\n{'='*58}\n  TEST SET  {TEST_YEARS}  — LOSS={LOSS}\n{'='*58}")
report(pred_test_raw, 'raw model output')
if CALIBRATE:
    report(pred_test, 'quantile-mapped')
print(f"{'='*58}", flush=True)