"""
train_unet.py
-------------
Trains a U-Net (SimpleCNN) on Jones et al. CPLRSTW features to predict
hourly lightning counts over Israel / Eastern Mediterranean.

Input:  data/jones_israel_*.nc   (one file per year)
Output: models/unet_jones.pt     (best checkpoint)
        models/unet_jones_last.pt (final epoch)
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATTERN = 'data/jones_israel_*.nc'
MODEL_DIR    = 'models'

FEATURE_VARS = ['cape', 'precipitation', 'lsm', 'rh', 'shear', 't2m', 'wcd', 'vertical_velocity_500hPa']
TARGET_VAR   = 'ltg'

BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 0.005
PATIENCE     = 5
NUM_WORKERS  = 0      # 0 = main process only (safest with large in-RAM tensors)
SEED         = 13

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Model ──────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder0 = self._enc(in_channels, 32)
        self.encoder1 = self._enc(32, 16)
        self.center   = self._conv(16, 8)
        self.decoder1 = self._dec(8, 16)
        self.decoder0 = self._dec(16, 32)
        self.out      = nn.Conv2d(32, 1, kernel_size=1)

    def _conv(self, c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True))

    def _enc(self, c_in, c_out):
        return nn.Sequential(self._conv(c_in, c_out), nn.MaxPool2d(2))

    def _dec(self, c_in, c_out):
        return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        c  = self.center(e1)
        d1 = self.decoder1(c)
        d0 = self.decoder0(d1)
        return self.out(d0)


def load_split_to_ram(ds, time_slice, feat_mean, feat_std, tgt_mean, tgt_std, label=''):
    """Normalise and load a contiguous time slice into RAM as float32 numpy arrays."""
    print(f"  Loading {label} into RAM...", flush=True)
    sub = ds.isel(time=time_slice)

    # Normalise features
    X_norm = ((sub[FEATURE_VARS].groupby('time.month') - feat_mean
               ).groupby('time.month') / feat_std).fillna(0)
    # Store as float16 to halve RAM usage; cast to float32 on GPU in training loop
    X = np.stack([X_norm[v].values for v in FEATURE_VARS], axis=1).astype('float16')

    # Normalise target
    y_norm = ((sub[TARGET_VAR].groupby('time.month') - tgt_mean
               ).groupby('time.month') / tgt_std).fillna(0)
    y = y_norm.values[:, np.newaxis, :, :].astype('float16')

    print(f"  {label}: X={X.shape}  y={y.shape}  "
          f"RAM used: {X.nbytes/1e9:.1f}GB + {y.nbytes/1e9:.1f}GB", flush=True)
    return X, y


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Using device: {DEVICE}", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Open all years lazily ──────────────────────────────────────────────────
    print("Opening dataset...", flush=True)
    ds = xr.open_mfdataset(DATA_PATTERN, combine='by_coords', chunks={'time': 200})

    n_lat = (ds.sizes['lat'] // 4) * 4
    n_lon = (ds.sizes['lon'] // 4) * 4
    ds = ds.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
    print(f"Grid: {n_lat} lat × {n_lon} lon, total timesteps: {ds.sizes['time']}", flush=True)

    # ── Contiguous splits ──────────────────────────────────────────────────────
    n_times = ds.sizes['time']
    n_train   = int(n_times * 0.6)
    n_val     = int(n_times * 0.2)
    sl_tr     = slice(0, n_train)
    val_start = n_train
    val_stop  = n_train + n_val
    sl_val    = slice(val_start, val_stop)
    te_start  = val_stop
    sl_te     = slice(te_start, n_times)
    print(f"Splits — train: {n_train}, val: {n_val}, test: {n_times - n_train - n_val}", flush=True)

    # ── Normalisation stats (fit on train only) ────────────────────────────────
    print("Computing normalisation stats...", flush=True)
    ds_tr      = ds.isel(time=sl_tr)
    feat_mean  = ds_tr[FEATURE_VARS].groupby('time.month').mean(dim=['time','lat','lon']).compute()
    feat_std   = ds_tr[FEATURE_VARS].groupby('time.month').std(dim=['time','lat','lon']).compute()
    tgt_mean   = ds_tr[TARGET_VAR].groupby('time.month').mean(dim=['time','lat','lon']).compute()
    tgt_std    = ds_tr[TARGET_VAR].groupby('time.month').std(dim=['time','lat','lon']).compute()
    print("Stats done.", flush=True)

    # ── Load ONLY train into RAM; val + test streamed in chunks ───────────────
    X_tr, y_tr = load_split_to_ram(ds, sl_tr, feat_mean, feat_std, tgt_mean, tgt_std, 'train')

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE == 'cuda'))

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = SimpleCNN(in_channels=len(FEATURE_VARS)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_train_batches = len(train_loader)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss    = float('inf')
    patience_counter = 0

    print("\nTraining...", flush=True)
    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        tr_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_loader, 1):
            X, y = X.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            if batch_idx % 200 == 0 or batch_idx == n_train_batches:
                print(f"  Epoch {epoch:3d}  batch {batch_idx:4d}/{n_train_batches}  "
                      f"train_loss={tr_loss/batch_idx:.5f}", flush=True)
        tr_loss /= n_train_batches

        # Validate (streamed in chunks to save RAM)
        model.eval()
        val_loss, val_batches = 0.0, 0
        CHUNK = 500
        with torch.no_grad():
            for start in range(val_start, val_stop, CHUNK):
                end = min(start + CHUNK, sl_val.stop)
                X_c, y_c = load_split_to_ram(ds, slice(start, end),
                                             feat_mean, feat_std, tgt_mean, tgt_std, '')
                loader_c = DataLoader(TensorDataset(torch.from_numpy(X_c), torch.from_numpy(y_c)),
                                      batch_size=BATCH_SIZE, shuffle=False)
                for X, y in loader_c:
                    val_loss += criterion(model(X.to(DEVICE, dtype=torch.float32)), y.to(DEVICE, dtype=torch.float32)).item()
                    val_batches += 1
                del X_c, y_c
        val_loss /= max(val_batches, 1)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.5f}  val={val_loss:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'unet_jones.pt'))
            print(f"  ✓ best model saved (val={val_loss:.5f})", flush=True)
        else:
            patience_counter += 1
            print(f"  patience {patience_counter}/{PATIENCE}", flush=True)
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'unet_jones_last.pt'))

    # ── Test evaluation (lazy, no RAM spike) ───────────────────────────────────
    print("\nEvaluating on test set (streaming from disk)...", flush=True)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'unet_jones.pt'), map_location=DEVICE))
    model.eval()

    all_preds, all_true = [], []
    CHUNK = 500
    for start in range(te_start, n_times, CHUNK):
        end = min(start + CHUNK, n_times)
        X_chunk, y_chunk = load_split_to_ram(ds, slice(start, end),
                                             feat_mean, feat_std, tgt_mean, tgt_std,
                                             f'test chunk {start}-{end}')
        with torch.no_grad():
            preds = model(torch.from_numpy(X_chunk).to(DEVICE)).cpu().numpy().squeeze(1)
        all_preds.append(preds)
        all_true.append(y_chunk.squeeze(1))

    preds_np = np.concatenate(all_preds, axis=0)
    true_np  = np.concatenate(all_true,  axis=0)
    mask = np.isfinite(true_np) & np.isfinite(preds_np)
    ss   = 1 - np.mean((preds_np[mask] - true_np[mask])**2) / np.var(true_np[mask])
    print(f"Test skill score (1 - MSE/Var): {ss:.4f}", flush=True)

    times_test = ds.isel(time=sl_te).time.values
    xr.DataArray(preds_np, dims=['time','lat','lon'],
                 coords={'time': times_test, 'lat': ds.lat.values, 'lon': ds.lon.values}
                 ).to_dataset(name='ltg').to_netcdf('data/unet_jones_predictions.nc')
    print("Predictions saved → data/unet_jones_predictions.nc", flush=True)


if __name__ == '__main__':
    main()
