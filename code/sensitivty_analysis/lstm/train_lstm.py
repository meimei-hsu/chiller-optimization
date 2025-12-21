import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# Configuration
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "chiller_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_noise_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "HVAC_electricity_demand_rate"
LOOKBACK = 24
BATCH_SIZE = 64      # 稍微加大 batch，加速（CPU 通常也有幫助）
EPOCHS = 10          # ✅ 只要趨勢，不用訓練到極致
SEED = 42

# ✅ Train subsample for speed (keeps time order by sorting index after sampling)
TRAIN_SAMPLE_SIZE = 50000   # 可改 20000/30000/50000，看你電腦

# Noise sweep (test only)
NOISE_LEVELS = np.arange(0.0, 3.01, 0.1)

# Optional: early stop training if it’s clearly converged
EARLY_STOP_PATIENCE = 2
EARLY_STOP_MIN_DELTA = 1e-4


# =====================================================
# Reproducibility
# =====================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(SEED)


# =====================================================
# Model
# =====================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# =====================================================
# Utilities
# =====================================================
def make_sequences(x_concat, y, lookback):
    """
    x_concat: (N, F) where F includes [scaled_X + scaled_y_lag]
    y: (N, 1) scaled target
    """
    xs, ys = [], []
    for t in range(lookback, len(x_concat)):
        xs.append(x_concat[t - lookback:t])
        ys.append(y[t])
    return np.array(xs), np.array(ys)


class SimpleEarlyStop:
    def __init__(self, patience=2, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, metric):
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


# =====================================================
# Load & preprocess data
# =====================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Cannot find CSV at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c != TARGET_COL]

# Time-order split
split = int(0.8 * len(df))
df_train_full = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

# ✅ Subsample train for speed (keep temporal variety, then sort back by index)
if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(df_train_full):
    df_train = (
        df_train_full
        .sample(n=TRAIN_SAMPLE_SIZE, random_state=SEED)
        .sort_index()
        .copy()
    )
else:
    df_train = df_train_full

print(f"Device: {device}")
print(f"Train size: {len(df_train)} (from {len(df_train_full)}), Test size: {len(df_test)}")
print(f"Features: {len(feature_cols)}, Lookback: {LOOKBACK}")
print("Fitting scalers on TRAIN (subsampled) ...")

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_raw = df_train[feature_cols].values
y_train_raw = np.log1p(df_train[TARGET_COL].values).reshape(-1, 1)

scaler_x.fit(x_train_raw)
scaler_y.fit(y_train_raw)

x_train_scaled = scaler_x.transform(x_train_raw)
y_train_scaled = scaler_y.transform(y_train_raw)

# Build training sequences (concat target lag into input)
x_train_concat = np.concatenate([x_train_scaled, y_train_scaled], axis=1)
x_train_seq, y_train_seq = make_sequences(x_train_concat, y_train_scaled, LOOKBACK)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(x_train_seq), torch.FloatTensor(y_train_seq)),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Train sequences: {x_train_seq.shape}")


# =====================================================
# Train once (noise = 0)
# =====================================================
model = LSTMModel(input_size=x_train_seq.shape[2]).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.L1Loss()

early_stop = SimpleEarlyStop(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)

train_losses = []
print("\nTraining baseline LSTM (noise = 0)...")

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    avg_loss = float(np.mean(batch_losses))
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train MAE (scaled log space): {avg_loss:.4f}")

    if early_stop.step(avg_loss):
        print("Early stop: training loss plateaued.")
        break

# Save training loss curve
plt.figure(figsize=(7, 4))
plt.plot(train_losses, marker="o")
plt.title("Training Loss (baseline, noise=0)")
plt.xlabel("Epoch")
plt.ylabel("MAE (scaled log space)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_loss_curve.png"))
plt.close()

print("Training done.\n")


# =====================================================
# Noise sweep (TEST only)
# =====================================================
model.eval()

# Prepare clean test scaled arrays
x_test_raw = df_test[feature_cols].values
y_test_raw = np.log1p(df_test[TARGET_COL].values).reshape(-1, 1)

x_test_scaled_clean = scaler_x.transform(x_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

results = []
print("Running noise sensitivity (test-time only)...")

for i, noise_std in enumerate(NOISE_LEVELS):
    # Make noise reproducible per noise level
    np.random.seed(SEED + i)

    temp_feature_idx = feature_cols.index("outdoor_temperature")
    noise = np.random.normal(loc=0.0, scale=float(noise_std), size=(len(x_test_scaled_clean), 1))
    x_test_noisy = x_test_scaled_clean[temp_feature_idx] + noise

    # Build sequences (still using true y_lag scaled as in your original design)
    x_test_concat = np.concatenate([x_test_noisy, y_test_scaled], axis=1)
    x_seq, y_seq = make_sequences(x_test_concat, y_test_scaled, LOOKBACK)

    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(x_seq).to(device)).cpu().numpy()

    # Inverse transform to original scale
    preds_log = scaler_y.inverse_transform(preds_scaled)
    preds = np.expm1(preds_log).ravel()

    y_true_log = scaler_y.inverse_transform(y_seq)
    y_true = np.expm1(y_true_log).ravel()

    r2 = float(r2_score(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))

    collapsed = (r2 < 0.0)

    results.append({
        "noise_std": float(noise_std),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "collapsed_R2<0": collapsed
    })

    print(f"noise={noise_std:.2f}  R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  collapsed={collapsed}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "noise_sensitivity_results.csv"), index=False)

# Plot noise vs R2
plt.figure(figsize=(8, 5))
plt.plot(results_df["noise_std"], results_df["R2"], marker="o")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Gaussian noise std (added to scaled X at test time)")
plt.ylabel("Test R² (original scale)")
plt.title("LSTM Robustness: Noise vs R² (train once, test noisy)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "noise_vs_r2.png"))
plt.close()

# Optional: save example forecast plot at a few noise levels
example_levels = [0.0, 0.5, 1.0, 2.0]
for noise_std in example_levels:
    # find closest
    idx = int(np.argmin(np.abs(results_df["noise_std"].values - noise_std)))
    noise_std = float(results_df.loc[idx, "noise_std"])

    np.random.seed(SEED + 1000 + idx)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=x_test_scaled_clean.shape)
    x_test_noisy = x_test_scaled_clean + noise
    x_test_concat = np.concatenate([x_test_noisy, y_test_scaled], axis=1)
    x_seq, y_seq = make_sequences(x_test_concat, y_test_scaled, LOOKBACK)

    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(x_seq).to(device)).cpu().numpy()

    preds = np.expm1(scaler_y.inverse_transform(preds_scaled)).ravel()
    y_true = np.expm1(scaler_y.inverse_transform(y_seq)).ravel()

    plt.figure(figsize=(12, 4))
    n = 200
    plt.plot(y_true[:n], label="Actual")
    plt.plot(preds[:n], label=f"Pred (noise={noise_std:.2f})", alpha=0.8)
    plt.title(f"Forecast Comparison (first {n} points) - noise={noise_std:.2f}")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"forecast_noise_{noise_std:.2f}.png"))
    plt.close()

# Print breaking point
collapsed_rows = results_df[results_df["R2"] < 0.0]
if len(collapsed_rows) > 0:
    first = collapsed_rows.iloc[0]
    print(f"\nFirst collapse (R² < 0) at noise_std ≈ {first['noise_std']:.2f}")
else:
    print("\nNo collapse observed (R² never dropped below 0).")

print("\nDone. Outputs saved to:", OUTPUT_DIR)
