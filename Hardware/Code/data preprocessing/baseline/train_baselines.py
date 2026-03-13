"""
BASELINE MODEL COMPARISON
Train simple baselines to compare against SNN

This script trains:
1. Random Forest (traditional ML)
2. LSTM (temporal neural network)
3. Simple MLP (feedforward neural network)
4. Event-Driven Spiking Neural Network (SNN)

The script loops over detection (PRED_HORIZON=0) and prediction (PRED_HORIZON=5)
Then compares:
- Accuracy
- Number of parameters
- Inference time
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import snntorch as snn
from snntorch import surrogate
import time
import joblib
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("  MODEL COMPARISON ON SENSOR.CSV")
print("="*70)
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & ENGINEER FEATURES
# ═══════════════════════════════════════════════════════════════════════

print("Loading data...")
df = pd.read_csv("sensor.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df[df["valid"] == 1].copy()
df = df[df["baseline_cm"] > 0].copy()
df = df[(df["dist_f_cm"] > 20) & (df["dist_f_cm"] <= 600)].copy()

# Feature engineering (from pre future new)
group_cols = []
if "session_id" in df.columns:
    group_cols.append("session_id")

df["delta"] = df["dist_f_cm"] - df["baseline_cm"]
df["dist_to_enter"] = df["dist_f_cm"] - df["enter_thr_cm"]
df["dist_to_exit"]  = df["dist_f_cm"] - df["exit_thr_cm"]

if group_cols:
    df["time_gap"] = df.groupby(group_cols)["time_ms"].diff()
else:
    df["time_gap"] = df["time_ms"].diff()
df = df.dropna(subset=["time_gap"])
df["time_gap"] = df["time_gap"].replace(0, np.nan)
df = df.dropna(subset=["time_gap"])

if group_cols:
    df["velocity"]     = df.groupby(group_cols)["dist_f_cm"].diff() / df["time_gap"]
    df["acceleration"] = df.groupby(group_cols)["velocity"].diff()  / df["time_gap"]
else:
    df["velocity"]     = df["dist_f_cm"].diff() / df["time_gap"]
    df["acceleration"] = df["velocity"].diff()  / df["time_gap"]
df = df.dropna(subset=["velocity", "acceleration"])

if "state" not in df.columns:
    df["state"] = (df["danger"] > 0).astype(int)
label_col = "state"
feature_cols = ["time_gap", "dist_f_cm", "delta", "velocity", "acceleration"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_windows(df, feature_cols, label_col, group_cols, window_size, step_size, pred_horizon):
    X, y, sids = [], [], []
    for group_key, group in df.groupby(group_cols):
        group = group.sort_values("time_ms").reset_index(drop=True)
        if len(group) < window_size + pred_horizon:
            continue
        sid = group_key if not isinstance(group_key, tuple) else group_key[0]
        for start in range(0, len(group) - window_size - pred_horizon + 1, step_size):
            end = start + window_size
            label_idx = end - 1 + pred_horizon
            X.append(group.iloc[start:end][feature_cols].values)
            y.append(group.iloc[label_idx][label_col])
            sids.append(sid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(sids)

def session_based_split(X, y, session_ids, train_ratio=0.70, val_ratio=0.15, seed=42):
    unique = np.unique(session_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n = len(unique)

    train_end = max(1, int(train_ratio * n))
    val_end   = max(train_end + 1, int((train_ratio + val_ratio) * n))
    if val_end >= n:
        val_end = n - 1
    if train_end >= val_end:
        train_end = val_end - 1

    train_s = set(unique[:train_end])
    val_s   = set(unique[train_end:val_end])
    test_s  = set(unique[val_end:])

    train_mask = np.isin(session_ids, list(train_s))
    val_mask   = np.isin(session_ids, list(val_s))
    test_mask  = np.isin(session_ids, list(test_s))

    return (X[train_mask], y[train_mask], X[val_mask], y[val_mask], X[test_mask], y[test_mask])

# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class MLPModel(nn.Module):
    def __init__(self, input_size=75, hidden_size=64, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)

def delta_encoder(x, threshold):
    batch_size, time_steps, features = x.shape
    spikes = torch.zeros((batch_size, time_steps, features * 2), device=x.device)
    reference = x[:, 0, :]
    for t in range(1, time_steps):
        current_val = x[:, t, :]
        diff = current_val - reference
        on_spikes  = (diff >=  threshold).float()
        off_spikes = (diff <= -threshold).float()
        spike_mask = torch.logical_or(on_spikes > 0, off_spikes > 0).float()
        reference  = reference * (1 - spike_mask) + current_val * spike_mask
        spikes[:, t, :features] = on_spikes
        spikes[:, t, features:] = off_spikes
    return spikes

class EventDrivenSNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=2, beta=0.9, encode_threshold=0.5):
        super().__init__()
        self.encode_threshold = encode_threshold
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        spike_grad = surrogate.fast_sigmoid()
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        spk_in = delta_encoder(x, self.encode_threshold)
        time_steps = spk_in.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem3_rec = []
        for t in range(time_steps):
            cur1 = self.fc1(spk_in[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            mem3_rec.append(mem3)
        return torch.stack(mem3_rec, dim=0)

def evaluate_pytorch_model(model, loader, is_snn=False):
    model.eval()
    y_true, y_pred, times = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            start = time.time()
            outputs = model(X_batch)
            if is_snn: outputs = outputs[-1]
            inf_time = (time.time() - start) * 1000
            times.append(inf_time / len(X_batch))
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())
            
    rep = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    danger_recall = rep.get('1', rep.get(1, {})).get('recall', 0.0)
    danger_f1 = rep.get('1', rep.get(1, {})).get('f1-score', 0.0)
    
    return acc, danger_recall, danger_f1, np.mean(times)

# ═══════════════════════════════════════════════════════════════════════
# 3. PIPELINE
# ═══════════════════════════════════════════════════════════════════════
WINDOW_SIZE = 15
STEP_SIZE = 5
EPOCHS = 20
BATCH_SIZE = 64

def run_pipeline(pred_horizon):
    print(f"\n{'='*70}")
    print(f"  HORIZON = {pred_horizon}")
    print(f"{'='*70}\n")
    
    X, y, sids = build_windows(df, feature_cols, label_col, group_cols, WINDOW_SIZE, STEP_SIZE, pred_horizon)
    X_tr, y_tr, X_va, y_va, X_te, y_te = session_based_split(X, y, sids)
    
    n_tr, t_dim, f_dim = X_tr.shape
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, f_dim)).reshape(n_tr, t_dim, f_dim)
    X_te_s = scaler.transform(X_te.reshape(-1, f_dim)).reshape(X_te.shape)

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t)

    mk = lambda x, y, shuf: DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=shuf
    )
    train_dl = mk(X_tr_s, y_tr, True)
    test_dl  = mk(X_te_s, y_te, False)

    print(f"  Train: {len(X_tr):,} windows | Val: {len(X_va):,} windows | Test: {len(X_te):,} windows")
    print(f"  Total generated: {len(X_tr) + len(X_va) + len(X_te):,} windows")

    # --- 1. Random Forest --- 
    print("\nTraining Random Forest...")
    rf_start = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1)
    rf_model.fit(X_tr_s.reshape(len(X_tr_s), -1), y_tr)
    rf_train_time = time.time() - rf_start
    rf_start = time.time()
    rf_pred = rf_model.predict(X_te_s.reshape(len(X_te_s), -1))
    rf_inf_time = (time.time() - rf_start) / len(X_te_s) * 1000
    rf_acc = accuracy_score(y_te, rf_pred)
    rf_rep = classification_report(y_te, rf_pred, output_dict=True)
    rf_rec = rf_rep.get('1', rf_rep.get(1, {})).get('recall', 0.0)
    rf_f1  = rf_rep.get('1', rf_rep.get(1, {})).get('f1-score', 0.0)
    rf_params = sum([tree.tree_.node_count for tree in rf_model.estimators_])

    # --- 2. LSTM ---
    lstm_model = LSTMModel(input_size=len(feature_cols), hidden_size=32, num_layers=2, output_size=2).to(device)
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print("Training LSTM...")
    opt = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    lstm_start = time.time()
    for ep in range(EPOCHS):
        lstm_model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(lstm_model(Xb), yb)
            loss.backward()
            opt.step()
    lstm_train_time = time.time() - lstm_start
    lstm_acc, lstm_rec, lstm_f1, lstm_inf_time = evaluate_pytorch_model(lstm_model, test_dl)

    # --- 3. MLP ---
    mlp_model = MLPModel(input_size=WINDOW_SIZE*len(feature_cols), hidden_size=64, output_size=2).to(device)
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print("Training MLP...")
    opt = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    mlp_start = time.time()
    for ep in range(EPOCHS):
        mlp_model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(mlp_model(Xb), yb)
            loss.backward()
            opt.step()
    mlp_train_time = time.time() - mlp_start
    mlp_acc, mlp_rec, mlp_f1, mlp_inf_time = evaluate_pytorch_model(mlp_model, test_dl)

    # --- 4. SNN ---
    snn_model = EventDrivenSNN(input_size=len(feature_cols), hidden_size=32, output_size=2).to(device)
    snn_params = sum(p.numel() for p in snn_model.parameters())
    print("Training SNN...")
    opt = torch.optim.Adam(snn_model.parameters(), lr=1e-3)
    snn_start = time.time()
    for ep in range(EPOCHS):
        snn_model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(snn_model(Xb)[-1], yb)
            loss.backward()
            opt.step()
    snn_train_time = time.time() - snn_start
    snn_acc, snn_rec, snn_f1, snn_inf_time = evaluate_pytorch_model(snn_model, test_dl, is_snn=True)

    print("\n  COMPARISON SUMMARY")
    print("| Model         | Accuracy | Danger Recall | Danger F1 | Parameters | Inf Time (ms) | Train Time (s) |")
    print("|---------------|----------|---------------|-----------|------------|---------------|----------------|")
    print(f"| Random Forest | {rf_acc*100:>6.2f}%  | {rf_rec*100:>12.2f}% | {rf_f1*100:>8.2f}% | {rf_params:>10,} | {rf_inf_time:>13.3f} | {rf_train_time:>14.1f} |")
    print(f"| LSTM          | {lstm_acc*100:>6.2f}%  | {lstm_rec*100:>12.2f}% | {lstm_f1*100:>8.2f}% | {lstm_params:>10,} | {lstm_inf_time:>13.3f} | {lstm_train_time:>14.1f} |")
    print(f"| MLP           | {mlp_acc*100:>6.2f}%  | {mlp_rec*100:>12.2f}% | {mlp_f1*100:>8.2f}% | {mlp_params:>10,} | {mlp_inf_time:>13.3f} | {mlp_train_time:>14.1f} |")
    print(f"| SNN (Ours)    | {snn_acc*100:>6.2f}%  | {snn_rec*100:>12.2f}% | {snn_f1*100:>8.2f}% | {snn_params:>10,} | {snn_inf_time:>13.3f} | {snn_train_time:>14.1f} |")
    print()

for h in [0, 5]:
    run_pipeline(h)

print("="*70)
print("  [DONE] BASELINE COMPARISON COMPLETE")
print("="*70)
