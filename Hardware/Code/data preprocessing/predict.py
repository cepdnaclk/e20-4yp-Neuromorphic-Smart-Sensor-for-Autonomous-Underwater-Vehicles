"""
Real-World SNN Obstacle Predictor
=================================
Load the trained Event-Driven SNN model and scaler, then predict whether
an obstacle (danger) is ahead of the AUV based on live ultrasonic sensor data.

Usage
-----
Stand-alone demo (uses dummy data):
    python predict.py

Integration into your AUV firmware / ROS node:
    from predict import ObstaclePredictor
    predictor = ObstaclePredictor("snn_obstacle_model.pth", "feature_scaler.pkl")
    label, confidence = predictor.predict(sensor_readings_dataframe)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

import snntorch as snn
from snntorch import surrogate


# ── 1. Delta encoder (must match training) ───────────────────────
def delta_encoder(x, threshold):
    """Send-on-Delta modulation: continuous → ON/OFF spikes."""
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


# ── 2. SNN model (must match training architecture) ──────────────
class EventDrivenSNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=2,
                 beta=0.9, encode_threshold=0.5):
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
        return torch.stack(mem3_rec, dim=0)  # (time_steps, batch, output_size)


# ── 3. High-level predictor class ────────────────────────────────
class ObstaclePredictor:
    """
    Wraps model loading, feature engineering, scaling, and inference
    in a single reusable object.

    Parameters
    ----------
    model_path : str
        Path to the saved checkpoint  (e.g. "snn_obstacle_model.pth").
    scaler_path : str
        Path to the saved StandardScaler (e.g. "feature_scaler.pkl").
    device : str, optional
        "cuda" or "cpu".  Defaults to auto-detect.
    """

    LABELS = {0: "Safe", 1: "Danger"}

    def __init__(self, model_path: str, scaler_path: str, device: str = None):
        # ── pick device ──
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── load checkpoint ──
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.feature_cols  = checkpoint["feature_cols"]
        self.window_size   = checkpoint["window_size"]
        self.pred_horizon  = checkpoint["pred_horizon"]

        # ── rebuild model from saved hyperparameters ──
        self.model = EventDrivenSNN(
            input_size       = checkpoint["input_size"],
            hidden_size      = checkpoint["hidden_size"],
            output_size      = checkpoint["output_size"],
            beta             = checkpoint["beta"],
            encode_threshold = checkpoint["encode_threshold"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # ── load scaler ──
        self.scaler = joblib.load(scaler_path)

        print(f"[ObstaclePredictor] Model loaded  (pred_horizon={self.pred_horizon})")
        print(f"[ObstaclePredictor] Features     : {self.feature_cols}")
        print(f"[ObstaclePredictor] Window size  : {self.window_size}")
        print(f"[ObstaclePredictor] Device       : {self.device}")

    # ──────────────────────────────────────────────────────────────
    def predict(self, window_df: pd.DataFrame):
        """
        Predict the obstacle state for a single window of sensor data.

        Parameters
        ----------
        window_df : pd.DataFrame
            A DataFrame with at least `self.window_size` rows and columns
            matching `self.feature_cols`. Typically you maintain a rolling
            buffer of recent sensor readings.

        Returns
        -------
        label : str
            "Safe" or "Danger"
        confidence : float
            Softmax probability of the predicted class  (0–1).
        """
        # take the last `window_size` rows
        window = window_df[self.feature_cols].tail(self.window_size).values

        if len(window) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} rows, got {len(window)}"
            )

        # scale (flatten → scale → reshape, same as during training)
        n_features = len(self.feature_cols)
        window_scaled = self.scaler.transform(
            window.reshape(-1, n_features)
        ).reshape(1, self.window_size, n_features)   # batch=1

        # to tensor
        x = torch.tensor(window_scaled, dtype=torch.float32).to(self.device)

        # forward pass
        with torch.no_grad():
            mem_out = self.model(x)          # (time_steps, 1, 2)
            logits  = mem_out[-1]            # last time-step  → (1, 2)
            probs   = torch.softmax(logits, dim=1)
            pred    = torch.argmax(probs, dim=1).item()
            conf    = probs[0, pred].item()

        return self.LABELS[pred], conf


# ── 4. Feature engineering helper ─────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the same feature engineering used during training.
    Call this on your raw sensor dataframe BEFORE passing a window
    to the predictor.

    Expected raw columns (at minimum):
        time_ms, dist_f_cm, baseline_cm, session_id (optional)
    """
    df = df.copy()

    # group column
    group_cols = []
    if "session_id" in df.columns:
        group_cols.append("session_id")

    # delta
    df["delta"] = df["dist_f_cm"] - df["baseline_cm"]

    # time gap
    if group_cols:
        df["time_gap"] = df.groupby(group_cols)["time_ms"].diff()
    else:
        df["time_gap"] = df["time_ms"].diff()
    df = df.dropna(subset=["time_gap"])

    # guard zero gaps
    df["time_gap"] = df["time_gap"].replace(0, np.nan)
    df = df.dropna(subset=["time_gap"])

    # velocity & acceleration
    if group_cols:
        df["velocity"]     = df.groupby(group_cols)["dist_f_cm"].diff() / df["time_gap"]
        df["acceleration"] = df.groupby(group_cols)["velocity"].diff()  / df["time_gap"]
    else:
        df["velocity"]     = df["dist_f_cm"].diff() / df["time_gap"]
        df["acceleration"] = df["velocity"].diff()  / df["time_gap"]
    df = df.dropna(subset=["velocity", "acceleration"])

    return df


# ── 5. Demo / standalone usage ───────────────────────────────────
if __name__ == "__main__":
    import os, sys

    MODEL_PATH  = "snn_obstacle_model.pth"
    SCALER_PATH = "feature_scaler.pkl"
    DATA_PATH   = "sensor.csv"

    # Check files exist
    for fpath in (MODEL_PATH, SCALER_PATH, DATA_PATH):
        if not os.path.isfile(fpath):
            print(f"ERROR: '{fpath}' not found.  Run the training notebook first.")
            sys.exit(1)

    # 1. Load the predictor
    predictor = ObstaclePredictor(MODEL_PATH, SCALER_PATH)

    # 2. Load and clean raw sensor data (same cleaning as training)
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df[df["valid"] == 1].copy()
    df = df[df["baseline_cm"] > 0].copy()
    df = df[(df["dist_f_cm"] > 20) & (df["dist_f_cm"] <= 600)].copy()

    # 3. Engineer features
    df = engineer_features(df)

    # 4. Grab a random window and predict
    window_size = predictor.window_size
    if len(df) < window_size:
        print("Not enough data for even one window!")
        sys.exit(1)

    # pick a random starting point
    np.random.seed(42)
    start = np.random.randint(0, len(df) - window_size)
    window_df = df.iloc[start: start + window_size]

    label, confidence = predictor.predict(window_df)
    print(f"\n{'='*50}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.4f}")
    print(f"{'='*50}")

    # ── Real-world loop example ──
    print("\n--- Example: Simulated real-time prediction loop ---")
    # In an actual AUV, you'd read from the ultrasonic sensor in a loop.
    # Here we simulate it by sliding through the dataset.

    STEP = 5  # advance 5 readings per iteration (matches training step_size)
    for i in range(0, min(50, len(df) - window_size), STEP):
        window_df = df.iloc[i: i + window_size]
        label, conf = predictor.predict(window_df)
        time_ms = window_df["time_ms"].iloc[-1]
        print(f"  t={time_ms:>10} ms  ->  {label:<6s}  (confidence: {conf:.4f})")
