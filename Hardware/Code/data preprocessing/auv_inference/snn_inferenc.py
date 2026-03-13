"""
============================================================
  Neuromorphic Smart Sensor — Real-Time Inference Engine
  Jetson Nano / Orin Nano
  Group 22 | CO422 Final Year Project
============================================================

Reads live UART from the ESP32 (4x JSN-SR04T firmware),
reconstructs the EXACT feature pipeline used in training,
runs the EventDrivenSNN, and outputs navigation decisions.

Actual firmware CSV format (11 fields):
  time_ms, sensor_id, echo_us, valid, dist_cm, dist_f_cm,
  baseline_cm, enter_thr_cm, exit_thr_cm, danger, event

Usage:
  # Demo mode — replay sensor.csv (no hardware needed)
  python3 snn_inference.py --demo --speed 3.0

  # Live mode — ESP32 plugged into Jetson
  python3 snn_inference.py --port /dev/ttyUSB0 --baud 115200

  # Specific sensor only (front = sensor_id 1)
  python3 snn_inference.py --port /dev/ttyUSB0 --sensor 1
"""

import argparse
import time
import sys
import os
CLEAR = "cls" if os.name == "nt" else "clear"
import threading
import queue
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import joblib

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SNN")


# ══════════════════════════════════════════════════════════════════
#  1.  SNN MODEL — exact copy from pre_future_new.ipynb
# ══════════════════════════════════════════════════════════════════

def delta_encoder(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Send-on-Delta encoding (matches training notebook exactly).
    Input  : (batch, time_steps, n_features)
    Output : (batch, time_steps, n_features*2)  ON/OFF spike pairs
    """
    batch_size, time_steps, features = x.shape
    spikes    = torch.zeros((batch_size, time_steps, features * 2), device=x.device)
    reference = x[:, 0, :].clone()

    for t in range(1, time_steps):
        current_val = x[:, t, :]
        diff        = current_val - reference
        on_spikes   = (diff >=  threshold).float()
        off_spikes  = (diff <= -threshold).float()
        spike_mask  = torch.logical_or(on_spikes > 0, off_spikes > 0).float()
        reference   = reference * (1 - spike_mask) + current_val * spike_mask
        spikes[:, t, :features] = on_spikes
        spikes[:, t, features:] = off_spikes

    return spikes


class EventDrivenSNN(nn.Module):
    """
    3-layer FC-SNN with Leaky Integrate-and-Fire neurons.
    Architecture matches training exactly:
      input_size=5, hidden_size=32, output_size=2, beta=0.9
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=2,
                 beta=0.9, encode_threshold=0.5):
        super().__init__()
        self.encode_threshold = encode_threshold
        spike_grad = surrogate.fast_sigmoid()

        self.fc1  = nn.Linear(input_size * 2, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3  = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spk_in     = delta_encoder(x, self.encode_threshold)
        time_steps = spk_in.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem3_rec = []
        for t in range(time_steps):
            cur1       = self.fc1(spk_in[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3       = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            mem3_rec.append(mem3)
        return torch.stack(mem3_rec, dim=0)   # (T, batch, 2)


# ══════════════════════════════════════════════════════════════════
#  2.  SENSOR READING — matches actual firmware CSV exactly
# ══════════════════════════════════════════════════════════════════

@dataclass
class SensorReading:
    """
    One parsed line from the ESP32 firmware.

    Firmware CSV column order:
      time_ms, sensor_id, echo_us, valid, dist_cm, dist_f_cm,
      baseline_cm, enter_thr_cm, exit_thr_cm, danger, event
    """
    time_ms:      float
    sensor_id:    int
    echo_us:      float
    valid:        int
    dist_cm:      float
    dist_f_cm:    float     # EMA-filtered (alpha=0.25, with drop-reject)
    baseline_cm:  float     # median of 25 stability-gated samples
    enter_thr_cm: float     # baseline - 60 cm
    exit_thr_cm:  float     # enter_thr + 20 cm
    danger:       float     # 0.0 – 1.0 gradient score from firmware
    event:        int       # 0=no event, 1=danger entered


def parse_uart_line(line: str) -> Optional[SensorReading]:
    """
    Parse one CSV line from the ESP32 UART stream.

    Expected 11 fields (firmware order):
      time_ms, sensor_id, echo_us, valid, dist_cm, dist_f_cm,
      baseline_cm, enter_thr_cm, exit_thr_cm, danger, event
    """
    try:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("E") \
                    or line.startswith("W") or line.startswith("-") \
                    or line.startswith("B") or line.startswith("C"):
            return None   # skip comment/warning/status lines
        parts = line.split(",")
        if len(parts) != 11:
            return None
        return SensorReading(
            time_ms      = float(parts[0]),
            sensor_id    = int(parts[1]),
            echo_us      = float(parts[2]),
            valid        = int(parts[3]),
            dist_cm      = float(parts[4]),
            dist_f_cm    = float(parts[5]),
            baseline_cm  = float(parts[6]),
            enter_thr_cm = float(parts[7]),
            exit_thr_cm  = float(parts[8]),
            danger       = float(parts[9]),

        )
    except (ValueError, IndexError):
        return None


# ══════════════════════════════════════════════════════════════════
#  3.  FEATURE PIPELINE — mirrors training notebook exactly
#
#  Features (same 5 as training):
#    time_gap     : ms between consecutive valid readings
#    dist_f_cm    : firmware EMA-filtered distance
#    delta        : dist_f_cm - baseline_cm
#    velocity     : Δdist_f_cm / Δtime_ms  (cm/ms)
#    acceleration : Δvelocity  / Δtime_ms  (cm/ms²)
# ══════════════════════════════════════════════════════════════════

FEATURE_COLS  = ["time_gap", "dist_f_cm", "delta", "velocity", "acceleration"]
WINDOW_SIZE   = 15    # must match training WINDOW_SIZE
ENCODE_THRESH = 0.5   # must match training encode_threshold


class FeaturePipeline:
    """
    Rolling buffer of valid SensorReadings.
    Computes the same 5 features as the training notebook on demand.

    Notes:
    - Only readings where baseline_cm > 0 are accepted
      (mirrors training: df[df['baseline_cm'] > 0])
    - Only readings where 20 < dist_f_cm <= 600 are accepted
      (mirrors training clamp)
    - Requires window_size + 2 readings for one full window
      (+2 because velocity/acceleration need two diffs)
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._buf: deque = deque(maxlen=window_size + 2)

    def push(self, r: SensorReading) -> bool:
        """
        Accept a reading into the buffer if it passes quality gates.
        Returns True if accepted.
        """
        if r.valid != 1:
            return False
        if r.baseline_cm <= 0:
            return False
        if not (20.0 < r.dist_f_cm <= 600.0):
            return False
        self._buf.append(r)
        return True

    def ready(self) -> bool:
        return len(self._buf) >= self.window_size + 2

    def extract(self) -> Optional[np.ndarray]:
        """
        Build (WINDOW_SIZE, 5) feature array from the current buffer.
        Returns None if buffer not full.
        """
        if not self.ready():
            return None

        readings = list(self._buf)[-(self.window_size + 2):]

        time_gaps, dist_f_vals, deltas, velocities, accelerations = [], [], [], [], []

        prev_time = readings[0].time_ms
        prev_dist = readings[0].dist_f_cm
        prev_vel  = 0.0

        for i in range(1, len(readings)):
            r  = readings[i]
            dt = r.time_ms - prev_time
            if dt <= 0:
                dt = 1.0   # guard against duplicate timestamps
            dd  = r.dist_f_cm - prev_dist
            vel = dd / dt
            acc = (vel - prev_vel) / dt

            time_gaps.append(dt)
            dist_f_vals.append(r.dist_f_cm)
            deltas.append(r.dist_f_cm - r.baseline_cm)
            velocities.append(vel)
            accelerations.append(acc)

            prev_time = r.time_ms
            prev_dist = r.dist_f_cm
            prev_vel  = vel

        features = np.column_stack([
            time_gaps[-self.window_size:],
            dist_f_vals[-self.window_size:],
            deltas[-self.window_size:],
            velocities[-self.window_size:],
            accelerations[-self.window_size:],
        ]).astype(np.float32)

        return features   # (WINDOW_SIZE, 5)


# ══════════════════════════════════════════════════════════════════
#  4.  INFERENCE RESULT
# ══════════════════════════════════════════════════════════════════

@dataclass
class InferenceResult:
    timestamp_ms:    float
    sensor_id:       int
    dist_f_cm:       float
    baseline_cm:     float
    firmware_danger: float    # raw danger score from ESP32
    snn_danger_prob: float    # SNN softmax P(danger)
    prediction:      str      # "SAFE" | "DANGER"
    confidence:      float    # max(P(safe), P(danger))
    latency_ms:      float
    nav_command:     str      # "FORWARD"|"CAUTION"|"STOP"|"REVERSE"


# ══════════════════════════════════════════════════════════════════
#  5.  INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════

class SNNInferenceEngine:
    """
    Loads the trained checkpoint, holds one FeaturePipeline per sensor,
    and runs inference whenever a new window is complete.
    """

    DANGER_THRESHOLD = 0.55   # SNN softmax P(danger) above this → DANGER

    def __init__(self, model_path: str, scaler_path: str,
                 target_sensor_id: int = 1,
                 device: Optional[str] = None):

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        log.info(f"Inference device : {self.device}")

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        log.info(f"Model loaded     : {model_path}")
        log.info(f"  Train accuracy : {ckpt.get('test_accuracy', '?'):.4f}")
        log.info(f"  Window size    : {ckpt.get('window_size', WINDOW_SIZE)}")
        log.info(f"  Pred horizon   : {ckpt.get('pred_horizon', 0)}")
        log.info(f"  Features       : {ckpt.get('feature_cols', FEATURE_COLS)}")

        self.model = EventDrivenSNN(
            input_size       = ckpt["input_size"],
            hidden_size      = ckpt["hidden_size"],
            output_size      = ckpt["output_size"],
            beta             = ckpt["beta"],
            encode_threshold = ckpt.get("encode_threshold", ENCODE_THRESH),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.window_size       = ckpt.get("window_size", WINDOW_SIZE)
        self.target_sensor_id  = target_sensor_id

        # Scaler
        self.scaler = joblib.load(scaler_path)
        log.info(f"Scaler loaded    : {scaler_path}")

        # One pipeline per sensor (firmware cycles through 1,2,3,5)
        self.pipelines = {
            sid: FeaturePipeline(self.window_size)
            for sid in [1, 2, 3, 5]
        }

        # Stats
        self._n_total   = 0
        self._n_danger  = 0
        self._latencies: deque = deque(maxlen=200)

    # ── Public ──────────────────────────────────────────────────

    def push_reading(self, r: SensorReading) -> Optional[InferenceResult]:
        """
        Feed one reading from the UART stream.
        Returns InferenceResult if target sensor window is ready, else None.
        """
        # Route to the right pipeline
        pipeline = self.pipelines.get(r.sensor_id)
        if pipeline is None:
            return None

        accepted = pipeline.push(r)

        # Only run inference for the target (front) sensor
        if r.sensor_id != self.target_sensor_id:
            return None

        if not accepted or not pipeline.ready():
            return None

        return self._infer(r, pipeline)

    # ── Private ─────────────────────────────────────────────────

    def _infer(self, latest: SensorReading,
               pipeline: FeaturePipeline) -> InferenceResult:

        t0 = time.perf_counter()

        # 1. Extract features
        features = pipeline.extract()           # (W, 5)
        w, f = features.shape

        # 2. Normalize — same scaler fitted on training data
        features_scaled = self.scaler.transform(
            features.reshape(-1, f)
        ).reshape(w, f).astype(np.float32)

        # 3. To tensor (batch=1)
        x = torch.tensor(features_scaled) \
                  .unsqueeze(0).to(self.device)  # (1, W, 5)

        # 4. SNN forward pass
        with torch.no_grad():
            mem_out    = self.model(x)           # (W, 1, 2)
            logits     = mem_out[-1, 0, :]       # last timestep logits
            probs      = torch.softmax(logits, dim=0)
            danger_p   = probs[1].item()
            pred_cls   = int(danger_p >= self.DANGER_THRESHOLD)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies.append(latency_ms)
        self._n_total += 1
        if pred_cls == 1:
            self._n_danger += 1

        prediction  = "DANGER" if pred_cls == 1 else "SAFE"
        nav_command = self._nav_command(pred_cls, danger_p,
                                        latest.dist_f_cm,
                                        latest.baseline_cm)

        return InferenceResult(
            timestamp_ms    = latest.time_ms,
            sensor_id       = latest.sensor_id,
            dist_f_cm       = latest.dist_f_cm,
            baseline_cm     = latest.baseline_cm,
            firmware_danger = latest.danger,
            snn_danger_prob = danger_p,
            prediction      = prediction,
            confidence      = max(danger_p, 1.0 - danger_p),
            latency_ms      = latency_ms,
            nav_command     = nav_command,
        )

    @staticmethod
    def _nav_command(pred_cls: int, danger_p: float,
                     dist_f: float, baseline: float) -> str:
        """
        Layered navigation logic on top of SNN output.

        SNN says SAFE:
          danger_p > 0.40  → CAUTION   (borderline, slow down)
          else             → FORWARD
        SNN says DANGER:
          dist < 35% of baseline  → REVERSE  (very close)
          else                    → STOP
        """
        if pred_cls == 0:
            return "CAUTION" if danger_p > 0.40 else "FORWARD"
        else:
            return "REVERSE" if (baseline > 0 and dist_f < baseline * 0.35) else "STOP"

    def stats(self) -> dict:
        lats = list(self._latencies)
        return {
            "total":        self._n_total,
            "danger_count": self._n_danger,
            "danger_rate":  self._n_danger / max(1, self._n_total),
            "avg_lat_ms":   float(np.mean(lats)) if lats else 0.0,
            "max_lat_ms":   float(np.max(lats))  if lats else 0.0,
        }

    def buf_size(self) -> int:
        return len(self.pipelines[self.target_sensor_id]._buf)

    def buf_needed(self) -> int:
        return self.window_size + 2


# ══════════════════════════════════════════════════════════════════
#  6.  UART READER THREAD
# ══════════════════════════════════════════════════════════════════

def uart_reader_thread(port: str, baud: int,
                       out_q: queue.Queue,
                       stop_ev: threading.Event) -> None:
    if not SERIAL_AVAILABLE:
        log.error("pyserial not installed: pip install pyserial")
        return
    while not stop_ev.is_set():
        try:
            log.info(f"Opening {port} @ {baud} baud …")
            with serial.Serial(port, baud, timeout=2.0) as ser:
                log.info("Serial port open — waiting for data …")
                while not stop_ev.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    r = parse_uart_line(line)
                    if r:
                        out_q.put(r)
        except serial.SerialException as e:
            log.warning(f"Serial error: {e}  — retrying in 2 s …")
            time.sleep(2.0)


# ══════════════════════════════════════════════════════════════════
#  7.  TERMINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════

C = {
    "green":  "\033[92m", "yellow": "\033[93m",
    "red":    "\033[91m", "mag":    "\033[95m",
    "cyan":   "\033[96m", "bold":   "\033[1m",
    "dim":    "\033[2m",  "reset":  "\033[0m",
}

NAV_COLOR = {
    "FORWARD": C["green"], "CAUTION": C["yellow"],
    "STOP":    C["red"],   "REVERSE": C["mag"],
}
PRED_COLOR = {"SAFE": C["green"], "DANGER": C["red"]}


def danger_bar(p: float, width: int = 32) -> str:
    filled = int(p * width)
    bar    = "█" * filled + "░" * (width - filled)
    col    = C["red"] if p > 0.7 else C["yellow"] if p > 0.4 else C["green"]
    return f"{col}{bar}{C['reset']}"


def render(result: InferenceResult, stats: dict,
           buf: int, buf_needed: int) -> None:
    os.system(CLEAR)
    B  = C["bold"];  R = C["reset"];  D = C["dim"]
    CY = C["cyan"]

    pred_c = PRED_COLOR.get(result.prediction, "")
    nav_c  = NAV_COLOR.get(result.nav_command, "")

    # Agreement between firmware heuristic and SNN
    fw_pred  = "DANGER" if result.firmware_danger > 0 else "SAFE"
    agree    = "✓ agree" if fw_pred == result.prediction else "✗ differ"
    ag_color = C["green"] if fw_pred == result.prediction else C["yellow"]

    print(f"{B}{CY}{'═'*60}{R}")
    print(f"{B}{CY}   NEUROMORPHIC AUV — REAL-TIME SNN OBSTACLE DETECTION{R}")
    print(f"{B}{CY}   Group 22 | CO422 | Sensor {result.sensor_id}{R}")
    print(f"{B}{CY}{'═'*60}{R}\n")

    print(f"  {D}Timestamp    {R}: {result.timestamp_ms/1000:>8.2f} s")
    print(f"  {D}Distance     {R}: {result.dist_f_cm:>7.1f} cm"
          f"  (baseline {result.baseline_cm:.1f} cm,"
          f"  Δ {result.dist_f_cm - result.baseline_cm:+.1f} cm)")
    print()
    print(f"  SNN Danger   : {danger_bar(result.snn_danger_prob)}"
          f"  {result.snn_danger_prob*100:5.1f}%")
    print(f"  FW  Danger   : {danger_bar(result.firmware_danger)}"
          f"  {result.firmware_danger*100:5.1f}%")
    print()
    print(f"  SNN Predict  : {pred_c}{B}{result.prediction:8s}{R}"
          f"  (confidence {result.confidence*100:.1f}%)")
    print(f"  FW  Predict  : {pred_c}{fw_pred:8s}"
          f"  {ag_color}{agree}{R}")
    print()
    print(f"  {B}NAV COMMAND  : {nav_c}{B}{result.nav_command}{R}")
    print()
    print(f"{'─'*60}")
    print(f"  {D}Inference latency : {result.latency_ms:.2f} ms{R}")
    print(f"  {D}Avg latency (200) : {stats['avg_lat_ms']:.2f} ms{R}")
    print(f"  {D}Max latency (200) : {stats['max_lat_ms']:.2f} ms{R}")
    print(f"  {D}Total inferences  : {stats['total']}{R}")
    print(f"  {D}Danger rate       : {stats['danger_rate']*100:.1f}%{R}")
    print(f"  {D}Buffer fill       : {min(buf, buf_needed)}/{buf_needed}{R}")
    print(f"{'─'*60}")
    print(f"  {D}Press Ctrl+C to stop{R}")


def render_warmup(buf: int, buf_needed: int) -> None:
    os.system(CLEAR)
    pct = min(buf / buf_needed, 1.0)
    bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
    print(f"\n  {C['cyan']}{C['bold']}Warming up baseline …{C['reset']}")
    print(f"\n  [{C['yellow']}{bar}{C['reset']}]  {buf}/{buf_needed} readings\n")
    print(f"  {C['dim']}Waiting for baseline lock on ESP32 (needs ~25 stable pings){C['reset']}\n")


# ══════════════════════════════════════════════════════════════════
#  8.  DEMO MODE — replay sensor.csv
# ══════════════════════════════════════════════════════════════════

def run_demo(model_path: str, scaler_path: str, csv_path: str,
             speed: float = 3.0, sensor_id: int = 1) -> None:
    import pandas as pd

    log.info("═"*60)
    log.info("  DEMO MODE — replaying sensor.csv")
    log.info(f"  Speed: {speed}x  |  Target sensor: {sensor_id}")
    log.info("═"*60)

    engine = SNNInferenceEngine(model_path, scaler_path,
                                target_sensor_id=sensor_id)

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.sort_values("time_ms").reset_index(drop=True)

    prev_time_ms = None

    for _, row in df.iterrows():
        r = SensorReading(
            time_ms      = float(row["time_ms"]),
            sensor_id    = int(row["sensor_id"]),
            echo_us      = float(row.get("echo_us", 0)),
            valid        = int(row["valid"]),
            dist_cm      = float(row["dist_cm"]),
            dist_f_cm    = float(row["dist_f_cm"]),
            baseline_cm  = float(row["baseline_cm"]),
            enter_thr_cm = float(row["enter_thr_cm"]),
            exit_thr_cm  = float(row["exit_thr_cm"]),
            danger       = float(row["danger"]),
            event        = int(row["event"]),
        )

        # Pace playback to match real timing
        if speed > 0 and prev_time_ms is not None:
            gap_s = (r.time_ms - prev_time_ms) / 1000.0 / speed
            if 0 < gap_s < 2.0:
                time.sleep(gap_s)
        prev_time_ms = r.time_ms

        result = engine.push_reading(r)
        buf    = engine.buf_size()
        need   = engine.buf_needed()

        if result is None:
            render_warmup(buf, need)
        else:
            render(result, engine.stats(), buf, need)

    log.info("Demo complete.")
    s = engine.stats()
    log.info(f"Final stats → {s}")


# ══════════════════════════════════════════════════════════════════
#  9.  LIVE MODE — real ESP32 over UART
# ══════════════════════════════════════════════════════════════════

def run_live(port: str, baud: int, model_path: str,
             scaler_path: str, sensor_id: int = 1) -> None:

    engine    = SNNInferenceEngine(model_path, scaler_path,
                                   target_sensor_id=sensor_id)
    data_q    = queue.Queue(maxsize=500)
    stop_ev   = threading.Event()

    reader = threading.Thread(
        target=uart_reader_thread,
        args=(port, baud, data_q, stop_ev),
        daemon=True,
    )
    reader.start()
    log.info(f"Listening on {port} @ {baud} baud …")
    log.info(f"Monitoring sensor ID: {sensor_id}")

    try:
        while True:
            try:
                r = data_q.get(timeout=5.0)
            except queue.Empty:
                log.warning("No data for 5 s — check ESP32 connection and baud rate.")
                continue

            result = engine.push_reading(r)
            buf    = engine.buf_size()
            need   = engine.buf_needed()

            if result is None:
                render_warmup(buf, need)
            else:
                render(result, engine.stats(), buf, need)

    except KeyboardInterrupt:
        log.info("Stopping …")
    finally:
        stop_ev.set()
        reader.join(timeout=2.0)
        log.info(f"Session stats: {engine.stats()}")


# ══════════════════════════════════════════════════════════════════
#  10.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Neuromorphic AUV — Real-Time SNN Inference"
    )
    parser.add_argument("--model",  default="snn_obstacle_model.pth",
                        help="Trained checkpoint (.pth)")
    parser.add_argument("--scaler", default="feature_scaler.pkl",
                        help="Fitted StandardScaler (.pkl)")
    parser.add_argument("--port",   default="/dev/ttyUSB0",
                        help="Serial port (e.g. /dev/ttyUSB0 on Jetson)")
    parser.add_argument("--baud",   default=115200, type=int)
    parser.add_argument("--sensor", default=1, type=int,
                        help="Sensor ID to monitor (1=front, 2=down, 3=right, 5=left)")
    parser.add_argument("--demo",   action="store_true",
                        help="Demo mode: replay sensor.csv (no hardware needed)")
    parser.add_argument("--csv",    default="sensor.csv",
                        help="CSV for demo mode")
    parser.add_argument("--speed",  default=3.0, type=float,
                        help="Demo replay speed multiplier (0=max speed)")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.model, args.scaler, args.csv,
                 speed=args.speed, sensor_id=args.sensor)
    else:
        run_live(args.port, args.baud, args.model, args.scaler,
                 sensor_id=args.sensor)


if __name__ == "__main__":
    main()
