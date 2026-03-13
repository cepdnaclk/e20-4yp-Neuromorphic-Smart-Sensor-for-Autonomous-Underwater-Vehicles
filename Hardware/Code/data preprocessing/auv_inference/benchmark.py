"""
Runs the full inference pipeline on sensor.csv and reports:
  - Classification accuracy vs ground truth event labels
  - Agreement rate with firmware heuristic danger score
  - Latency percentiles (important for Jetson demo)
"""

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from snn_inference import SNNInferenceEngine, SensorReading


def run_benchmark(model_path: str, scaler_path: str,
                  csv_path: str, sensor_id: int = 1) -> None:

    print("\n" + "═"*60)
    print("  INFERENCE BENCHMARK")
    print(f"  Model  : {model_path}")
    print(f"  Sensor : {sensor_id} (front)")
    print("═"*60 + "\n")

    engine = SNNInferenceEngine(model_path, scaler_path,
                                target_sensor_id=sensor_id)

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    # Keep only target sensor — matches how sensor.csv was built
    df = df[df["sensor_id"] == sensor_id].copy()
    df = df.sort_values("time_ms").reset_index(drop=True)

    print(f"  Rows for sensor {sensor_id}: {len(df)}")

    # Ground truth: mirrors training exactly
    # df["state"] = (df["danger"] > 0).astype(int)
    # event is 0 or 1 only (1 = danger entered)
    df["gt"] = ((df["danger"] > 0) | (df["event"] == 1)).astype(int)

    y_true, y_pred      = [], []
    fw_agree            = []
    n_skipped           = 0
    t_start             = time.perf_counter()

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

        result = engine.push_reading(r)
        if result is None:
            n_skipped += 1
            continue

        gt_label   = int(row["gt"])
        snn_label  = 1 if result.prediction == "DANGER" else 0
        fw_label   = 1 if result.firmware_danger > 0 else 0

        y_true.append(gt_label)
        y_pred.append(snn_label)
        fw_agree.append(int(snn_label == fw_label))

    elapsed = time.perf_counter() - t_start

    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    fw_agree = np.array(fw_agree)

    print(f"\n  ── Throughput ──────────────────────────")
    print(f"  Rows total        : {len(df)}")
    print(f"  Buffer warm-up    : {n_skipped} skipped")
    print(f"  Inferences        : {len(y_true)}")
    print(f"  Total time        : {elapsed:.3f} s")
    print(f"  Throughput        : {len(y_true)/elapsed:.1f} inf/s")

    s = engine.stats()
    print(f"\n  ── Latency (ms) ────────────────────────")
    lats = list(engine._latencies)
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p:>2}  : {np.percentile(lats, p):.3f} ms")
    print(f"  max  : {s['max_lat_ms']:.3f} ms")

    print(f"\n  ── Accuracy vs ground truth ────────────")
    print(classification_report(y_true, y_pred,
                                target_names=["Safe", "Danger"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                  Pred Safe   Pred Danger")
    print(f"  True Safe   :  {cm[0,0]:>9d}   {cm[0,1]:>11d}")
    print(f"  True Danger :  {cm[1,0]:>9d}   {cm[1,1]:>11d}")

    print(f"\n  ── SNN vs Firmware heuristic ───────────")
    print(f"  Agreement rate : {fw_agree.mean()*100:.2f}%")
    print(f"  (% of frames where SNN and firmware agree on Safe/Danger)")

    print(f"\n  ✓ Benchmark complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="snn_obstacle_model.pth")
    parser.add_argument("--scaler", default="feature_scaler.pkl")
    parser.add_argument("--csv",    default="sensor.csv")
    parser.add_argument("--sensor", default=1, type=int)
    args = parser.parse_args()
    run_benchmark(args.model, args.scaler, args.csv, args.sensor)
