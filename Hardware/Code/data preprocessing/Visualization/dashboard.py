"""
═══════════════════════════════════════════════════════════════════════
  Neuromorphic AUV — Live SNN Inference Dashboard
  Group 22 | CO422 Final Year Project

  REAL model inference: loads snn_obstacle_model.pth + feature_scaler.pkl
  Replays sensor.csv row-by-row, runs the SNN on each window,
  and shows SNN prediction vs ESP32 firmware heuristic side-by-side.

  Run:  streamlit run dashboard.py
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import time
import os

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import joblib

# ──────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuromorphic AUV — SNN Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

.main { background-color: #0a0f1e; }
[data-testid="stAppViewContainer"] { background: #0a0f1e; }
[data-testid="stSidebar"] { background: #0d1529; border-right: 1px solid #1e3a5f; }

.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.snn-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.metric-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin: 4px 0;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a7fa5;
    margin-bottom: 4px;
}
.agree-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.agree { background: #0f3d2a; color: #2ecc71; border: 1px solid #2ecc71; }
.disagree { background: #3d1010; color: #e74c3c; border: 1px solid #e74c3c; }
.caution { background: #3d2d00; color: #f39c12; border: 1px solid #f39c12; }

.nav-forward { color: #2ecc71; font-size: 1.8rem; font-weight: 800; }
.nav-caution { color: #f39c12; font-size: 1.8rem; font-weight: 800; }
.nav-stop    { color: #e74c3c; font-size: 1.8rem; font-weight: 800; }
.nav-reverse { color: #e74c3c; font-size: 1.8rem; font-weight: 800; animation: pulse 0.6s infinite; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

.spike-bar-container { display: flex; gap: 2px; align-items: flex-end; height: 40px; }
.spike-bar { width: 8px; border-radius: 2px 2px 0 0; transition: height 0.2s; }

.divider { border: none; border-top: 1px solid #1e3a5f; margin: 12px 0; }
.tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; 
       background: #0d1f3c; border: 1px solid #1e3a5f; border-radius: 4px;
       padding: 1px 6px; color: #4a7fa5; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
#  SNN MODEL — exact replica from training notebook
# ──────────────────────────────────────────────────────────────────────
def delta_encoder(x: torch.Tensor, threshold: float) -> torch.Tensor:
    batch_size, time_steps, features = x.shape
    spikes    = torch.zeros((batch_size, time_steps, features * 2))
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

    def forward(self, x: torch.Tensor):
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
        return torch.stack(mem3_rec, dim=0)  # (T, batch, 2)


# ──────────────────────────────────────────────────────────────────────
#  FEATURE PIPELINE — mirrors training exactly
# ──────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 15
FEATURE_COLS  = ["time_gap", "dist_f_cm", "delta", "velocity", "acceleration"]
DANGER_THRESH = 0.55


class FeatureBuffer:
    """Rolling buffer that computes the same 5 features as training."""
    def __init__(self):
        self._buf = deque(maxlen=WINDOW_SIZE + 2)

    def push(self, row):
        if row["valid"] != 1:           return False
        if row["baseline_cm"] <= 0:     return False
        if not (20 < row["dist_f_cm"] <= 600): return False
        self._buf.append(row)
        return True

    def ready(self):
        return len(self._buf) >= WINDOW_SIZE + 2

    def extract(self):
        if not self.ready():
            return None, None
        readings = list(self._buf)[-(WINDOW_SIZE + 2):]
        tg, df, dl, vl, ac = [], [], [], [], []
        prev_t = readings[0]["time_ms"]
        prev_d = readings[0]["dist_f_cm"]
        prev_v = 0.0
        for i in range(1, len(readings)):
            r  = readings[i]
            dt = max(r["time_ms"] - prev_t, 1.0)
            dd = r["dist_f_cm"] - prev_d
            v  = dd / dt
            a  = (v - prev_v) / dt
            tg.append(dt);            df.append(r["dist_f_cm"])
            dl.append(r["dist_f_cm"] - r["baseline_cm"])
            vl.append(v);             ac.append(a)
            prev_t = r["time_ms"]; prev_d = r["dist_f_cm"]; prev_v = v

        feats = np.column_stack([
            tg[-WINDOW_SIZE:], df[-WINDOW_SIZE:], dl[-WINDOW_SIZE:],
            vl[-WINDOW_SIZE:], ac[-WINDOW_SIZE:]
        ]).astype(np.float32)

        # spike pattern for display (count non-zero channels per timestep)
        spikes_display = []
        ref = feats[0].copy()
        for row in feats[1:]:
            diff = row - ref
            fired = (np.abs(diff) >= 0.5).astype(int)  # pre-scaling approx
            spikes_display.append(int(fired.sum()))
            ref[fired > 0] = row[fired > 0]
        spikes_display = [0] + spikes_display  # align length

        return feats, spikes_display

    def reset(self):
        self._buf.clear()


# ──────────────────────────────────────────────────────────────────────
#  LOAD MODEL + SCALER
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    try:
        ckpt   = torch.load(model_path, map_location="cpu", weights_only=False)
        model  = EventDrivenSNN(
            input_size       = ckpt.get("input_size", 5),
            hidden_size      = ckpt.get("hidden_size", 32),
            output_size      = ckpt.get("output_size", 2),
            beta             = ckpt.get("beta", 0.9),
            encode_threshold = ckpt.get("encode_threshold", 0.5),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        scaler = joblib.load(scaler_path)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df[df["valid"] == 1].copy()
    df = df[df["baseline_cm"] > 0].copy()
    df = df.sort_values(["session_id", "time_ms"]).reset_index(drop=True)
    return df


def run_snn_inference(model, scaler, feats_raw):
    """Run one SNN forward pass. Returns (danger_prob, prediction, latency_ms, spike_count)."""
    t0 = time.perf_counter()
    w, f = feats_raw.shape
    scaled = scaler.transform(feats_raw.reshape(-1, f)).reshape(w, f).astype(np.float32)
    x      = torch.tensor(scaled).unsqueeze(0)          # (1, W, 5)

    # count spikes after encoding
    with torch.no_grad():
        spk_in   = delta_encoder(x, model.encode_threshold)
        n_spikes = int(spk_in.sum().item())
        mem_out  = model(x)                              # (W, 1, 2)
        logits   = mem_out[-1, 0, :]
        probs    = torch.softmax(logits, dim=0)
        danger_p = float(probs[1].item())

    latency_ms = (time.perf_counter() - t0) * 1000
    prediction = "DANGER" if danger_p >= DANGER_THRESH else "SAFE"
    return danger_p, prediction, latency_ms, n_spikes


def nav_command(snn_pred, danger_p, dist_f, baseline):
    if snn_pred == "SAFE":
        return "CAUTION" if danger_p > 0.40 else "FORWARD"
    else:
        return "REVERSE" if (baseline > 0 and dist_f < baseline * 0.35) else "STOP"


# ──────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 SNN Dashboard")
    st.markdown('<span class="tag">CO422 · Group 22</span>', unsafe_allow_html=True)
    st.markdown("---")

    model_path  = st.text_input("Model path",  "snn_obstacle_model.pth")
    scaler_path = st.text_input("Scaler path", "feature_scaler.pkl")
    csv_path    = st.text_input("CSV path",    "sensor.csv")

    model, scaler, load_err = load_model_and_scaler(model_path, scaler_path)

    if load_err:
        st.error(f"Model load failed:\n{load_err}")
        st.info("Check that snn_obstacle_model.pth and feature_scaler.pkl are in the same folder as dashboard.py")
        st.stop()
    else:
        st.success("✓ Model loaded  (1,474 params)")
        st.success("✓ Scaler loaded")

    df = load_csv(csv_path)
    st.success(f"✓ CSV loaded  ({len(df):,} rows)")

    st.markdown("---")
    sessions = sorted(df["session_id"].unique())
    session_id = st.selectbox("Session", sessions, index=0)
    speed      = st.slider("Playback speed", 0.1, 10.0, 2.0, 0.1)
    auto_play  = st.checkbox("Auto-play", value=False)

    session_df = df[df["session_id"] == session_id].reset_index(drop=True)
    st.markdown(f"**{len(session_df)} readings · {session_df['scenario'].nunique()} scenarios**")

    st.markdown("---")
    st.markdown("#### Legend")
    st.markdown("🟢 **SNN** — model output")
    st.markdown("🔵 **FW** — ESP32 firmware")
    st.markdown("🟡 **DISAGREE** — they differ")


# ──────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────
if "idx"        not in st.session_state: st.session_state.idx        = 0
if "session_id" not in st.session_state: st.session_state.session_id = session_id
if "history"    not in st.session_state: st.session_state.history    = []
if "buf"        not in st.session_state: st.session_state.buf        = FeatureBuffer()

# Reset buffer when session changes
if st.session_state.session_id != session_id:
    st.session_state.session_id = session_id
    st.session_state.idx        = 0
    st.session_state.history    = []
    st.session_state.buf        = FeatureBuffer()

idx     = st.session_state.idx
buf     = st.session_state.buf
history = st.session_state.history

# ──────────────────────────────────────────────────────────────────────
#  PLAYBACK CONTROLS
# ──────────────────────────────────────────────────────────────────────
st.markdown("## 🧠 Neuromorphic AUV — Real-Time SNN Inference")
st.markdown('<span class="tag">LIVE MODEL INFERENCE</span> &nbsp; reading ESP32 sensor stream, running EventDrivenSNN, comparing to firmware heuristic', unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,4])
with c1:
    if st.button("⏮ Reset"):
        st.session_state.idx     = 0
        st.session_state.history = []
        st.session_state.buf     = FeatureBuffer()
        st.rerun()
with c2:
    step_back = st.button("◀ Back")
with c3:
    step_fwd  = st.button("Next ▶")
with c4:
    auto_play = st.checkbox("▶▶ Auto", value=auto_play)
with c5:
    st.progress(idx / max(len(session_df) - 1, 1))

if step_back and idx > 0:
    st.session_state.idx = max(0, idx - 1)
    st.rerun()
if step_fwd:
    st.session_state.idx = min(idx + 1, len(session_df) - 1)
    st.rerun()

# ──────────────────────────────────────────────────────────────────────
#  RUN INFERENCE ON CURRENT FRAME
# ──────────────────────────────────────────────────────────────────────
row = session_df.iloc[idx].to_dict()
buf.push(row)

snn_danger_p  = None
snn_pred      = None
snn_nav       = None
latency_ms    = None
n_spikes      = 0
feats_raw     = None

if buf.ready():
    feats_raw, spike_pattern = buf.extract()
    snn_danger_p, snn_pred, latency_ms, n_spikes = run_snn_inference(model, scaler, feats_raw)
    snn_nav = nav_command(snn_pred, snn_danger_p, row["dist_f_cm"], row["baseline_cm"])
else:
    spike_pattern = []

# Firmware values from CSV
fw_danger  = float(row["danger"])
fw_pred    = "DANGER" if fw_danger > 0 else "SAFE"
fw_nav_raw = row.get("event", 0)

# Agreement
if snn_pred is not None:
    agree = snn_pred == fw_pred
    history.append({
        "idx":       idx,
        "time_s":    row["time_ms"] / 1000,
        "dist_f":    row["dist_f_cm"],
        "baseline":  row["baseline_cm"],
        "fw_danger": fw_danger,
        "snn_danger":snn_danger_p if snn_danger_p is not None else 0,
        "fw_pred":   fw_pred,
        "snn_pred":  snn_pred if snn_pred else "WARMING",
        "agree":     agree,
        "scenario":  row.get("scenario", ""),
        "n_spikes":  n_spikes,
    })
    if len(history) > 300:
        history.pop(0)

st.session_state.history = history

# ──────────────────────────────────────────────────────────────────────
#  TOP METRICS ROW
# ──────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="snn-card">
      <div class="metric-label">Distance</div>
      <div class="metric-big" style="color:#2e86ab">{row['dist_f_cm']:.1f}<span style="font-size:1rem;color:#4a7fa5"> cm</span></div>
      <div style="font-size:0.75rem;color:#4a7fa5">baseline {row['baseline_cm']:.0f} cm · Δ {row['dist_f_cm']-row['baseline_cm']:+.1f}</div>
    </div>""", unsafe_allow_html=True)

with m2:
    col = "#2ecc71" if (snn_danger_p or 0) < 0.4 else "#f39c12" if (snn_danger_p or 0) < DANGER_THRESH else "#e74c3c"
    val = f"{(snn_danger_p or 0)*100:.1f}%" if snn_danger_p is not None else "WARMING…"
    st.markdown(f"""
    <div class="snn-card">
      <div class="metric-label">🟢 SNN Danger P</div>
      <div class="metric-big" style="color:{col}">{val}</div>
      <div style="font-size:0.75rem;color:#4a7fa5">threshold 55% · {snn_pred or '—'}</div>
    </div>""", unsafe_allow_html=True)

with m3:
    col2 = "#2ecc71" if fw_danger == 0 else "#e74c3c"
    st.markdown(f"""
    <div class="snn-card">
      <div class="metric-label">🔵 FW Danger Score</div>
      <div class="metric-big" style="color:{col2}">{fw_danger*100:.1f}%</div>
      <div style="font-size:0.75rem;color:#4a7fa5">ESP32 heuristic · {fw_pred}</div>
    </div>""", unsafe_allow_html=True)

with m4:
    if snn_pred is not None:
        badge = '<span class="agree-badge agree">✓ AGREE</span>' if agree else '<span class="agree-badge disagree">✗ DISAGREE</span>'
        nav_col = {"FORWARD":"#2ecc71","CAUTION":"#f39c12","STOP":"#e74c3c","REVERSE":"#e74c3c"}.get(snn_nav,"#fff")
        st.markdown(f"""
        <div class="snn-card">
          <div class="metric-label">SNN Nav Command</div>
          <div class="metric-big" style="color:{nav_col};font-size:1.6rem">{snn_nav}</div>
          <div style="margin-top:6px">{badge}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="snn-card">
          <div class="metric-label">SNN Nav Command</div>
          <div class="metric-big" style="color:#4a7fa5;font-size:1.2rem">WARMING UP</div>
          <div style="font-size:0.75rem;color:#4a7fa5">need {WINDOW_SIZE+2} readings</div>
        </div>""", unsafe_allow_html=True)

with m5:
    lat_str  = f"{latency_ms:.2f} ms" if latency_ms else "—"
    spk_str  = f"{n_spikes}" if snn_pred else "—"
    spar_str = f"{(1-n_spikes/150)*100:.0f}%" if n_spikes and snn_pred else "—"
    st.markdown(f"""
    <div class="snn-card">
      <div class="metric-label">SNN Performance</div>
      <div style="display:flex;gap:20px;margin-top:4px">
        <div><div style="font-size:0.7rem;color:#4a7fa5">Latency</div><div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#e8e8e8">{lat_str}</div></div>
        <div><div style="font-size:0.7rem;color:#4a7fa5">Spikes</div><div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#e8e8e8">{spk_str}/150</div></div>
        <div><div style="font-size:0.7rem;color:#4a7fa5">Sparsity</div><div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#2ecc71">{spar_str}</div></div>
      </div>
      <div style="font-size:0.72rem;color:#4a7fa5;margin-top:4px">scenario: {row.get('scenario','')}</div>
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
#  MAIN CHARTS
# ──────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])

with left:
    if len(history) >= 2:
        hist_df = pd.DataFrame(history)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Distance (cm)", "Danger — SNN 🟢 vs Firmware 🔵", "Spike Activity per Window"),
            vertical_spacing=0.08,
            row_heights=[0.42, 0.35, 0.23]
        )

        # ── Distance
        fig.add_trace(go.Scatter(
            x=hist_df["time_s"], y=hist_df["dist_f"],
            name="dist_f_cm", line=dict(color="#2e86ab", width=2),
            hovertemplate="t=%{x:.2f}s<br>dist=%{y:.1f}cm"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=hist_df["time_s"], y=hist_df["baseline"],
            name="baseline", line=dict(color="#06a77d", width=1.5, dash="dot"),
        ), row=1, col=1)

        # enter threshold
        if "enter_thr_cm" in session_df.columns:
            thr_vals = session_df.iloc[max(0,idx-len(history)):idx]["enter_thr_cm"].values
            thr_times = hist_df["time_s"].values[-len(thr_vals):]
            fig.add_trace(go.Scatter(
                x=thr_times, y=thr_vals,
                name="enter_thr", line=dict(color="#e74c3c", width=1, dash="dash"),
            ), row=1, col=1)

        # mark disagree points
        disagree_df = hist_df[~hist_df["agree"]]
        if not disagree_df.empty and snn_pred is not None:
            fig.add_trace(go.Scatter(
                x=disagree_df["time_s"], y=disagree_df["dist_f"],
                name="⚡ disagree", mode="markers",
                marker=dict(color="#f39c12", size=10, symbol="x"),
            ), row=1, col=1)

        # current position
        fig.add_trace(go.Scatter(
            x=[row["time_ms"]/1000], y=[row["dist_f_cm"]],
            name="now", mode="markers",
            marker=dict(color="white", size=8, symbol="diamond",
                        line=dict(color="#2e86ab", width=2))
        ), row=1, col=1)

        # ── Danger comparison
        fig.add_trace(go.Scatter(
            x=hist_df["time_s"], y=hist_df["snn_danger"],
            name="SNN P(danger)", line=dict(color="#2ecc71", width=2.5),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.08)",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=hist_df["time_s"], y=hist_df["fw_danger"],
            name="FW danger", line=dict(color="#2e86ab", width=2, dash="dot"),
        ), row=2, col=1)

        # threshold line
        fig.add_hline(y=DANGER_THRESH, line=dict(color="#e74c3c", width=1, dash="dash"),
                      row=2, col=1, annotation_text=f"SNN threshold {DANGER_THRESH}")

        # ── Spikes
        fig.add_trace(go.Bar(
            x=hist_df["time_s"], y=hist_df["n_spikes"],
            name="active spikes", marker_color="#7b2fff",
            opacity=0.8,
        ), row=3, col=1)

        fig.add_hline(y=150*0.077, line=dict(color="#4a7fa5", width=1, dash="dot"),
                      row=3, col=1, annotation_text="avg 7.7%")

        fig.update_layout(
            height=640,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,31,60,0.6)",
            font=dict(color="#a0b4c8", family="JetBrains Mono"),
            legend=dict(orientation="h", y=1.02, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        for r_i in [1,2,3]:
            fig.update_xaxes(gridcolor="#1e3a5f", row=r_i, col=1)
            fig.update_yaxes(gridcolor="#1e3a5f", row=r_i, col=1)
        fig.update_xaxes(title_text="time (s)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Warming up… need {WINDOW_SIZE + 2} readings. Current: {len(history) + idx}")

# ──────────────────────────────────────────────────────────────────────
#  RIGHT PANEL — Feature window + agreement log
# ──────────────────────────────────────────────────────────────────────
with right:
    # Feature window heatmap
    if feats_raw is not None:
        st.markdown("**Feature window — scaled (15 × 5)**")
        feat_labels = ["time_gap","dist_f","delta","velocity","accel"]
        # Normalise each feature row independently so velocity and
        # acceleration are visible — same as what the SNN actually sees
        feats_display = feats_raw.copy()
        for i in range(feats_display.shape[1]):
            row_vals = feats_display[:, i]
            rng = row_vals.max() - row_vals.min()
            if rng > 1e-10:
                feats_display[:, i] = (row_vals - row_vals.min()) / rng
            else:
                feats_display[:, i] = 0.5
        fig2 = go.Figure(go.Heatmap(
            z=feats_display.T,
            x=[f"t{i+1}" for i in range(WINDOW_SIZE)],
            y=feat_labels,
            customdata=feats_raw.T,
            colorscale=[[0,"#0d1f3c"],[0.5,"#2e86ab"],[1,"#e74c3c"]],
            showscale=False,
            zmin=0, zmax=1,
            hovertemplate="<b>%{y}</b> at %{x}<br>raw: %{customdata:.5f}<extra></extra>",
        ))
        fig2.update_layout(
            height=200, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,31,60,0.6)",
            font=dict(color="#a0b4c8", size=10, family="JetBrains Mono"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div style='font-size:0.65rem;color:#2a4a6a'>each row normalised independently — shows relative change within each feature</div>", unsafe_allow_html=True)

        # Spike pattern bars
        st.markdown("**Spike pattern (ON+OFF per step)**")
        bar_html = '<div class="spike-bar-container">'
        max_sp = max(spike_pattern) if spike_pattern and max(spike_pattern)>0 else 1
        for sp in spike_pattern:
            h = int(sp / max_sp * 36) if sp else 2
            col_sp = "#7b2fff" if sp > 0 else "#1e3a5f"
            bar_html += f'<div class="spike-bar" style="height:{h}px;background:{col_sp}"></div>'
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Agreement log
    st.markdown("**Recent decisions**")
    if history:
        log_df = pd.DataFrame(history[-15:][::-1])
        for _, h in log_df.iterrows():
            color = "#2ecc71" if h["agree"] else "#f39c12"
            sym   = "✓" if h["agree"] else "✗"
            snn_c = "#e74c3c" if h["snn_pred"]=="DANGER" else "#2ecc71"
            fw_c  = "#e74c3c" if h["fw_pred"]=="DANGER" else "#2e86ab"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #1e3a5f;font-family:JetBrains Mono,monospace;font-size:0.72rem">'
                f'<span style="color:{color};font-weight:700">{sym}</span>'
                f'<span style="color:#4a7fa5">{h["time_s"]:.1f}s</span>'
                f'<span style="color:{snn_c}">SNN:{h["snn_pred"][:3]}</span>'
                f'<span style="color:{fw_c}">FW:{h["fw_pred"][:3]}</span>'
                f'<span style="color:#4a7fa5">{int(h["n_spikes"])}sp</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ──────────────────────────────────────────────────────────────────────
#  DISAGREE ANALYSIS
# ──────────────────────────────────────────────────────────────────────
if len(history) > 20:
    hist_df    = pd.DataFrame(history)
    n_total    = len(hist_df)
    n_disagree = (~hist_df["agree"]).sum()
    n_agree    = hist_df["agree"].sum()
    agree_rate = n_agree / n_total * 100

    st.markdown("---")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Decisions made", n_total)
    with a2:
        st.metric("Agreement", f"{agree_rate:.1f}%")
    with a3:
        n_snn_only   = ((hist_df["snn_pred"]=="DANGER") & (hist_df["fw_pred"]=="SAFE")).sum()
        st.metric("SNN-only danger", int(n_snn_only), help="SNN detected danger, firmware missed it")
    with a4:
        n_fw_only    = ((hist_df["snn_pred"]=="SAFE") & (hist_df["fw_pred"]=="DANGER")).sum()
        st.metric("FW-only danger",  int(n_fw_only),  help="Firmware detected danger, SNN said safe")

# ──────────────────────────────────────────────────────────────────────
#  AUTO ADVANCE
# ──────────────────────────────────────────────────────────────────────
if auto_play:
    if idx < len(session_df) - 1:
        st.session_state.idx = idx + 1
        time.sleep(max(0.05, 0.3 / speed))
        st.rerun()
    else:
        st.info("Session complete. Press Reset to replay.")

# ──────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#2a4a6a'>"
    "EventDrivenSNN · 1,474 params · 5.76 KB · snntorch + PyTorch · "
    "Horowitz 2014 energy model · University of Peradeniya · CO422</div>",
    unsafe_allow_html=True
)
