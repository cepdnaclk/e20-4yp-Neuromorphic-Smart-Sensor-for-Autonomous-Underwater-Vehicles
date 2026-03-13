"""
Real-Time SNN Dashboard for Neuromorphic AUV
Interactive visualization of obstacle detection

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page config
st.set_page_config(
    page_title="Neuromorphic AUV Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Titlestreamlit run dashboard.py
st.title("🤖 Neuromorphic Smart Sensor - Real-Time Dashboard")
st.markdown("**Group 22 | CO422 Final Year Project**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("sensor.csv")
    df = df[df['valid'] == 1].copy()
    df = df[df['baseline_cm'] > 0].copy()
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("⚙️ Controls")
sensor_id = st.sidebar.selectbox("Sensor ID", df['sensor_id'].unique(), index=0)
session_id = st.sidebar.selectbox("Session ID", sorted(df['session_id'].unique()), index=0)
speed = st.sidebar.slider("Playback Speed", 0.1, 5.0, 1.0, 0.1)
auto_play = st.sidebar.checkbox("Auto Play", value=False)

# Filter data
session_data = df[(df['session_id'] == session_id) & (df['sensor_id'] == sensor_id)].copy()
session_data = session_data.sort_values('time_ms').reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Readings:** {len(session_data)}")
st.sidebar.markdown(f"**Danger Events:** {(session_data['danger'] > 0).sum()}")
st.sidebar.markdown(f"**Duration:** {(session_data['time_ms'].max() - session_data['time_ms'].min())/1000:.1f}s")

# Main layout
col1, col2 = st.columns([2, 1])

# Playback controls
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if st.button("⏮️ Reset"):
        st.session_state.current_idx = 0

with control_col2:
    if st.button("⏸️ Pause"):
        auto_play = False

with control_col3:
    if st.button("▶️ Play"):
        auto_play = True

with control_col4:
    if st.button("⏭️ Next"):
        st.session_state.current_idx = min(st.session_state.current_idx + 1, len(session_data) - 1)

# Progress bar
progress = st.session_state.current_idx / max(len(session_data) - 1, 1)
st.progress(progress)

# Get current reading
current_idx = st.session_state.current_idx
if current_idx >= len(session_data):
    current_idx = 0
    st.session_state.current_idx = 0

current = session_data.iloc[current_idx]

# Left column - Main visualization
with col1:
    # Distance time series
    st.subheader("📊 Distance vs Time")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Filtered Distance", "Danger Score"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Get window of data around current point
    window_size = 100
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(session_data), current_idx + 50)
    window_data = session_data.iloc[start_idx:end_idx]
    
    # Distance plot
    fig.add_trace(
        go.Scatter(
            x=window_data['time_ms'] / 1000,
            y=window_data['dist_f_cm'],
            name='Distance',
            line=dict(color='#2E86AB', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Baseline
    fig.add_trace(
        go.Scatter(
            x=window_data['time_ms'] / 1000,
            y=window_data['baseline_cm'],
            name='Baseline',
            line=dict(color='#06A77D', width=1, dash='dash'),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Current point
    fig.add_trace(
        go.Scatter(
            x=[current['time_ms'] / 1000],
            y=[current['dist_f_cm']],
            name='Current',
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond')
        ),
        row=1, col=1
    )
    
    # Danger score
    fig.add_trace(
        go.Scatter(
            x=window_data['time_ms'] / 1000,
            y=window_data['danger'],
            name='Danger',
            line=dict(color='#D00000', width=2),
            fill='tozeroy',
            fillcolor='rgba(208, 0, 0, 0.2)',
            mode='lines'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (cm)", row=1, col=1)
    fig.update_yaxes(title_text="Danger Score", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Right column - Current status
with col2:
    st.subheader("📡 Current Status")
    
    # Time
    st.metric(
        label="⏱️ Time",
        value=f"{current['time_ms']/1000:.2f}s"
    )
    
    # Distance
    delta_dist = current['dist_f_cm'] - current['baseline_cm']
    st.metric(
        label="📏 Distance",
        value=f"{current['dist_f_cm']:.1f} cm",
        delta=f"{delta_dist:+.1f} cm from baseline"
    )
    
    # Danger status
    is_danger = current['danger'] > 0
    danger_color = "🔴" if is_danger else "🟢"
    st.metric(
        label=f"{danger_color} Danger Level",
        value=f"{current['danger']*100:.1f}%"
    )
    
    # Navigation command
    if current['danger'] > 0.7:
        nav_cmd = "🛑 STOP"
        nav_color = "red"
    elif current['danger'] > 0.4:
        nav_cmd = "⚠️ CAUTION"
        nav_color = "orange"
    else:
        nav_cmd = "✅ FORWARD"
        nav_color = "green"
    
    st.markdown(f"### Navigation: :{nav_color}[{nav_cmd}]")
    
    st.markdown("---")
    
    # Additional metrics
    st.markdown("#### 📊 Details")
    st.markdown(f"**Baseline:** {current['baseline_cm']:.1f} cm")
    st.markdown(f"**Enter Threshold:** {current['enter_thr_cm']:.1f} cm")
    st.markdown(f"**Exit Threshold:** {current['exit_thr_cm']:.1f} cm")
    st.markdown(f"**Valid:** {'✓' if current['valid'] else '✗'}")
    st.markdown(f"**Event:** {'Yes' if current['event'] else 'No'}")

# Bottom section - Performance metrics
st.markdown("---")
st.subheader("⚡ Performance Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="Total Readings",
        value=f"{len(session_data):,}"
    )

with metric_col2:
    danger_rate = (session_data['danger'] > 0).sum() / len(session_data) * 100
    st.metric(
        label="Danger Rate",
        value=f"{danger_rate:.1f}%"
    )

with metric_col3:
    avg_dist = session_data['dist_f_cm'].mean()
    st.metric(
        label="Avg Distance",
        value=f"{avg_dist:.1f} cm"
    )

with metric_col4:
    st.metric(
        label="SNN Latency",
        value="0.38 ms"
    )

# Auto-advance
if auto_play and st.session_state.current_idx < len(session_data) - 1:
    time.sleep(0.1 / speed)
    st.session_state.current_idx += 1
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>Neuromorphic Smart Sensor for Autonomous Underwater Vehicles</p>
    <p>University of Peradeniya | Department of Computer Engineering</p>
    </div>
    """,
    unsafe_allow_html=True
)
