# app.py — Version ULTRA PRO (Cybersecurity Dashboard style Splunk/Kibana)

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    preprocess_data,
    get_model_metrics,
    log_alert,
    read_alerts,
    clear_alerts,
    generate_sample_nsl_kdd,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NIDS Pro", layout="wide")

# ─────────────────────────────────────────────
# THEME SWITCH + STATE
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

with st.sidebar:
    st.title("🛡️ NIDS Control Panel")

    theme = st.radio("Theme", ["dark", "light"], index=0)
    st.session_state.theme = theme

    selected_model = st.selectbox("Model", ["Random Forest", "XGBoost"])
    threshold = st.slider("Threat Threshold", 50, 99, 75)

# ─────────────────────────────────────────────
# CSS PRO DASHBOARD
# ─────────────────────────────────────────────

def load_css(theme="dark"):
    if theme == "dark":
        return """
        <style>
        .stApp {background: radial-gradient(circle at top, #0f172a, #020617); color:#e2e8f0;}
        [data-testid="stSidebar"] {background:#020617;border-right:1px solid #1e293b;}
        
        .card {
            background: rgba(15,23,42,0.7);
            border:1px solid rgba(56,189,248,0.2);
            border-radius:16px;
            padding:1rem;
            backdrop-filter: blur(10px);
        }

        .metric {
            font-size:1.8rem;
            font-weight:700;
            color:#38bdf8;
        }

        .danger {color:#f87171;}
        .ok {color:#4ade80;}

        .stButton>button {
            background: linear-gradient(135deg,#0ea5e9,#22d3ee);
            border:none;
            border-radius:10px;
            color:white;
        }
        </style>
        """

    else:
        return """
        <style>
        .stApp {background:#f1f5f9;color:#0f172a;}
        [data-testid="stSidebar"] {background:#e2e8f0;}

        .card {
            background:white;
            border:1px solid #cbd5e1;
            border-radius:16px;
            padding:1rem;
        }
        </style>
        """

st.markdown(load_css(st.session_state.theme), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────
if st.session_state.theme == "dark":
    plt.style.use("dark_background")
else:
    plt.style.use("default")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col1, col2 = st.columns([1,8])
with col1:
    st.markdown("## 🛡️")
with col2:
    st.markdown("## Network Intrusion Detection System")
    st.caption("Real-time Threat Monitoring Dashboard")

st.markdown("---")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = generate_sample_nsl_kdd(300)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
n_total = len(df)
n_attack = (df["category"] != "Normal").sum()
n_normal = n_total - n_attack

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class='card'>
        <div>Total Traffic</div>
        <div class='metric'>{n_total}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='card'>
        <div class='ok'>Normal</div>
        <div class='metric'>{n_normal}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='card'>
        <div class='danger'>Threats</div>
        <div class='metric'>{n_attack}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots()
    df["category"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
    st.pyplot(fig)

with colB:
    fig, ax = plt.subplots()
    df["protocol_type"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ─────────────────────────────────────────────
# REAL-TIME DETECTION SIMULATION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Live Threat Detection")

if st.button("Run Analysis"):
    preds = np.random.choice(["Normal","DoS","Probe"], len(df))
    df["Prediction"] = preds

    st.dataframe(df.head(50))

    for p in preds[:20]:
        if p != "Normal":
            log_alert(p, "192.168.0.1", 0.95)

# ─────────────────────────────────────────────
# LOGS PANEL
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📡 Threat Logs")

logs = read_alerts()
st.text_area("", logs, height=250)
