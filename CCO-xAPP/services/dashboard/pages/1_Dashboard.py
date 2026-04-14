import streamlit as st
import pandas as pd
import json, os
import threading
import time
#import numpy as np
import matplotlib.pyplot as plt
import openai
from streamlit_chat import message
from streamlit_autorefresh import st_autorefresh

# ===============================
# Page Setup
# ===============================
st.set_page_config(
    page_title="CCO-xAPP: Near-RT RIC Closed-Loop Energy Optimization",
    layout="wide"
)

# ===============================
# CSS – Dark Theme and Header (Unified with LLM)
# ===============================
st.markdown("""
<style>
/* ===== Remove extra top gap below header ===== */
main > div:first-child {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* ===== Enhanced Dark Theme + Compact Layout ===== */

/* Overall app styling */
.stApp {
    background-color:#0f1115;
    color:#E0E0E0;
    font-family:'Segoe UI','Roboto',sans-serif;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, header {visibility:hidden;}

/* ===== Sidebar Styling ===== */
section[data-testid="stSidebar"] {
    background-color:#14161a;
    border-right:1px solid #222;
}
/* Sidebar text and nav links (bright + readable) */
section[data-testid="stSidebar"] div[role="listbox"] *,
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] *,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] a {
    color:#E0E0E0 !important;
    font-weight:500 !important;
    text-decoration:none !important;
    transition:all 0.3s ease-in-out !important;
}
/* Hover glow effect */
section[data-testid="stSidebar"] a:hover,
section[data-testid="stSidebar"] div[role="listbox"] *:hover {
    color:#00E5A0 !important;
    text-shadow:0 0 6px rgba(0,229,160,0.6);
}
/* Active page highlight */
section[data-testid="stSidebar"] a[aria-current="page"],
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLinkActive"],
section[data-testid="stSidebar"] div[data-testid="stSidebarNavLink"]:has(a[aria-current="page"]) {
    background-color:rgba(0,229,160,0.1) !important;
    border-left:3px solid #00E5A0 !important;
    color:#00E5A0 !important;
    box-shadow:inset 0 0 6px rgba(0,229,160,0.4);
    border-radius:4px;
    padding-left:12px !important;
}

/* ===== Header Bar ===== */
.header-container {
    background-color:#181C1F;
    padding:18px 0;
    border-bottom:1px solid #222;
    border-radius:8px;
    box-shadow:0 2px 4px rgba(0,0,0,0.3);
    margin-bottom:20px;
    text-align:center;
}
.header-title {
    color:#00E5A0;
    font-size:30px;
    font-weight:700;
    letter-spacing:0.6px;
}

/* ===== Metrics ===== */
[data-testid="stMetric"] {
    background-color:#111416;
    border-radius:12px;
    padding:15px;
    box-shadow:0 0 10px rgba(0,229,160,0.1);
}
[data-testid="stMetricLabel"] {
    color:#AAB4BE !important;
    font-size:0.9rem;
}
[data-testid="stMetricValue"] {
    color:#00E5A0 !important;
    font-weight:700;
    font-size:1.8rem;
}
[data-testid="stMetricDelta"] {
    color:#2ECC71 !important;
}

/* ===== DataFrames ===== */
div[data-testid="stDataFrame"] {
    background-color:#111416;
    border-radius:10px;
    padding:5px !important;
    border:1px solid #222;
    margin-top:0.25rem !important;
    margin-bottom:0.25rem !important;
}
div[data-testid="stDataFrame"] table {
    background-color:#202124;
    color:#E0E0E0;
}
div[data-testid="stDataFrame"] th {
    background-color:#181C1F !important;
    color:#00E5A0 !important;
}

/* ===== Compact Layouts ===== */
section[data-testid="stVerticalBlock"] {
    padding-top:0.25rem !important;
    padding-bottom:0.25rem !important;
    margin-top:0.25rem !important;
    margin-bottom:0.25rem !important;
}
h2, h3 {
    color:#00E5A0;
    text-transform:uppercase;
    font-weight:700;
    letter-spacing:0.5px;
    margin-top:0.4rem !important;
    margin-bottom:0.2rem !important;
}

/* ===== Compact Plots ===== */
.element-container:has(canvas) {
    margin-top:0.25rem !important;
    margin-bottom:0.25rem !important;
}

/* ===== Fix bottom gap ===== */
section.main > div {
    margin-bottom:0rem !important;
}

/* ===== Fix vertical gaps between main sections ===== */
div.block-container > div:nth-child(n) {
    margin-top:0rem !important;
    margin-bottom:0rem !important;
    padding-top:0rem !important;
    padding-bottom:0rem !important;
}

/* Tighter spacing between top and bottom dashboard rows */
section[data-testid="stHorizontalBlock"] {
    margin-top:0.25rem !important;
    margin-bottom:0.25rem !important;
}

/* Reduce subheader spacing */
h3 {
    margin-top:0.3rem !important;
    margin-bottom:0.15rem !important;
}
</style>

<div class="header-container">
  <div class="header-title">CCO-xAPP: AI-Driven Near-RT RIC Closed-Loop Energy Optimization</div>
</div>
""", unsafe_allow_html=True)


# ===============================
# Helpers
# ===============================
SHARED = "/shared"

def read_jsonl_lines(path):
    """Return DataFrame from a JSONL file (one JSON object per line).
    Robust: skips bad lines instead of failing the whole file.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                # Log to container stdout; dashboard keeps working
                print(f"[dashboard] skipping bad JSON line {i} in {path}: {e}")
                continue

    return pd.DataFrame(rows)

def read_json_array(path):
    """Return DataFrame from a JSON array file."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        return pd.DataFrame(arr)
    except Exception:
        return pd.DataFrame()

def read_any(*names):
    """Try multiple filenames (jsonl, json) and return the first non-empty DF."""
    for name in names:
        p = os.path.join(SHARED, name)
        if name.endswith(".jsonl"):
            df = read_jsonl_lines(p)
        else:
            df = read_json_array(p)
        if len(df):
            return df
    return pd.DataFrame()

# ===============================
# Auto-refresh
# ===============================
# Sidebar refresh control
#refresh = st.sidebar.slider("Auto-refresh (sec)", 3, 30, 5)
refresh = st.sidebar.slider("Auto-refresh (sec)", 1, 10, 3)
st_autorefresh(interval=refresh * 1000, key="dashboard_autorefresh")

# ===============================
# Main KPIs
# ===============================
c1, c2 = st.columns(2)

# ---------- Collector ----------
with c1:
    st.subheader("Collector: Cell KPIs (Live Feed)")
    df_col = read_any("collector_log.jsonl")  # expected jsonl
    df_col = df_col.tail(500)

    if len(df_col):
        # Ensure timestamp usability
        if "ts" not in df_col.columns:
            df_col["ts"] = range(len(df_col))
        else:
            df_col["ts"] = pd.to_datetime(df_col["ts"], errors="coerce")

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # PRB Utilization & UE Count per cell
        for cell_id, g in df_col.groupby("cell_id"):
            if "prb_util" in g.columns:
                axes[0].plot(g["ts"], g["prb_util"] * 100, label=f"{cell_id} PRB Util (%)")
            if "ue_count" in g.columns:
                axes[0].plot(g["ts"], g["ue_count"], label=f"{cell_id} UE Count", linestyle="--")

        axes[0].set_ylabel("PRB (%) / UEs")
        axes[0].set_title("Cell Load Dynamics per Cell")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Power vs HO SR
        if {"ptot", "ho_success_rate"}.issubset(df_col.columns):
            for cell_id, g in df_col.groupby("cell_id"):
                axes[1].plot(g["ts"], g["ptot"], label=f"{cell_id} Power (W)")
                axes[1].plot(g["ts"], g["ho_success_rate"], label=f"{cell_id} HO SR (%)", linestyle="--")

        axes[1].set_ylabel("Power (W) / HO SR (%)")
        axes[1].set_xlabel("Timestamp")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)

        # Average KPIs per cell
        st.markdown("**Average KPIs per Cell**")
        cols = [c for c in ["prb_util", "ue_count", "ho_success_rate", "ptot"] if c in df_col.columns]
        if cols:
            summary = df_col.groupby("cell_id")[cols].mean().reset_index()
            if "prb_util" in summary.columns:
                summary["prb_util"] = summary["prb_util"] * 100
            summary = summary.rename(columns={
                "cell_id": "Cell",
                "prb_util": "Avg PRB Util (%)",
                "ue_count": "Avg UE Count",
                "ho_success_rate": "Avg HO SR (%)",
                "ptot": "Avg Power (W)"
            })
            st.dataframe(summary, use_container_width=True)
        else:
            summary = pd.DataFrame()

#### Trainer ####
with c2:
    st.subheader("ML Trainer: Model Performance")
    df_tr = read_any("trainer_metrics.jsonl").tail(20)

    if len(df_tr):
        model_map = {
            "logreg": "Logistic Regression",
            "xgb": "XGBoost",
            "gb": "Gradient Boosting",
            "rf": "Random Forest",
            "knn": "K-Nearest Neighbors",
            "svc": "Support Vector Classifier",
            "svm": "Support Vector Classifier",
        }
        if "best_model" in df_tr.columns:
            df_tr["best_model"] = df_tr["best_model"].replace(model_map)

        view_cols = [c for c in ["ts", "best_model", "accuracy", "f1"] if c in df_tr.columns]
        if "ts" in df_tr.columns:
            df_tr["ts"] = pd.to_datetime(df_tr["ts"], errors="coerce")
        st.dataframe(df_tr[view_cols].reset_index(drop=True), use_container_width=True)

        # Accuracy & F1 bar plot
        if {"accuracy", "f1"}.issubset(df_tr.columns):
            df_tr["accuracy"] = pd.to_numeric(df_tr["accuracy"], errors="coerce")
            df_tr["f1"] = pd.to_numeric(df_tr["f1"], errors="coerce")

            fig, ax = plt.subplots(figsize=(8, 4))
            width = 0.35
            x = range(len(df_tr))
            ax.bar([i - width/2 for i in x], df_tr["accuracy"], width, label="Accuracy")
            ax.bar([i + width/2 for i in x], df_tr["f1"], width, label="F1 Score")
            if "best_model" in df_tr.columns:
                ax.set_xticks(list(x))
                ax.set_xticklabels(df_tr["best_model"], rotation=45, ha="right")
            ax.set_title("Model Comparison per Training Cycle")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        # Dominance (last 100)
        if "best_model" in df_tr.columns and len(df_tr):
            st.markdown("### Model Dominance (Last 100 Training Cycles)")
            model_counts = df_tr["best_model"].tail(100).value_counts().sort_values(ascending=True)
            if len(model_counts):
                fig_dom, ax_dom = plt.subplots(figsize=(6, 3))
                ax_dom.barh(model_counts.index, model_counts.values)
                ax_dom.set_xlabel("Frequency (times selected as best model)")
                ax_dom.set_ylabel("Model Name")
                ax_dom.set_title("Dominance of Models over Recent Cycles")
                ax_dom.grid(True, linestyle="--", alpha=0.4)
                st.markdown("""
                <div style="width: 100%; display: flex; justify-content: center;">
                     <div style="width: 60%;">
                """, unsafe_allow_html=True)
                st.pyplot(fig_dom)
                st.markdown("""
                   </div>
                </div>
                """, unsafe_allow_html=True)

# ===============================
# Inference & Actuator
# ===============================
c3, c4 = st.columns(2)

# ---------- Inference ----------
with c3:
    st.subheader("Inference: Prediction Confidence (per cell)")
    df_inf = read_any("inference_scores.jsonl").tail(500)
    df_inf = df_inf.tail(500)

    if len(df_inf):
        if "ts" not in df_inf.columns:
            df_inf["ts"] = range(len(df_inf))
        else:
            df_inf["ts"] = pd.to_datetime(df_inf["ts"], errors="coerce")

        fig, ax = plt.subplots(figsize=(8, 5))
        if "p_sleep" in df_inf.columns:
            for cell_id, g in df_inf.groupby("cell_id"):
                ax.plot(g["ts"], g["p_sleep"], label=f"Cell {cell_id}")
        ax.set_title("xApp Decision Confidence by Cell")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("p(sleep)")
        ax.legend(title="Cell ID", loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

        # Try to merge nearest power_saved from actuator
        df_act_full = read_any("actions_log.jsonl", "actions_log.json")  # support both
        try:
            if len(df_act_full):
                if "ts" in df_act_full.columns:
                    df_act_full["ts"] = pd.to_datetime(df_act_full["ts"], errors="coerce")
                if "ts" in df_inf.columns:
                    df_inf["ts"] = pd.to_datetime(df_inf["ts"], errors="coerce")
                # clean null keys
                df_act_full = df_act_full.dropna(subset=["ts", "cell_id"])
                df_inf = df_inf.dropna(subset=["ts", "cell_id"])
                if {"cell_id", "ts"}.issubset(df_act_full.columns):
                    df_inf = pd.merge_asof(
                        df_inf.sort_values("ts"),
                        df_act_full.sort_values("ts")[["ts", "cell_id", "power_saved"]],
                        on="ts",
                        by="cell_id",
                        direction="nearest",
                        tolerance=pd.Timedelta("3s")
                    )
        except Exception as e:
            st.warning(f"Unable to merge power data: {e}")

        def hl_inf(row):
            if row.get("decision") == "sleep":
                return ["background-color: #d9fdd3"] * len(row)
            if "power_saved" in row and pd.notna(row["power_saved"]) and float(row["power_saved"]) == 0:
                return ["background-color: #f0f0f0"] * len(row)
            return [""] * len(row)

# ---------- Actuator ----------
with c4:
    st.subheader("Actuator: E2 Control Actions from Near-RT RIC to O-DU")
    df_act = read_any("actions_log.jsonl", "actions_log.json").tail(100)
    df_act = df_act.tail(100)

    if len(df_act):
        def hl_act(row):
            return ["background-color: #d9fdd3"] * len(row) if row.get("action") == "sleep" else ["" for _ in row]

        display_cols = [c for c in ["ts", "cell_id", "action", "score", "ptot_active", "power_saved"] if c in df_act.columns]
        st.dataframe(
            df_act[display_cols].reset_index(drop=True).style.apply(hl_act, axis=1),
            use_container_width=True
        )

        # Energy totals (J, kJ, Wh) — assuming power_saved is W over sample window
        if "power_saved" in df_act.columns:
            total_saved_w = pd.to_numeric(df_act["power_saved"], errors="coerce").fillna(0).sum()
            avg_saved_w = pd.to_numeric(df_act["power_saved"], errors="coerce").fillna(0).mean()

            # If your actuator samples every 5s, energy over a single entry = W * 5s (J)
            sample_interval_s = 5
            E_joules = total_saved_w * sample_interval_s
            E_kj = E_joules / 1000.0
            E_wh = E_joules / 3600.0

            # Display metrics horizontally
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Energy Saved (kJ)", f"{E_kj:.2f}")

            with col2:
                st.metric("Total Energy Saved (Wh)", f"{E_wh:.2f}")

            with col3:
                st.metric("Avg Power Saved per Action (W)", f"{avg_saved_w:.2f}")

            if "ts" in df_act.columns:
                df_act["ts"] = pd.to_datetime(df_act["ts"], errors="coerce")