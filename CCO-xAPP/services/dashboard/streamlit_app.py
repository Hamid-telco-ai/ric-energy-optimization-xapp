import streamlit as st
from PIL import Image
import os

# ===============================
# Page Setup
# ===============================
st.set_page_config(
    page_title="Inside the Mind of the RAN Intelligent Controller (RIC)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Custom CSS (Dark Theme & Layout)
# ===============================
st.markdown("""
<style>
/* ===== General App Styling ===== */
.stApp {
    background-color: #0f1115;
    color: #E0E0E0;
    font-family: 'Segoe UI','Roboto', sans-serif;
}
#MainMenu, footer, header {visibility:hidden;}

/* ===== Header ===== */
.header-container {
    background-color:#181C1F;
    padding:20px 0;
    border-bottom:1px solid #222;
    border-radius:8px;
    box-shadow:0 2px 6px rgba(0,0,0,0.4);
    text-align:center;
    margin-bottom:30px;
}
.header-title {
    color:#00E5A0;
    font-size:32px;
    font-weight:700;
    letter-spacing:0.6px;
}
.header-subtitle {
    color:#AAB4BE;
    font-size:16px;
}

/* ===== Sidebar Styling ===== */
section[data-testid="stSidebar"] {
    background-color:#14161a;
    border-right:1px solid #222;
}

/* Sidebar link text */
section[data-testid="stSidebar"] div[role="listbox"] * ,
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

/* Hover effect */
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

/* ===== Card Styling ===== */
.info-card {
    background-color:#111416;
    border-radius:12px;
    padding:25px;
    box-shadow:0 0 10px rgba(0,229,160,0.1);
    margin:20px 0;
}
.info-card h3 {
    color:#00E5A0;
    font-weight:600;
    margin-bottom:10px;
}
.info-card p {
    color:#C8D0D8;
    font-size:15px;
}
</style>

<div class="header-container">
  <div class="header-title">Inside the Mind of the RAN Intelligent Controller (RIC) - Gen. 1</div>
  <div class="header-subtitle">Where O-RAN xApp Thinks, Acts, and Talks • LLM Reasoning Meets Real-Time Energy Optimization</div>
  <div class="header-author">By: Hamidreza Saberkari, PhD</div>
</div>
""", unsafe_allow_html=True)


# ===============================
# Main Content
# ===============================
st.markdown("""
<div class="info-card">
<h3>Welcome!</h3>
<p>Use the sidebar to open:</p>
<ul>
  <li><b>Dashboard</b> — Real-time KPI and Energy Optimization view</li>
  <li><b>LLM Assistant</b> — Chat-based reasoning over live RIC data</li>
</ul>
</div>

<div class="info-card">
  <h3>Description</h3>
  <p>
  The <b>CCO-xAPP demo</b> brings to life the <b>O-RAN AI/ML-assisted energy-saving framework</b> through an end-to-end, closed-loop architecture.
  The <b>Collector</b> continuously gathers O1-style KPIs from simulated cells, while the <b>Trainer</b> within the <b>Non-RT RIC</b> learns long-term traffic patterns to build predictive ML models for load and energy optimization.
  These models are deployed into the <b>Near-RT RIC</b>, where the <b>Inference Engine (xApp)</b> makes real-time decisions — determining when each cell should <b>sleep</b>, <b>wake</b>, or <b>remain active</b> based on live KPIs and learned behavior.
  The <b>Actuator</b> applies these actions, closing the control loop through an <b>E2-like interface</b>.
  On top of this loop, an <b>LLM Assistant</b> interprets the system’s reasoning, translating raw telemetry into human-understandable insights.
  <br><br>
  </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# Data Flow Overview Image
# ===============================
from PIL import Image
import os

IMG_PATH = "./assets/dataflow.png"

if os.path.exists(IMG_PATH):
    st.markdown("<h3 style='text-align:center; color:#00E5A0;'>Data Flow Overview</h3>", unsafe_allow_html=True)
    img = Image.open(IMG_PATH)
    st.image(img, use_container_width=True)
else:
    st.warning("Data flow diagram not found at ./assets/dataflow.png")

st.markdown("""
<div style='text-align:center; color:#AAB4BE; font-size:15px; margin-top:-10px;'>
</div>
""", unsafe_allow_html=True)
