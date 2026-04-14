import streamlit as st

# --- Force Streamlit to reload everything ---
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from openai import OpenAI

# ===============================
# Persistent Chat History Path
# ===============================
HISTORY_PATH = "/persisted_shared/llm_chat_history.json"
SHARED = "/shared"

# ===============================
# Page Setup
# ===============================
st.set_page_config(page_title="COO-xAPP: RIC LLM Assistant", layout="wide")

# ===============================
# Session State Init
# ===============================
if "messages" not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                saved_msgs = json.load(f)
            st.session_state["messages"] = saved_msgs if isinstance(saved_msgs, list) else []
        except Exception:
            st.session_state["messages"] = []
    else:
        st.session_state["messages"] = []

# ===============================
# Unified Dark Style
# ===============================
st.markdown("""
<style>
.stApp {
    background-color:#0f1115;
    color:#E0E0E0;
    font-family:'Segoe UI','Roboto',sans-serif;
}
#MainMenu, footer, header {visibility:hidden;}

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

section[data-testid="stSidebar"] {
    background-color:#14161a;
    border-right:1px solid #222;
}

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

section[data-testid="stSidebar"] a:hover,
section[data-testid="stSidebar"] div[role="listbox"] *:hover {
    color:#00E5A0 !important;
    text-shadow:0 0 6px rgba(0,229,160,0.6);
}

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

div[data-testid="stChatMessage"] div[role="user"] {
    background-color:#2B2F36 !important;
    color:#FFFFFF !important;
    border-radius:10px;
    padding:12px 16px !important;
}

div[data-testid="stChatMessage"] div[role="assistant"] {
    background-color:#1E2428 !important;
    color:#E0E0E0 !important;
    border-left:4px solid #00E5A0 !important;
    border-radius:10px;
    padding:12px 16px !important;
}

div[data-testid="stChatMessage"] svg {
    color:#00E5A0 !important;
}
div[data-testid="stChatMessage"] {
    margin-bottom:12px !important;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] li {
    color:#E8E8E8 !important;
    font-size:15px;
    line-height:1.5;
}
</style>

<div class="header-container">
  <div class="header-title">RIC LLM Assistant</div>
  <div class="header-subtitle">Where the RIC explains its thoughts — powered by live KPIs, ML decisions, and closed-loop intelligence.</div>
</div>
""", unsafe_allow_html=True)

# ===============================
# Sidebar Controls
# ===============================
with st.sidebar:
    st.markdown("### Assistant Controls")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state["messages"] = []
        try:
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
        except Exception:
            pass
        st.rerun()

# ===============================
# Helper Functions
# ===============================
def safe_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default

def read_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def read_json(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

def read_any(*names: str) -> pd.DataFrame:
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(SHARED, name)
        df = read_jsonl(path) if path.endswith(".jsonl") else read_json(path)
        if len(df):
            return df
    return pd.DataFrame()

def normalize_trainer_df(df: pd.DataFrame, use_tail: int = 20) -> pd.DataFrame:
    if not len(df):
        return df

    df = df.copy()
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    if "f1" in df.columns:
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # Keep the same slice the visible dashboard likely shows
    if use_tail is not None and len(df) > use_tail:
        df = df.tail(use_tail)

    df = df.dropna(subset=["accuracy", "f1"], how="all")
    return df

def get_trainer_grounding(df_tr: pd.DataFrame) -> Dict[str, object]:
    result = {
        "available": False,
        "best_acc_row": None,
        "best_f1_row": None,
        "latest_row": None,
        "acc_mean": None,
        "f1_mean": None,
    }

    if not len(df_tr):
        return result

    result["available"] = True
    result["latest_row"] = df_tr.iloc[-1]
    result["acc_mean"] = pd.to_numeric(df_tr["accuracy"], errors="coerce").mean() if "accuracy" in df_tr.columns else None
    result["f1_mean"] = pd.to_numeric(df_tr["f1"], errors="coerce").mean() if "f1" in df_tr.columns else None

    if "accuracy" in df_tr.columns and df_tr["accuracy"].notna().any():
        result["best_acc_row"] = df_tr.loc[df_tr["accuracy"].idxmax()]
    if "f1" in df_tr.columns and df_tr["f1"].notna().any():
        result["best_f1_row"] = df_tr.loc[df_tr["f1"].idxmax()]

    return result

def format_ts(ts_value) -> str:
    if pd.isna(ts_value):
        return "unknown timestamp"
    if isinstance(ts_value, pd.Timestamp):
        return ts_value.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts_value)

# ===============================
# Deterministic Trainer Answers
# ===============================
def answer_trainer_question(prompt: str) -> Optional[str]:
    q = prompt.lower().strip()

    # Detect intent
    asks_model = any(phrase in q for phrase in [
        "which model",
        "what model",
        "model achieved",
        "model got",
        "best model",
    ])
    asks_time = any(phrase in q for phrase in [
        "what time",
        "when",
        "timestamp",
        "at what time",
    ])
    asks_best_acc = "best accuracy" in q
    asks_best_f1 = ("best f1" in q) or ("best f1 score" in q) or ("best f1 score?" in q)

    trainer_keywords = asks_model or asks_time or asks_best_acc or asks_best_f1
    if not trainer_keywords:
        return None

    df_tr = read_any("trainer_metrics.jsonl")
    df_tr = normalize_trainer_df(df_tr, use_tail=20)

    if not len(df_tr):
        return "I could not find trainer metrics data."

    grounding = get_trainer_grounding(df_tr)
    best_acc_row = grounding["best_acc_row"]
    best_f1_row = grounding["best_f1_row"]

    if best_acc_row is None and best_f1_row is None:
        return "I found the trainer metrics file, but I could not compute the best accuracy or best F1 score."

    # PRIORITY 1: model questions
    if asks_model and asks_best_acc and best_acc_row is not None:
        return (
            f"Based on the trainer table, the model with the best accuracy is "
            f"{best_acc_row.get('best_model', 'unknown')} with accuracy "
            f"{safe_float(best_acc_row['accuracy']):.4f} at "
            f"{format_ts(best_acc_row.get('ts'))}."
        )

    if asks_model and asks_best_f1 and best_f1_row is not None:
        return (
            f"Based on the trainer table, the model with the best F1 score is "
            f"{best_f1_row.get('best_model', 'unknown')} with F1 "
            f"{safe_float(best_f1_row['f1']):.4f} at "
            f"{format_ts(best_f1_row.get('ts'))}."
        )

    # PRIORITY 2: time questions
    if asks_time and asks_best_acc and best_acc_row is not None:
        return (
            f"Based on the trainer table, the best accuracy "
            f"({safe_float(best_acc_row['accuracy']):.4f}) occurred at "
            f"{format_ts(best_acc_row.get('ts'))}."
        )

    if asks_time and asks_best_f1 and best_f1_row is not None:
        return (
            f"Based on the trainer table, the best F1 score "
            f"({safe_float(best_f1_row['f1']):.4f}) occurred at "
            f"{format_ts(best_f1_row.get('ts'))}."
        )

    # PRIORITY 3: value-only questions
    if asks_best_acc and best_acc_row is not None:
        return (
            f"Based on the trainer table, the best accuracy is "
            f"{safe_float(best_acc_row['accuracy']):.4f}."
        )

    if asks_best_f1 and best_f1_row is not None:
        return (
            f"Based on the trainer table, the best F1 score is "
            f"{safe_float(best_f1_row['f1']):.4f}."
        )

    # Fallback combined answer
    acc_part = ""
    f1_part = ""

    if best_acc_row is not None:
        acc_part = (
            f"the best accuracy is {safe_float(best_acc_row['accuracy']):.4f} "
            f"at {format_ts(best_acc_row.get('ts'))} using "
            f"{best_acc_row.get('best_model', 'unknown')}"
        )

    if best_f1_row is not None:
        f1_part = (
            f"the best F1 score is {safe_float(best_f1_row['f1']):.4f} "
            f"at {format_ts(best_f1_row.get('ts'))} using "
            f"{best_f1_row.get('best_model', 'unknown')}"
        )

    if acc_part and f1_part:
        return f"Based on the trainer table, {acc_part}, and {f1_part}."
    if acc_part:
        return f"Based on the trainer table, {acc_part}."
    if f1_part:
        return f"Based on the trainer table, {f1_part}."

    return None

# ===============================
# Build Context from Shared Data
# ===============================
def build_runtime_context() -> str:
    parts = []

    try:
        # --- Training summary ---
        df_tr = read_any("trainer_metrics.jsonl")
        df_tr = normalize_trainer_df(df_tr, use_tail=20)

        if len(df_tr):
            grounding = get_trainer_grounding(df_tr)
            last = grounding["latest_row"]
            acc_mean = grounding["acc_mean"]
            f1_mean = grounding["f1_mean"]
            best_acc_row = grounding["best_acc_row"]
            best_f1_row = grounding["best_f1_row"]

            parts.append(
                f"Recent training → latest model={last.get('best_model', 'n/a')}, "
                f"latest acc={safe_float(last.get('accuracy')):.3f}, "
                f"latest f1={safe_float(last.get('f1')):.3f}, "
                f"avg acc={safe_float(acc_mean):.3f}, avg f1={safe_float(f1_mean):.3f}."
            )

            if "best_model" in df_tr.columns:
                top_counts = df_tr["best_model"].value_counts().head(3)
                dom_summary = ", ".join([f"{m}: {c}" for m, c in top_counts.items()])
                parts.append(f"Model dominance (last {len(df_tr)} rows): {dom_summary}")

                grouped = (
                    df_tr.groupby("best_model")[["accuracy", "f1"]]
                    .agg(["count", "mean", "min", "max"])
                    .round(3)
                )
                grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
                per_model_stats = "; ".join(
                    [
                        f"{m}: n={int(row['accuracy_count'])}, "
                        f"acc(avg={row['accuracy_mean']}, min={row['accuracy_min']}, max={row['accuracy_max']}), "
                        f"f1(avg={row['f1_mean']}, min={row['f1_min']}, max={row['f1_max']})"
                        for m, row in grouped.iterrows()
                    ]
                )
                parts.append(f"Per-model stats → {per_model_stats}")

            if best_acc_row is not None or best_f1_row is not None:
                trainer_grounding_text = "Authoritative trainer grounding (use only this) → "

                if best_acc_row is not None:
                    trainer_grounding_text += (
                        f"best_accuracy={safe_float(best_acc_row['accuracy']):.4f} "
                        f"at ts={format_ts(best_acc_row.get('ts'))} "
                        f"with model={best_acc_row.get('best_model', 'unknown')}"
                    )

                if best_f1_row is not None:
                    if best_acc_row is not None:
                        trainer_grounding_text += "; "
                    trainer_grounding_text += (
                        f"best_f1={safe_float(best_f1_row['f1']):.4f} "
                        f"at ts={format_ts(best_f1_row.get('ts'))} "
                        f"with model={best_f1_row.get('best_model', 'unknown')}"
                    )

                trainer_grounding_text += ". Use these exact values for any question about best accuracy, best F1, model, or timestamp."
                parts.append(trainer_grounding_text)

        # --- Inference summary ---
        df_inf = read_any("inference_scores.jsonl")
        if len(df_inf):
            df_inf = df_inf.tail(2000)
            avg_p = pd.to_numeric(df_inf.get("p_sleep", 0), errors="coerce").mean()
            std_p = pd.to_numeric(df_inf.get("p_sleep", 0), errors="coerce").std()
            sleep_rate = (df_inf.get("decision") == "sleep").mean() * 100 if "decision" in df_inf else None

            msg = f"Inference → avg p_sleep={safe_float(avg_p):.3f}, std={safe_float(std_p):.3f}"
            if sleep_rate is not None:
                msg += f", sleep decisions={sleep_rate:.1f}% of total."
            parts.append(msg)

            if "cell_id" in df_inf.columns and "decision" in df_inf.columns:
                recent_df = df_inf.tail(500)
                sleep_cells = recent_df[recent_df["decision"] == "sleep"]["cell_id"].value_counts().head(5)
                keep_cells = recent_df[recent_df["decision"] == "keep"]["cell_id"].value_counts().head(5)
                wake_cells = recent_df[recent_df["decision"] == "wake"]["cell_id"].value_counts().head(5)

                if len(sleep_cells):
                    sleep_list = ", ".join([f"{cid} ({cnt})" for cid, cnt in sleep_cells.items()])
                    parts.append(f"Top cells in sleep mode (last 500 inferences): {sleep_list}")
                if len(keep_cells):
                    keep_list = ", ".join([f"{cid} ({cnt})" for cid, cnt in keep_cells.items()])
                    parts.append(f"Stable (keep) cells: {keep_list}")
                if len(wake_cells):
                    wake_list = ", ".join([f"{cid} ({cnt})" for cid, cnt in wake_cells.items()])
                    parts.append(f"Most active cells: {wake_list}")

                total_counts = recent_df["decision"].value_counts()
                parts.append(
                    f"State distribution → sleep: {total_counts.get('sleep', 0)}, "
                    f"keep: {total_counts.get('keep', 0)}, wake: {total_counts.get('wake', 0)} "
                    f"(out of {len(recent_df)} inferences)."
                )

                parts.append(
                    f"Verified from raw inference data → mean p_sleep={safe_float(avg_p):.3f}, "
                    f"std={safe_float(std_p):.3f}. Use this numeric summary instead of approximations."
                )

        # --- Actuator summary ---
        df_act = read_any("actions_log.jsonl", "actions_log.json")
        if len(df_act):
            required_cols = {"cell_id", "action", "ts"}
            if required_cols.issubset(df_act.columns):
                df_act = df_act.dropna(subset=["cell_id", "action", "ts"]).copy()
                df_act["ts"] = pd.to_datetime(df_act["ts"], errors="coerce")
                df_act = df_act.dropna(subset=["ts"])

                ps = pd.to_numeric(
                    df_act.loc[df_act["action"] == "sleep", "power_saved"],
                    errors="coerce"
                ).fillna(0)

                total_w = ps.sum() if not ps.empty else 0.0
                avg_w = ps.mean() if not ps.empty else 0.0
                min_w = ps.min() if not ps.empty else 0.0
                max_w = ps.max() if not ps.empty else 0.0
                e_j = total_w * 5

                n_sleep_total = (df_act["action"] == "sleep").sum()
                n_keep_total = (df_act["action"] == "keep").sum()
                n_wake_total = (df_act["action"] == "wake").sum()

                now_ts = df_act["ts"].max()
                cutoff = now_ts - pd.Timedelta(hours=24)
                df_recent = df_act[df_act["ts"] >= cutoff]

                n_sleep_24h = (df_recent["action"] == "sleep").sum()
                n_keep_24h = (df_recent["action"] == "keep").sum()
                n_wake_24h = (df_recent["action"] == "wake").sum()

                e_j_24h = (
                    pd.to_numeric(
                        df_recent.loc[df_recent["action"] == "sleep", "power_saved"],
                        errors="coerce"
                    ).fillna(0).sum() * 5
                )

                parts.append(
                    f"In the last 24 hours → sleep={n_sleep_24h}, keep={n_keep_24h}, wake={n_wake_24h}, "
                    f"energy_saved={e_j_24h/1000:.2f} kJ ({e_j_24h/3600:.2f} Wh)."
                )

                latest = (
                    df_act.sort_values("ts")
                    .groupby("cell_id", as_index=False)
                    .tail(1)
                )

                n_sleep_cells = (latest["action"] == "sleep").sum()
                n_keep_cells = (latest["action"] == "keep").sum()
                n_wake_cells = (latest["action"] == "wake").sum()

                ps_sleep_latest = pd.to_numeric(
                    latest.loc[latest["action"] == "sleep", "power_saved"],
                    errors="coerce"
                ).fillna(0)
                total_w_latest = ps_sleep_latest.sum()
                e_j_latest = total_w_latest * 5

                total_summary = (
                    f"Total recent actions → sleep={n_sleep_total}, "
                    f"keep={n_keep_total}, wake={n_wake_total}."
                )
                cell_summary = (
                    f"Latest per-cell states → sleep={n_sleep_cells}, keep={n_keep_cells}, wake={n_wake_cells} "
                    f"(from {len(latest)} unique cells). "
                    f"Current power_saved={total_w_latest:.2f} W, energy_saved={e_j_latest/1000:.2f} kJ ({e_j_latest/3600:.2f} Wh)."
                )

                parts.append(
                    f"Actuator → {len(df_act)} actions logged, total power_saved={total_w:.2f} W, "
                    f"avg={avg_w:.2f} W, min={min_w:.2f} W, max={max_w:.2f} W, "
                    f"energy={e_j/1000:.2f} kJ ({e_j/3600:.2f} Wh). {total_summary} {cell_summary}"
                )

                try:
                    recent_cols = [c for c in ["cell_id", "action", "score"] if c in df_act.columns]
                    recent = df_act.tail(3)[recent_cols].to_dict(orient="records")
                    if len(recent):
                        formatted_items = []
                        for r in recent:
                            score_part = ""
                            if "score" in r and pd.notna(r["score"]):
                                score_part = f" (score={safe_float(r['score']):.2f})"
                            formatted_items.append(f"{r.get('cell_id', 'unknown')} → {r.get('action', 'unknown')}{score_part}")
                        parts.append("Recent actuator actions: " + ", ".join(formatted_items))
                except Exception as e:
                    parts.append(f"(could not parse recent actions: {e})")

        # --- KPI rollups per cell ---
        def _num(s):
            return pd.to_numeric(s, errors="coerce")

        df_col = read_any("collector_log.jsonl")
        if not len(df_col):
            parts.append("(no collector data available for KPI summary)")
        else:
            df_col = df_col.dropna(subset=["cell_id"]).copy()
            if "ts" in df_col.columns:
                df_col["ts"] = pd.to_datetime(df_col["ts"], errors="coerce")

            for c in ["prb_util", "ue_count", "avg_rsrp", "avg_sinr", "ho_success_rate", "ptot"]:
                if c in df_col.columns:
                    df_col[c] = _num(df_col[c])

            needed = [c for c in ["prb_util", "ue_count", "ho_success_rate", "ptot"] if c in df_col.columns]
            if needed:
                kpi_all = df_col.groupby("cell_id")[needed].mean(numeric_only=True).reset_index()
                rows_txt = []
                for _, r in kpi_all.sort_values("cell_id").iterrows():
                    prb = f"{r['prb_util'] * 100:.2f}%" if "prb_util" in kpi_all.columns and pd.notna(r.get("prb_util")) else "n/a"
                    ue = f"{r['ue_count']:.2f}" if "ue_count" in kpi_all.columns and pd.notna(r.get("ue_count")) else "n/a"
                    ho = f"{r['ho_success_rate']:.2f}%" if "ho_success_rate" in kpi_all.columns and pd.notna(r.get("ho_success_rate")) else "n/a"
                    pwr = f"{r['ptot']:.2f} W" if "ptot" in kpi_all.columns and pd.notna(r.get("ptot")) else "n/a"
                    rows_txt.append(f"{r['cell_id']}: PRB {prb} | UE {ue} | HO SR {ho} | Pwr {pwr}")
                parts.append("Average KPIs per cell (all-time) → " + " | ".join(rows_txt))

            if "ts" in df_col.columns and df_col["ts"].notna().any():
                cutoff = datetime.utcnow() - timedelta(hours=24)
                df_24 = df_col[df_col["ts"] >= cutoff]
                if len(df_24) and needed:
                    kpi_24 = df_24.groupby("cell_id")[needed].mean(numeric_only=True).reset_index()
                    rows_txt = []
                    for _, r in kpi_24.sort_values("cell_id").iterrows():
                        prb = f"{r['prb_util'] * 100:.2f}%" if "prb_util" in kpi_24.columns and pd.notna(r.get("prb_util")) else "n/a"
                        ue = f"{r['ue_count']:.2f}" if "ue_count" in kpi_24.columns and pd.notna(r.get("ue_count")) else "n/a"
                        ho = f"{r['ho_success_rate']:.2f}%" if "ho_success_rate" in kpi_24.columns and pd.notna(r.get("ho_success_rate")) else "n/a"
                        pwr = f"{r['ptot']:.2f} W" if "ptot" in kpi_24.columns and pd.notna(r.get("ptot")) else "n/a"
                        rows_txt.append(f"{r['cell_id']}: PRB {prb} | UE {ue} | HO SR {ho} | Pwr {pwr}")
                    parts.append("Average KPIs per cell (last 24h) → " + " | ".join(rows_txt))

    except Exception as e:
        parts.append(f"(context error: {e})")

    return " | ".join(parts) if parts else "No live context available."

# ===============================
# OpenAI Setup
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set inside container.")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("RIC_LLM_MODEL", "gpt-4o-mini")

# ===============================
# Chat Interface
# ===============================
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about KPIs, models, inference, or energy savings…")

def stream_reply(system_prompt: str, user_prompt: str):
    history = st.session_state.get("messages", [])[-6:]  # reduce contamination from long old history
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_prompt}
    ]

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        temperature=0.1,
        max_tokens=700,
    )

    full = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full.append(delta)
            yield delta

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # First: deterministic exact answer for trainer "best/timestamp/model" questions
    direct_reply = answer_trainer_question(prompt)

    if direct_reply is not None:
        with st.chat_message("assistant"):
            st.markdown(direct_reply)
        st.session_state["messages"].append({"role": "assistant", "content": direct_reply})
    else:
        runtime_context = build_runtime_context()

        system_prompt = (
            "You are an expert assistant for an O-RAN Near-RT RIC demo. "
            "When answering numeric questions, use only values from lines marked "
            "'Verified from raw inference data' or 'Authoritative trainer grounding (use only this)'. "
            "Do not estimate, infer, or reuse older assistant mistakes. "
            "If authoritative values are present, restate them exactly. "
            "Always say the answer is based on verified data when such lines exist.\n\n"
            f"Context: {runtime_context}"
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            collected = []
            for token in stream_reply(system_prompt, prompt):
                collected.append(token)
                placeholder.markdown("".join(collected))
            reply = "".join(collected).strip()

        st.session_state["messages"].append({"role": "assistant", "content": reply})

# ===============================
# Persist Messages
# ===============================
try:
    os.makedirs("/persisted_shared", exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f, indent=2)
except Exception as e:
    st.warning(f"Could not save chat history: {e}")

# --- Optional daily backup ---
backup_name = f"/shared/llm_chat_history_{datetime.now().strftime('%Y-%m-%d')}.json"
try:
    with open(backup_name, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f, indent=2)
except Exception:
    pass