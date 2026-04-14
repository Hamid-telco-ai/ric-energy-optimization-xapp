import os, json, time, joblib, pathlib, sys
from datetime import datetime, date
from kafka import KafkaConsumer, KafkaProducer

# --- Import common schema ---
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from common.schema import Action

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "coo_model.joblib")

# --- JSON serializer helper ---
def json_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda v: json.dumps(v, default=json_serializer).encode()
)

consumer = KafkaConsumer(
    "ue.measurements",
    bootstrap_servers=BROKER,
    value_deserializer=lambda v: json.loads(v.decode()),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="inference-v1"
)

model = None

# --------------------------------------------------------
# Hysteresis and Stability Parameters (Anti Ping-Pong)
# --------------------------------------------------------
T_OFF = float(os.getenv("T_OFF", "0.25"))   # Sleep threshold
T_ON  = float(os.getenv("T_ON",  "0.40"))   # Wake threshold
N_CYCLES = int(os.getenv("N_CYCLES", "3"))  # Stability window

# SWES(1,1): fraction of local load that will be transferred to neighbors
SPILL_FACTOR = float(os.getenv("SPILL_FACTOR", "0.3"))

# Per-cell state memory
cell_state = {}        # "ON" or "OFF"
sleep_counter = {}     # counts consecutive sleep conditions
wake_counter = {}      # counts consecutive wake conditions
last_fb = {}           # store last network-impact value F_b per cell

# --------------------------------------------------------
# Load model if available
# --------------------------------------------------------
def load_model_if_available():
    global model
    p = pathlib.Path(MODEL_PATH)
    if p.exists():
        model = joblib.load(p)
        print(f"[inference] model loaded: {p}")
    else:
        print(f"[inference] no model at {p}")

# --------------------------------------------------------
# Simple heuristic (fallback logic)
# --------------------------------------------------------
def heuristic(m):
    util = m["prb_util"]
    ue = m["ue_count"]
    ho = m["ho_success_rate"]
    # Simple G.1-style rule
    if util < 0.25 and ue < 12 and ho > 90:
        return "sleep", 0.55, "heuristic low-load stable"
    if util > 0.6 or ue > 40 or ho < 85:
        return "wake", 0.60, "heuristic busy/unstable"
    return "keep", 0.50, "heuristic neutral"

# --------------------------------------------------------
# Main inference loop
# --------------------------------------------------------
def main():
    load_model_if_available()
    print("[inference] listening…")
    os.makedirs("/shared", exist_ok=True)

    for msg in consumer:
        m = msg.value

        xi = [[
            m.get("prb_util", 0.0),
            m.get("ue_count", 0),
            m.get("avg_sinr", 0.0),
            m.get("downlink_mbps", 0.0),
            m.get("uplink_mbps", 0.0),
            m.get("ho_success_rate", 0.0),
            m.get("ptot", 0.0)
        ]]

        # --- Predict or use heuristic ---
        if model is not None:
            try:
                proba_sleep = float(model.predict_proba(xi)[0][1])
                action = "sleep" if proba_sleep > 0.58 else ("wake" if proba_sleep < 0.48 else "keep")
                reason = f"ML p(sleep)={proba_sleep:.2f}"
                score = proba_sleep
            except Exception as e:
                action, score, reason = heuristic(m)
                reason = f"heuristic fallback: {reason} ({e})"
        else:
            action, score, reason = heuristic(m)

        # --------------------------------------------------------
        # SWES(1,1)-style Network Impact F_b
        # --------------------------------------------------------
        cid = m["cell_id"]
        prb = float(m.get("prb_util", 0.0))

        neighbors = m.get("neighbors", {}) or {}
        # ensure numeric
        neighbors = {k: float(v) for k, v in neighbors.items() if v is not None}

        if neighbors:
            rho_b = prb
            rho_transfer = SPILL_FACTOR * rho_b
            # F_b = max_n (rho_n + rho_{b->n})
            Fb = max(v + rho_transfer for v in neighbors.values())
        else:
            # no neighbors, fall back to local load
            Fb = prb

        last_fb[cid] = Fb

        # --------------------------------------------------------
        # Hysteresis + N-Cycle Stability Filter (FINAL Decision)
        # --------------------------------------------------------
        # Initialize memory for new cell
        if cid not in cell_state:
            cell_state[cid] = "ON"          # assume initially ON
            sleep_counter[cid] = 0
            wake_counter[cid] = 0

        final_action = "keep"

        # CASE 1: Cell is ON → Try to Sleep
        # sleep allowed only if: local load low AND neighbors can absorb traffic (F_b < T_OFF)
        if cell_state[cid] == "ON":
            if prb < T_OFF and Fb < T_OFF:
                sleep_counter[cid] += 1
                if sleep_counter[cid] >= N_CYCLES:
                    final_action = "sleep"
                    cell_state[cid] = "OFF"
                    sleep_counter[cid] = 0
                    wake_counter[cid] = 0
            else:
                sleep_counter[cid] = 0

        # CASE 2: Cell is OFF → Try to Wake
        elif cell_state[cid] == "OFF":
            # simple rule: wake if local potential load is high or neighbors are stressed
            if prb > T_ON or Fb > T_ON:
                wake_counter[cid] += 1
                if wake_counter[cid] >= N_CYCLES:
                    final_action = "wake"
                    cell_state[cid] = "ON"
                    wake_counter[cid] = 0
                    sleep_counter[cid] = 0
            else:
                wake_counter[cid] = 0

        # Override previous ML/heuristic action with hysteresis + SWES decision
        action = final_action
        reason = f"SWES(1,1) hysteresis: Fb={Fb:.2f}, Δh={T_ON - T_OFF:.2f}, N={N_CYCLES}"

        # --------------------------------------------------------
        # Power Saving Estimation (3GPP style)
        # --------------------------------------------------------
        P0, Pmax, delta_p, f_sleep = 130, 260, 0.5, 0.2  # macro baseline
        prb_util = float(m.get("prb_util", 0.5))
        P_active = P0 + delta_p * Pmax * prb_util

        if action == "sleep":
            P_sleep = f_sleep * P_active
            power_saved = P_active - P_sleep
        else:
            power_saved = 0.0

        # --------------------------------------------------------
        # Construct enriched message
        # --------------------------------------------------------
        out = Action(cell_id=m["cell_id"], action=action, reason=reason, score=score).model_dump()
        out_with_ts = {
            "ts": datetime.utcnow().isoformat(),
            **out,
            "prb_util": prb_util,
            "ptot_active": round(P_active, 2),
            "power_saved": round(power_saved, 2),
            "Fb": round(Fb, 3),
        }

        # Send enriched message to Kafka
        producer.send("xapp.actions", out_with_ts)
        print(f"[inference] → {out_with_ts}")

        # --------------------------------------------------------
        # Log inference scores (for dashboard)
        # --------------------------------------------------------
        try:
            # --- Inline Power Saving Calculation for Inference Table ---
            P0, Pmax, delta_p, f_sleep = 130, 260, 0.5, 0.2  # macro baseline
            prb_util = float(m.get("prb_util", 0.5))
            P_active = P0 + delta_p * Pmax * prb_util
            P_sleep = f_sleep * P_active if action == "sleep" else P_active
            power_saved = P_active - P_sleep if action == "sleep" else 0.0

            result = {
                "ts": datetime.utcnow().isoformat(),
                "cell_id": m["cell_id"],
                "p_sleep": float(score),
                "decision": action,
                "power_saved": round(power_saved, 2),
                "state": cell_state.get(cid),
                "sleep_counter": sleep_counter.get(cid, 0),
                "wake_counter": wake_counter.get(cid, 0),
                "T_OFF": T_OFF,
                "T_ON": T_ON,
                "N": N_CYCLES,
                "Fb": round(Fb, 3),
                "spill_factor": SPILL_FACTOR
            }

            with open("/shared/inference_scores.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            print(f"[inference] could not log inference_scores: {e}")

        # --------------------------------------------------------
        # Append combined action log for OAM/dashboard
        # --------------------------------------------------------
        try:
            safe_json = json.dumps(out_with_ts, default=str)
            log_file = pathlib.Path("/shared/actions_log.jsonl")
            with open(log_file, "a") as f:
                f.write(safe_json + "\n")
        except Exception as e:
            print(f"[inference] log write error: {e}")


if __name__ == "__main__":
    main()