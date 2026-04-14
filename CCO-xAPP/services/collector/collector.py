import os, json, time, random
from datetime import datetime, date
from kafka import KafkaProducer
from common.schema import Measurement

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")

def json_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda v: json.dumps(v, default=json_serializer).encode()
)

cells = ["cell-A", "cell-B", "cell-C"]
bands = {"ngran-1": 3.5, "ngran-2": 3.7}  # GHz


# Maintain last known PRB per cell (for neighbor visibility)
last_prb = {c: 0.3 for c in cells}


def simulate_measurement(cell_id, node):
    """Generate realistic O1-style KPI data for congestion/sleep modeling."""

    # --- Load profile simulation (diurnal pattern + randomness) ---
    t = time.time()
    hour = int((t / 3600) % 24)
    base_util = 0.15 if 0 <= hour < 6 else (0.55 if 6 <= hour < 18 else 0.3)
    prb_util = round(max(0.05, min(0.95, base_util + random.uniform(-0.15, 0.15))), 2)
    ue_count = max(1, int(prb_util * 100 + random.randint(-10, 10)))

    # Update last PRB state
    last_prb[cell_id] = prb_util

    # --- RF quality ---
    ho_sr = round(random.uniform(85, 99), 2)
    avg_sinr = round(10 + random.uniform(-5, 5), 2)
    avg_rsrp = round(-95 + random.uniform(-10, 10), 2)

    # --- TX power (watts) ---
    ptot = round(10 + (prb_util * 100 * random.uniform(0.4, 0.8)), 2)

    # --- Labeling (O-RAN G.1 style) ---
    if prb_util < 0.5 and ue_count < 45 and ho_sr > 88:
        label = 1     # Sleep candidate
    elif prb_util > 0.65 or ue_count > 60 or ho_sr < 85:
        label = 0     # Busy / active
    else:
        label = random.choice([0, 1])  # Mid-load region

    # Build neighbor load dictionary (rho_n)
    neighbors = {
        c: last_prb[c]
        for c in cells if c != cell_id
    }

    # --- Compose measurement dict ---
    meas = {
        "ng_ran_node": node,
        "cell_id": cell_id,
        "prb_util": prb_util,
        "ue_count": ue_count,
        "avg_rsrp": avg_rsrp,
        "avg_sinr": avg_sinr,
        "downlink_mbps": round(prb_util * 200 + random.uniform(-5, 5), 2),
        "uplink_mbps": round(prb_util * 50 + random.uniform(-2, 2), 2),
        "ho_success_rate": ho_sr,
        "ptot": ptot,
        "label": label,

        # SWES neighbor visibility (rho_n)
        "neighbors": neighbors,

        "ts": datetime.utcnow().isoformat(),
    }

    return meas


def main():
    print("[collector] O1-style KPI generator with SWES neighbor visibility…")
    while True:
        for c in cells:
            for node in bands:
                m = simulate_measurement(c, node)
                producer.send("ue.measurements", m)

                pt = m.get("ptot", 0.0)
                neigh = ", ".join([f"{k}:{v:.2f}" for k, v in m["neighbors"].items()])
                print(
                    f"[SENT] {node}/{c}: "
                    f"PRB={m.get('prb_util', 0):.2f}, "
                    f"UEs={m.get('ue_count', 0)}, "
                    f"HO_SR={m.get('ho_success_rate', 0)}%, "
                    f"Ptot={pt}W, "
                    f"Neighbors=({neigh}), "
                    f"Label={m.get('label', '?')}"
                )

                # --- Log raw measurements for dashboard ---
                with open("/shared/collector_log.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")

        producer.flush()
        time.sleep(2)


if __name__ == "__main__":
    main()