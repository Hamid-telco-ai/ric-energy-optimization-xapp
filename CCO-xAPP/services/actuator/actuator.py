import os, json, time
from datetime import datetime
from kafka import KafkaConsumer

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
consumer = KafkaConsumer(
    "xapp.actions",
    bootstrap_servers=BROKER,
    value_deserializer=lambda v: json.loads(v.decode()),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="actuator"
)

SHARED = "/shared"
LOG_PATH = os.path.join(SHARED, "actions_log.jsonl")
print(f"[actuator] writing will go to: {LOG_PATH}")

def apply_action(act):
    ts = datetime.utcnow().isoformat()
    entry = {
        "ts": ts,
        "cell_id": act.get("cell_id"),
        "action": act.get("action"),
        "score": act.get("score"),
        "reason": act.get("reason"),
        "ptot_active": act.get("ptot_active", None),
        "power_saved": act.get("power_saved", 0.0)
    }

    print(f"[actuator] {entry['cell_id']} -> {entry['action']} "
          f"(score={entry['score']:.2f}) reason={entry['reason']}")

    # Ensure shared folder exists
    os.makedirs(SHARED, exist_ok=True)

    # Append to file and flush immediately
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()  #ensures immediate visibility for dashboard

def main():
    print(f"[actuator] Listening on topic xapp.actions @ {BROKER}")
    os.makedirs(SHARED, exist_ok=True)
    while True:
        for msg in consumer:
            apply_action(msg.value)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
