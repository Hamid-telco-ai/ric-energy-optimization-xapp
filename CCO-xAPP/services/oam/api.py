from fastapi import FastAPI
import os, json
from datetime import datetime

app = FastAPI(title="COO xAPP OAM", version="1.0.0")
SHARED_DIR = "/shared"
MODEL_FILE = os.path.join(SHARED_DIR, "model_status.json")
ACTIONS_FILE = os.path.join(SHARED_DIR, "actions.log")

@app.get("/health", summary="Health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/model/status", summary="Model Status")
def model_status():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE) as f:
            return json.load(f)
    return {"status": "No model info yet"}

@app.get("/actions/recent", summary="Recent Actions")
def recent_actions(limit: int = 100):
    if os.path.exists(ACTIONS_FILE):
        with open(ACTIONS_FILE) as f:
            lines = f.readlines()[-limit:]
        return {"actions": [line.strip() for line in lines]}
    return {"status": "No actions recorded"}
