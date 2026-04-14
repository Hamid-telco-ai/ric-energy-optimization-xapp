import os, json
from datetime import datetime
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import joblib

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "coo_model.joblib")
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "20"))

consumer = KafkaConsumer(
    "ue.measurements",
    bootstrap_servers=BROKER,
    value_deserializer=lambda v: json.loads(v.decode()),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="trainer-v1"
)

buffer = []

FEATURES = [
    "prb_util", "ue_count", "avg_sinr", "downlink_mbps", "uplink_mbps",
    "ho_success_rate", "ptot"
]

def to_df(data):
    df = pd.DataFrame(data)
    return df

def build_models():
    models = []
    models.append(("logreg",
                   Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))])))
    models.append(("rf", RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1)))
    models.append(("xgb", xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, objective="binary:logistic", n_jobs=4, eval_metric="logloss"
    )))
    models.append(("gb", GradientBoostingClassifier()))
    models.append(("svc", Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))])))
    models.append(("knn", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))])))
    return models

def evaluate_and_pick(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    best = None
    report = []
    for name, model in build_models():
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        acc = accuracy_score(yte, yp)
        prec = precision_score(yte, yp, zero_division=0)
        rec  = recall_score(yte, yp, zero_division=0)
        f1   = f1_score(yte, yp, zero_division=0)
        cm   = confusion_matrix(yte, yp)
        report.append((name, acc, prec, rec, f1, cm, model))
    # pick best
    report.sort(key=lambda r: r[4], reverse=True)
    print("\n📊 [trainer] MODEL PERFORMANCE")
    for name, acc, prec, rec, f1, cm, _ in report:
        print(f"  {name:6s} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}\n    CM=\n{cm}")
    #best = report[0][-1]
    best_name = report[0][0]
    best_acc = report[0][1]
    best_f1 = report[0][4]
    best_model = report[0][-1]

    print(f"[trainer] ✅ best model: {best_name}")
    joblib.dump(best_model, MODEL_PATH)
    #return best_model_name, best_acc, best_f1
    print(f"[trainer] ✅ saved → {MODEL_PATH}")
    #return best_name, best_acc, best_prec, best_rec, best_f1
    return best_name, best_acc, best_f1

def main():
    print(f"[trainer] waiting for labeled samples… target={MIN_SAMPLES}")
    for msg in consumer:
        m = msg.value
        if m.get("label") in (0, 1):
            buffer.append(m)

        if len(buffer) >= MIN_SAMPLES:
            df = to_df(buffer)
            df = df.dropna(subset=["label"])
            X = df[FEATURES].values
            y = df["label"].astype(int).values

            if len(set(y)) < 2:
                print("[trainer] ⚠ only one class present; keep buffering…")
                continue

            # ---- Train and evaluate all models ----
            best_model_name, best_acc, best_f1 = evaluate_and_pick(X, y)

            # ---- Log training metrics for dashboard ----
            metrics = {
                "ts": datetime.now().isoformat(),
                "best_model": best_model_name,
                "accuracy": best_acc,
                "f1": best_f1,
                "samples": len(X)
            }
            os.makedirs("/shared", exist_ok=True)
            with open("/shared/trainer_metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")

            print(f"[trainer] logged metrics: {metrics}")
            buffer.clear()
            print(f"[trainer] 🔄 cleared buffer; waiting for next {MIN_SAMPLES} samples")


if __name__ == "__main__":
    main()
