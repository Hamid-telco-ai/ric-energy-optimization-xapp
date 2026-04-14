# CCO xApp – AI-Driven O-RAN Near-RT RIC Energy Optimization

## Overview

This project implements a **Cell On/Off (CCO) xApp** for energy-efficient 5G RAN operation, inspired by O-RAN Near-Real-Time RIC architecture.

It simulates a **closed-loop AI-driven control system** where each cell is independently optimized for energy savings based on real-time KPIs.

The system includes an **LLM-powered assistant** that explains decisions using live data.

---

## System Architecture
<img width="1652" height="908" alt="DataFlow Overview" src="https://github.com/user-attachments/assets/76beac4d-ea2e-4962-be76-1b5c0cb2cff8" />

Collector → Trainer → Inference → Actuator → Dashboard (LLM)

- **Collector**: Generates UE measurements and KPIs  
- **Trainer**: Trains ML model on labeled data  
- **Inference**: Predicts cell state (`sleep / wake / keep`)  
- **Actuator**: Applies decisions (stub for real RAN control)  
- **Dashboard + LLM**: Visualization + explainability  

All modules communicate via **Kafka**, following O-RAN data exchange principles.

---

## Features

- Cell-level energy optimization  
- ML-based decision making with fallback heuristic  
- Closed-loop pipeline (training → inference → action)  
- Real-time KPI monitoring  
- LLM assistant for explainable RIC behavior  
- Energy savings estimation based on RF power model  

---

## Input Features (per cell)

Each cell is modeled using:

[prb_util, ue_count, avg_sinr, downlink_mbps,
uplink_mbps, ho_success_rate, ptot]


---

## Algorithm and Decision Logic

### 1. Traffic Modeling (time-of-day baseline)

| Time          | PRB Utilization |
|--------------|----------------|
| Night (00–06) | 0.15 (low)     |
| Day (06–18)   | 0.55 (high)    |
| Evening       | 0.30 (moderate)|

---

### 2. Heuristic (Bootstrap Phase)

Used before enough training data is available:

```python
if prb_util < 0.25 and ue_count < 12 and ho_success_rate > 90%:
    sleep
elif prb_util > 0.6 or ue_count > 40 or ho_success_rate < 85%:
    wake
else:
    keep

---

### 3. ML-Based Decision

- Model trained on historical KPI data  
- Continuously updated  
- Predicts optimal state:

```text
sleep
wake
keep

### 4. Neighbor Awareness

- No explicit topology used  
- **HO success rate acts as a proxy**  
  - High HO success → safe to sleep  
  - Low HO success → avoid sleeping  

### 5. Energy Model

P_active = P0 + Δp * Pmax * prb_util
P_sleep  = f_sleep * P_active
power_saved = P_active - P_sleep

Where:

f_sleep = 0.2 (~80% energy savings)
RF chain is turned off, baseband remains active

### 6. Switching Logic

if load < OFF_threshold and neighbors_load OK → sleep
if OFF and neighbor load high → wake

Example thresholds:

Night: 5% ON / 10% OFF
Day: 10% ON / 20% OFF

### 3. Build and run

```bash
docker compose build
docker compose up -d

### 4. Access services
Dashboard
http://localhost:8503
LLM Assistant
Available inside the dashboard
API
http://localhost:8000/status

### 5. View logs
docker compose logs -f
