from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

# ================================================================
# Extended Measurement schema (O1-style)
# ================================================================
class Measurement(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    ng_ran_node: str = "ngran-1"
    cell_id: str

    # --- Load KPIs ---
    prb_util: float = Field(ge=0, le=1)
    ue_count: int = 0
    prb_util_ul: Optional[float] = None
    ue_idle: Optional[int] = None

    # --- QoS / RF Quality ---
    avg_rsrp: float = -95.0
    avg_sinr: float = 10.0
    avg_rsrq: Optional[float] = None
    bler: Optional[float] = None
    latency_ms: Optional[float] = None
    ho_success_rate: Optional[float] = None

    # --- Traffic KPIs ---
    downlink_mbps: float = 0.0
    uplink_mbps: float = 0.0

    # --- Power / Energy ---
    p_total_w: Optional[float] = None
    p_active_w: Optional[float] = None
    p_sleep_w: Optional[float] = None
    p_tx_w: Optional[float] = None
    pa_efficiency: Optional[float] = None

    # --- Context / Config ---
    band: Optional[str] = None
    ng_arfcn: Optional[int] = None
    channel_bw_mhz: Optional[int] = None
    hour_of_day: Optional[int] = None

    # --- Optional label for training ---
    label: Optional[int] = None


# ================================================================
# Training request and control schemas
# ================================================================
class TrainRequest(BaseModel):
    cell_ids: List[str]
    min_samples: int = 200


class Action(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    cell_id: str
    action: Literal["sleep", "wake", "keep"]
    reason: str
    score: float = 0.0


class Feedback(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    cell_id: str
    applied: bool
    success: bool
    notes: Optional[str] = None
