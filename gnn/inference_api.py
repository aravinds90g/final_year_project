"""
inference_api.py — FastAPI Inference Server  (Binary: Normal vs DDoS)
======================================================================
Runs on port 5001.
Receives individual flow records from the Node.js backend,
builds a mini-graph, and returns Normal / DDoS classification.

Endpoints:
    GET  /health         → { status, model_loaded, classes }
    POST /predict        → { attack_type, confidence, label_id, all_probs }
    POST /predict/batch  → list of predictions for multiple flows
    GET  /classes        → class definitions with SDN actions
"""

import os
import sys
import json
import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, CLASS_NAMES, NUM_CLASSES

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "gnn_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ──────────────────────────────────────────────
# App & global state
# ──────────────────────────────────────────────
app = FastAPI(
    title="GNN IoT DDoS Detector",
    description="Real-time IoT DDoS detection using Graph Attention Network (binary classifier)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL  = None
SCALER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_scaler():
    """Load GNN model and feature scaler at startup."""
    global MODEL, SCALER

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python3 train.py` first."
        )

    ckpt   = torch.load(MODEL_PATH, map_location=DEVICE)
    MODEL  = build_model(in_channels=ckpt.get("in_channels", 8)).to(DEVICE)
    MODEL.load_state_dict(ckpt["model_state"])
    MODEL.eval()
    print(f"[inference_api] Model loaded (epoch={ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc', '?'):.4f})")
    print(f"[inference_api] Classes: {CLASS_NAMES}")

    if os.path.exists(SCALER_PATH):
        SCALER = joblib.load(SCALER_PATH)
        print("[inference_api] Scaler loaded ✓")
    else:
        print("[inference_api] ⚠  scaler.pkl not found — features won't be scaled")


@app.on_event("startup")
def startup_event():
    load_model_and_scaler()


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class FlowRecord(BaseModel):
    """
    Single network flow record from an IoT device (ESP8266 → backend → here).
    All fields match what the ESP8266 firmware sends.
    """
    src_ip:            str
    dst_ip:            str
    in_bytes:          float = 0
    out_bytes:         float = 0
    in_pkts:           float = 0
    out_pkts:          float = 0
    flow_duration_ms:  float = 0
    protocol:          float = 6    # 6=TCP, 17=UDP, 1=ICMP
    src_port:          float = 0
    dst_port:          float = 80
    tcp_flags:         float = 0
    src_to_dst_bps:    float = 0


class PredictionResult(BaseModel):
    src_ip:      str
    dst_ip:      str
    attack_type: str         # "Normal" or "DDoS"
    label_id:    int         # 0=Normal, 1=DDoS
    confidence:  float
    all_probs:   dict        # { "Normal": p0, "DDoS": p1 }
    blocked:     bool = False


# ──────────────────────────────────────────────
# Feature engineering (mirrors preprocess.py)
# ──────────────────────────────────────────────

# ── Per-IP flow accumulator ────────────────────────────────────────────────
# Training used PER-IP AGGREGATED features (mean over all flows for that IP).
# At inference we maintain a sliding window of recent flows per IP to compute
# the same aggregated statistics the model expects.
from collections import defaultdict, deque

IP_WINDOW_SIZE = 5    # 5-flow window: Normal builds unique_dst_ports=3-5; DDoS stays at 1
_ip_flow_history: dict = defaultdict(lambda: deque(maxlen=IP_WINDOW_SIZE))

def flow_to_node_features(flow: FlowRecord) -> np.ndarray:
    """
    Convert a flow record to an 8-dim AGGREGATED node feature vector.

    Training features (preprocess.py):
      [0] mean_in_bytes
      [1] mean_out_bytes
      [2] mean_in_pkts
      [3] mean_out_pkts
      [4] mean_protocol
      [5] mean_tcp_flags
      [6] mean_flow_duration_ms
      [7] unique_dst_ports    ← nunique over the IP's flows, NOT the raw port #

    We accumulate the last IP_WINDOW_SIZE flows and compute the same aggregation.
    """
    src_ip = flow.src_ip
    history = _ip_flow_history[src_ip]

    # Add current flow snapshot to history
    history.append({
        "in_bytes":         flow.in_bytes,
        "out_bytes":        flow.out_bytes,
        "in_pkts":          flow.in_pkts,
        "out_pkts":         flow.out_pkts,
        "protocol":         flow.protocol,
        "tcp_flags":        flow.tcp_flags,
        "flow_duration_ms": flow.flow_duration_ms,
        "dst_port":         flow.dst_port,
    })

    # Compute aggregated features (mirrors preprocess.py groupby.agg)
    mean_in_bytes  = np.mean([h["in_bytes"]         for h in history])
    mean_out_bytes = np.mean([h["out_bytes"]         for h in history])
    mean_in_pkts   = np.mean([h["in_pkts"]           for h in history])
    mean_out_pkts  = np.mean([h["out_pkts"]          for h in history])
    mean_protocol  = np.mean([h["protocol"]          for h in history])
    mean_tcp_flags = np.mean([h["tcp_flags"]          for h in history])
    mean_flow_dur  = np.mean([h["flow_duration_ms"]  for h in history])
    uniq_dst_ports = float(len(set(h["dst_port"] for h in history)))  # nunique

    features = np.array([
        mean_in_bytes,
        mean_out_bytes,
        mean_in_pkts,
        mean_out_pkts,
        mean_protocol,
        mean_tcp_flags,
        mean_flow_dur,
        uniq_dst_ports,
    ], dtype=np.float32)
    return features


def predict_single(flow: FlowRecord) -> PredictionResult:
    """
    Build a 2-node mini-graph (src → dst) and run the GAT model.
    Returns Normal (0) or DDoS (1).
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    # ── Node features ──────────────────────────────────────────────────
    src_feat = flow_to_node_features(flow)
    dst_feat = np.zeros(8, dtype=np.float32)   # dst node has no extra info

    node_features = np.vstack([src_feat, dst_feat])  # shape [2, 8]

    # Scale if scaler is available
    if SCALER is not None:
        node_features = SCALER.transform(node_features)

    # ── Edge: src(0) → dst(1) ─────────────────────────────────────────
    edge_index = torch.tensor([[0], [1]], dtype=torch.long).to(DEVICE)
    x          = torch.tensor(node_features, dtype=torch.float).to(DEVICE)

    # ── Forward pass ───────────────────────────────────────────────────
    with torch.no_grad():
        log_probs = MODEL(x, edge_index)
        probs     = torch.exp(log_probs)[0].cpu().numpy()   # src node prediction

    label_id    = int(probs.argmax())
    confidence  = float(probs.max())
    attack_type = CLASS_NAMES[label_id]
    all_probs   = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    return PredictionResult(
        src_ip      = flow.src_ip,
        dst_ip      = flow.dst_ip,
        attack_type = attack_type,
        label_id    = label_id,
        confidence  = round(confidence, 4),
        all_probs   = all_probs,
        blocked     = label_id == 1,   # block if DDoS
    )


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "device":       str(DEVICE),
        "classes":      CLASS_NAMES,
        "num_classes":  NUM_CLASSES,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(flow: FlowRecord):
    """
    Predict Normal or DDoS for a single flow record.
    Called by the Node.js backend when a device sends traffic data.
    """
    try:
        return predict_single(flow)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(flows: List[FlowRecord]):
    """
    Predict Normal or DDoS for multiple flow records at once.
    """
    try:
        return [predict_single(f) for f in flows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
def get_classes():
    """Return the 2 class definitions and their SDN actions."""
    return {
        "classes": [
            {"id": 0, "name": "Normal", "sdn_action": "Allow"},
            {"id": 1, "name": "DDoS",   "sdn_action": "Block source IP"},
        ]
    }


if __name__ == "__main__":
    print(f"[inference_api] Starting on http://0.0.0.0:5001")
    print(f"[inference_api] Binary classifier: {CLASS_NAMES}")
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)
