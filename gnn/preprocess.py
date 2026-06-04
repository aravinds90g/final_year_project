"""
preprocess.py — NF-ToN-IoT-V2 Dataset Preprocessor  (Binary: Normal vs DDoS)
==============================================================================
Reads the NF-ToN-IoT-V2.parquet file, maps attack labels to 2 classes,
builds a graph representation, and exports PyTorch Geometric Data objects.

Classes:
  0 — Normal  (Benign + everything that is NOT DDoS/DoS)
  1 — DDoS    (ddos, dos)

Graph construction:
  - Nodes  = unique virtual IPs (derived from src/dst port groups)
  - Edges  = network flows (directed)
  - Labels = per-node attack class (worst-class per node)

Node features (8 per node, aggregated from all flows of that node):
  [0] mean_in_bytes          [4] mean_protocol
  [1] mean_out_bytes         [5] mean_tcp_flags
  [2] mean_in_pkts           [6] mean_flow_duration_ms
  [3] mean_out_pkts          [7] unique_dst_ports (port diversity proxy)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "NF-ToN-IoT-V2.parquet")
OUT_DIR   = BASE_DIR

# ──────────────────────────────────────────────
# Binary label mapping  (case-insensitive)
#   0 = Normal / Benign  (everything not DDoS)
#   1 = DDoS / DoS
# ──────────────────────────────────────────────
CLASS_MAP = {
    # ── Class 0: Normal ───────────────────────
    "benign":               0,
    "normal":               0,
    # Port scan / recon → treat as Normal for binary task
    "scanning":             0,
    "vulnerability_scanner":0,
    "reconnaissance":       0,
    "vulnerabilityscanner": 0,
    # Data exfiltration → Normal for binary task
    "password":             0,
    "ransomware":           0,
    "data_exfiltration":    0,
    "theft":                0,
    # Botnet / Malware → Normal for binary task
    "backdoor":             0,
    "injection":            0,
    "xss":                  0,
    "mitm":                 0,
    "botnet":               0,
    "commandinjection":     0,
    "sqlinjection":         0,
    # ── Class 1: DDoS ─────────────────────────
    "ddos":                 1,
    "dos":                  1,
}

CLASS_NAMES = ["Normal", "DDoS"]
NUM_CLASSES = len(CLASS_NAMES)

# NetFlow numeric features used for node aggregation
FLOW_FEATURES = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "PROTOCOL",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "L4_DST_PORT",
]


def map_label(attack_str: str) -> int:
    """Map raw attack string to 0 (Normal) or 1 (DDoS)."""
    if isinstance(attack_str, str):
        key = attack_str.strip().lower()
        if key in CLASS_MAP:
            return CLASS_MAP[key]
        # Partial match fallback
        for k, v in CLASS_MAP.items():
            if k in key or key in k:
                return v
    return 0  # default Normal


def load_dataset(sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load the parquet file using pyarrow.
    sample_frac < 1.0 for development / quick runs.
    """
    print(f"[preprocess] Reading {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    print(f"[preprocess] Loaded {len(df):,} rows × {len(df.columns)} cols")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[preprocess] Sampled → {len(df):,} rows")

    return df


def _assign_virtual_ips(df: pd.DataFrame) -> pd.DataFrame:
    """
    NF-ToN-IoT-V2 has no real IP columns.
    We assign deterministic virtual IPs based on L4_SRC_PORT groups
    so the graph topology reflects realistic source diversity:
      - Normal/Benign: 192.168.1.x  (common client ports)
      - DDoS sources:  10.0.x.x     (many sources, high fan-in to victim)
    Destination is always the victim gateway: 192.168.1.100
    """
    import hashlib

    def ip_for_row(label, src_port):
        h = int(hashlib.md5(str(src_port).encode()).hexdigest(), 16) % 250 + 1
        if label == 0: return f"192.168.1.{h % 40 + 10}"   # Normal
        if label == 1: return f"10.0.{h % 254}.{h % 250 + 1}"  # DDoS
        return "192.168.1.1"

    print("[preprocess] No IP columns found → generating virtual topology from port/label…")
    df["IPV4_SRC_ADDR"] = [ip_for_row(lbl, prt)
                           for lbl, prt in zip(df["label"], df["L4_SRC_PORT"])]
    df["IPV4_DST_ADDR"] = "192.168.1.100"  # shared victim/gateway
    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Returns:
        data_batches  : list[torch_geometric.data.Data]  (one per time window)
        class_weights : torch.Tensor  (for weighted CrossEntropy)
        ip_to_idx     : dict  {ip_str -> node_index}
        scaler        : fitted StandardScaler
    """
    # ── 1. Map attack labels ──────────────────────────────────────────────
    if "Attack" in df.columns:
        df["label"] = df["Attack"].apply(map_label)
    elif "attack_type" in df.columns:
        df["label"] = df["attack_type"].apply(map_label)
    elif "Label" in df.columns:
        df["label"] = df["Label"].apply(lambda x: 1 if x == 1 else 0)
    else:
        raise ValueError("No label column found in dataset!")

    print("[preprocess] Label distribution (binary):")
    for i, name in enumerate(CLASS_NAMES):
        count = (df["label"] == i).sum()
        pct   = count / len(df) * 100
        print(f"  Class {i} ({name:10s}): {count:>10,}  ({pct:.1f}%)")

    # ── 2. Clean numeric features ─────────────────────────────────────────
    for col in FLOW_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── 3. Handle IP addresses ────────────────────────────────────────────
    if "IPV4_SRC_ADDR" not in df.columns or "IPV4_DST_ADDR" not in df.columns:
        df = _assign_virtual_ips(df)

    # ── 4. Global IP → node index mapping ────────────────────────────────
    all_ips   = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]]).unique()
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
    num_nodes = len(ip_to_idx)
    print(f"[preprocess] Unique IP nodes: {num_nodes:,}")

    # ── 5. Create temporal windows ────────────────────────────────────────
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["time_window"] = df["Timestamp"].dt.hour
    elif "FIRST_SWITCHED" in df.columns:
        df["FIRST_SWITCHED"] = pd.to_datetime(df["FIRST_SWITCHED"], errors="coerce")
        df["time_window"] = df["FIRST_SWITCHED"].dt.hour
    else:
        # Divide dataset into 24 synthetic windows
        df["time_window"] = (np.arange(len(df)) * 24 // len(df)).astype(int)

    n_windows = df["time_window"].nunique()
    print(f"[preprocess] Temporal windows: {n_windows}")

    # ── 6. Fit global scaler ──────────────────────────────────────────────
    print("[preprocess] Fitting global StandardScaler…")
    grp_global = df.groupby("IPV4_SRC_ADDR").agg(
        mean_in_bytes  = ("IN_BYTES",                   "mean"),
        mean_out_bytes = ("OUT_BYTES",                  "mean"),
        mean_in_pkts   = ("IN_PKTS",                    "mean"),
        mean_out_pkts  = ("OUT_PKTS",                   "mean"),
        mean_protocol  = ("PROTOCOL",                   "mean"),
        mean_tcp_flags = ("TCP_FLAGS",                  "mean"),
        mean_flow_dur  = ("FLOW_DURATION_MILLISECONDS", "mean"),
        uniq_dst_ports = ("L4_DST_PORT",                "nunique"),
    ).reset_index()

    feat_cols = ["mean_in_bytes", "mean_out_bytes", "mean_in_pkts", "mean_out_pkts",
                 "mean_protocol", "mean_tcp_flags", "mean_flow_dur", "uniq_dst_ports"]
    scaler = StandardScaler()
    scaler.fit(grp_global[feat_cols].fillna(0).values)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print("[preprocess] scaler.pkl saved ✓")

    # ── 7. Build per-window Data objects ─────────────────────────────────
    data_batches       = []
    node_labels_global = np.zeros(num_nodes, dtype=np.int64)

    for tw in sorted(df["time_window"].unique()):
        dw = df[df["time_window"] == tw]

        # Aggregate node features
        grp = dw.groupby("IPV4_SRC_ADDR").agg(
            mean_in_bytes  = ("IN_BYTES",                   "mean"),
            mean_out_bytes = ("OUT_BYTES",                  "mean"),
            mean_in_pkts   = ("IN_PKTS",                    "mean"),
            mean_out_pkts  = ("OUT_PKTS",                   "mean"),
            mean_protocol  = ("PROTOCOL",                   "mean"),
            mean_tcp_flags = ("TCP_FLAGS",                  "mean"),
            mean_flow_dur  = ("FLOW_DURATION_MILLISECONDS", "mean"),
            uniq_dst_ports = ("L4_DST_PORT",                "nunique"),
        ).reset_index()

        node_feat = np.zeros((num_nodes, 8), dtype=np.float32)
        for _, row in grp.iterrows():
            idx = ip_to_idx.get(row["IPV4_SRC_ADDR"])
            if idx is not None:
                node_feat[idx] = [
                    row["mean_in_bytes"], row["mean_out_bytes"],
                    row["mean_in_pkts"],  row["mean_out_pkts"],
                    row["mean_protocol"], row["mean_tcp_flags"],
                    row["mean_flow_dur"], row["uniq_dst_ports"],
                ]

        # Worst-case label per node
        grp_lbl = dw.groupby("IPV4_SRC_ADDR")["label"].max().reset_index()
        node_labels = np.zeros(num_nodes, dtype=np.int64)
        for _, row in grp_lbl.iterrows():
            idx = ip_to_idx.get(row["IPV4_SRC_ADDR"])
            if idx is not None:
                node_labels[idx] = int(row["label"])
                node_labels_global[idx] = max(node_labels_global[idx], int(row["label"]))

        # Edges
        src_idx = dw["IPV4_SRC_ADDR"].map(ip_to_idx).values
        dst_idx = dw["IPV4_DST_ADDR"].map(ip_to_idx).values
        if len(src_idx) == 0:
            continue

        edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
        x_scaled   = scaler.transform(node_feat)

        data_batches.append(Data(
            x          = torch.tensor(x_scaled,    dtype=torch.float),
            edge_index = edge_index,
            y          = torch.tensor(node_labels, dtype=torch.long),
            num_nodes  = num_nodes,
        ))

    print(f"[preprocess] Built {len(data_batches)} temporal graph batches")

    # ── 8. Class weights (inverse-frequency) ─────────────────────────────
    counts = np.bincount(node_labels_global, minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1
    class_weights = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float)

    print(f"\n[preprocess] Summary:")
    print(f"  Total nodes      : {num_nodes:,}")
    print(f"  Temporal batches : {len(data_batches)}")
    avg_edges = int(np.mean([d.edge_index.size(1) for d in data_batches]))
    print(f"  Avg edges/batch  : {avg_edges:,}")
    print(f"  Class weights    : {[round(w, 3) for w in class_weights.tolist()]}")
    print(f"  Classes          : {CLASS_NAMES}")

    return data_batches, class_weights, ip_to_idx, scaler


if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    df   = load_dataset(sample_frac=frac)
    batches, weights, ip_map, sc = preprocess(df)
    print(f"\n[preprocess] Done — {len(batches)} batches ready ✓")
