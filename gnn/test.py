"""
test.py — Model output verification
=====================================
Tests the trained GNN model on:
  1. Real dataset samples (per-class random samples)
  2. Full dataset slice evaluation with classification report

The model was trained on NODE-level aggregated features (mean_in_bytes,
mean_out_bytes, etc. over all flows per IP). At test time we must feed
the same aggregated representation, not raw single-flow values.

Usage:
    python3 test.py                     # use real dataset samples
    python3 test.py --sample 0.05       # also run full classification report
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, CLASS_NAMES, NUM_CLASSES
from preprocess import load_dataset, preprocess, CLASS_MAP

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "gnn_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "..", "Dataset", "NF-ToN-IoT-V2.parquet")


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        print(f"[test] ERROR: {MODEL_PATH} not found. Run train.py first.")
        sys.exit(1)

    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_model(in_channels=ckpt.get("in_channels", 8))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] Model loaded  → epoch={ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc', 0):.4f}, "
          f"val_f1={ckpt.get('val_f1', 0):.4f}")

    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("[test] Scaler loaded ✓")
    else:
        print("[test] ⚠  scaler.pkl not found")

    return model, scaler, ckpt


def map_label(attack_str):
    """Map attack string to class id (same as preprocess.py)."""
    if isinstance(attack_str, str):
        key = attack_str.strip().lower()
        if key in CLASS_MAP:
            return CLASS_MAP[key]
    return 0


def build_node_features_from_flows(flows_df, scaler):
    """
    Aggregate a group of flows (all from one IP) into the 8-dim node feature
    vector used during training.
    """
    feat = np.array([
        flows_df["IN_BYTES"].mean(),
        flows_df["OUT_BYTES"].mean(),
        flows_df["IN_PKTS"].mean(),
        flows_df["OUT_PKTS"].mean(),
        flows_df["PROTOCOL"].mean(),
        flows_df["TCP_FLAGS"].mean(),
        flows_df["FLOW_DURATION_MILLISECONDS"].mean(),
        flows_df["L4_DST_PORT"].nunique(),
    ], dtype=np.float32).reshape(1, -1)

    if scaler is not None:
        feat = scaler.transform(feat)
    return feat.squeeze()


def predict_from_node_feat(model, src_feat: np.ndarray) -> dict:
    """Predict class for a src node with its aggregated feature vector."""
    dst_feat = np.zeros(8, dtype=np.float32)
    node_feat = np.vstack([src_feat, dst_feat])

    x          = torch.tensor(node_feat, dtype=torch.float)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    with torch.no_grad():
        log_probs = model(x, edge_index)
        probs     = torch.exp(log_probs)[0].numpy()

    label_id   = int(probs.argmax())
    confidence = float(probs.max())
    return {
        "label_id":    label_id,
        "attack_type": CLASS_NAMES[label_id],
        "confidence":  confidence,
        "all_probs":   {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
    }


def run_per_class_tests(model, scaler, n_samples=5):
    """
    Load n_samples random rows per attack class from the real dataset,
    compute the aggregated node features, and predict.
    """
    print(f"\n[test] Loading dataset for per-class testing (small sample)…")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow",
                         columns=["IN_BYTES","OUT_BYTES","IN_PKTS","OUT_PKTS",
                                  "PROTOCOL","TCP_FLAGS","FLOW_DURATION_MILLISECONDS",
                                  "L4_DST_PORT","Attack"])
    df["label"] = df["Attack"].apply(map_label)

    # Ensure numeric
    for col in ["IN_BYTES","OUT_BYTES","IN_PKTS","OUT_PKTS",
                "PROTOCOL","TCP_FLAGS","FLOW_DURATION_MILLISECONDS","L4_DST_PORT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print("\n" + "="*70)
    print("  PER-CLASS REAL-DATA TESTS")
    print("="*70)
    print(f"{'Expected':<22} {'Predicted':<22} {'Conf':>7}  {'Match':>5}")
    print("-"*65)

    total, correct = 0, 0

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_df = df[df["label"] == class_id]
        if class_df.empty:
            print(f"{class_name:<22} (no samples in dataset)")
            continue

        # Take n_samples random rows as individual "flows" for this IP
        sample = class_df.sample(min(n_samples * 10, len(class_df)), random_state=42)

        # Aggregate as if these all came from one source IP
        src_feat = build_node_features_from_flows(sample, scaler)
        result   = predict_from_node_feat(model, src_feat)

        match = "✅" if result["label_id"] == class_id else "❌"
        if result["label_id"] == class_id:
            correct += 1
        total += 1

        print(f"{class_name:<22} {result['attack_type']:<22} "
              f"{result['confidence']:>7.2%}  {match:>5}")

    print("-"*65)
    print(f"Accuracy: {correct}/{total}  ({correct/total:.0%})\n")

    # Detailed probability table
    print("\nDetailed probability breakdown (real dataset aggregations):")
    header = f"{'Class':<22} " + "  ".join(f"{n:>18}" for n in CLASS_NAMES)
    print(header)
    print("-"*120)
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_df = df[df["label"] == class_id]
        if class_df.empty:
            continue
        sample   = class_df.sample(min(50, len(class_df)), random_state=42)
        src_feat = build_node_features_from_flows(sample, scaler)
        result   = predict_from_node_feat(model, src_feat)
        probs_str = "  ".join(f"{result['all_probs'][n]:>18.4f}" for n in CLASS_NAMES)
        print(f"{class_name:<22} {probs_str}")


def run_full_dataset_test(model, scaler, sample_frac=0.05):
    """Full evaluation with classification report on a dataset slice."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "="*70)
    print(f"  FULL DATASET EVALUATION ({sample_frac*100:.0f}% sample)")
    print("="*70)

    df = load_dataset(sample_frac=sample_frac)
    data_batches, _, _, _ = preprocess(df)

    all_preds, all_true = [], []
    model.eval()
    with torch.no_grad():
        for data in data_batches:
            out   = model(data.x, data.edge_index)
            preds = out.argmax(dim=1).numpy()
            all_preds.append(preds)
            all_true.append(data.y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':22}" + "".join(f"{n:>20}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:<22}" + "".join(f"{cm[i,j]:>20,}" for j in range(NUM_CLASSES))
        print(row)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        rs  = cm[i].sum()
        acc = cm[i,i] / rs if rs > 0 else 0.0
        print(f"  {name:22s}: {acc:.4f}  ({cm[i,i]:,} / {rs:,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained GNN on NF-ToN-IoT-V2")
    parser.add_argument("--sample", type=float, default=None,
                        help="If set, also run full eval on a dataset fraction (e.g. 0.05)")
    parser.add_argument("--n-per-class", type=int, default=5,
                        help="Number of flows to aggregate per class in per-class test (default=5)")
    args = parser.parse_args()

    model, scaler, ckpt = load_model_and_scaler()
    run_per_class_tests(model, scaler, n_samples=args.n_per_class)

    if args.sample:
        run_full_dataset_test(model, scaler, sample_frac=args.sample)
