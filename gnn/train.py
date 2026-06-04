"""
train.py — GNN Training Script
================================
Trains the GATDetector on NF-ToN-IoT-V2 dataset with:
  - Proper 80/20 train/val split
  - Temporal batch processing (memory-efficient)
  - Weighted cross-entropy (handles class imbalance)
  - Early stopping
  - Per-class accuracy & F1 report after training

Usage:
    python3 train.py                   # 20% data (default, ~5 min)
    python3 train.py --sample 0.5      # 50% data
    python3 train.py --sample 1.0      # full dataset
    python3 train.py --epochs 50       # quick test

Saves:
    gnn_model.pth     — best model weights
    scaler.pkl        — StandardScaler for inference
    training_log.json — per-epoch history
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_dataset, preprocess, CLASS_NAMES
from model import build_model

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUT_DIR, "gnn_model.pth")
LOG_PATH   = os.path.join(OUT_DIR, "training_log.json")


# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, batches, optimizer, class_weights, device, mini_batch=4):
    """One full training epoch over temporal batches."""
    model.train()
    total_loss  = 0.0
    num_updates = 0

    for i in range(0, len(batches), mini_batch):
        chunk = batches[i : i + mini_batch]
        optimizer.zero_grad()

        chunk_loss = torch.tensor(0.0, device=device)
        for data in chunk:
            x          = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y          = data.y.to(device)
            out        = model(x, edge_index)
            chunk_loss = chunk_loss + F.nll_loss(out, y, weight=class_weights.to(device))

        chunk_loss = chunk_loss / len(chunk)
        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += chunk_loss.item()
        num_updates += 1

    return total_loss / max(num_updates, 1)


@torch.no_grad()
def evaluate(model, batches, device):
    """Evaluate over a list of temporal graphs, return (acc, macro_f1, all_preds, all_true)."""
    model.eval()
    all_preds = []
    all_true  = []

    for data in batches:
        x          = data.x.to(device)
        edge_index = data.edge_index.to(device)
        out        = model(x, edge_index)
        preds      = out.argmax(dim=1).cpu().numpy()
        labels     = data.y.cpu().numpy()
        all_preds.append(preds)
        all_true.append(labels)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = float((y_pred == y_true).mean())
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1, y_pred, y_true


# ─────────────────────────────────────────────────────────────────────────────

def train(
    sample_frac: float = 0.2,
    epochs:      int   = 100,
    lr:          float = 1e-3,
    patience:    int   = 15,
    mini_batch:  int   = 4,
    val_split:   float = 0.2,
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device : {device}")
    print(f"[train] Sample : {sample_frac*100:.0f}%  |  Epochs: {epochs}  |  LR: {lr}")

    # ── Load & preprocess ──────────────────────────────────────────────────
    df = load_dataset(sample_frac=sample_frac)
    data_batches, class_weights, _, _ = preprocess(df)

    if not data_batches:
        print("[train] ERROR: No temporal batches generated. Aborting.")
        return

    # ── Train / Val split (on temporal batches) ───────────────────────────
    n_val    = max(1, int(len(data_batches) * val_split))
    n_train  = len(data_batches) - n_val
    train_batches = data_batches[:n_train]
    val_batches   = data_batches[n_train:]
    print(f"[train] Split  : {len(train_batches)} train  |  {len(val_batches)} val batches")

    # ── Model ─────────────────────────────────────────────────────────────
    in_channels = data_batches[0].num_node_features
    model       = build_model(in_channels=in_channels).to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-5)

    print(f"\n[train] Starting {epochs}-epoch training …\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Tr Acc':>7} | {'Tr F1':>7} | {'Val Acc':>7} | {'Val F1':>7} | {'Time':>6}")
    print("-" * 72)

    best_val_f1  = 0.0
    best_epoch   = 0
    no_improve   = 0
    history      = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        loss              = train_epoch(model, train_batches, optimizer, class_weights, device, mini_batch)
        tr_acc, tr_f1, _, _  = evaluate(model, train_batches, device)
        va_acc, va_f1, _, _  = evaluate(model, val_batches,   device)

        scheduler.step(va_f1)
        elapsed = time.time() - t0

        history.append({
            "epoch":     epoch,
            "loss":      round(loss,   4),
            "train_acc": round(tr_acc, 4),
            "train_f1":  round(tr_f1,  4),
            "val_acc":   round(va_acc, 4),
            "val_f1":    round(va_f1,  4),
        })

        print(f"{epoch:>6} | {loss:>8.4f} | {tr_acc:>7.4f} | {tr_f1:>7.4f} | "
              f"{va_acc:>7.4f} | {va_f1:>7.4f} | {elapsed:>5.1f}s")

        # Save best model
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     va_acc,
                "val_f1":      va_f1,
                "in_channels": in_channels,
                "class_names": CLASS_NAMES,
            }, MODEL_PATH)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n[train] Early stopping at epoch {epoch}  "
                  f"(best val_f1={best_val_f1:.4f} @ epoch {best_epoch})")
            break

    # ── Load best & final evaluation ──────────────────────────────────────
    print(f"\n[train] Loading best model (epoch {best_epoch}) …")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, _, y_pred, y_true = evaluate(model, val_batches, device)

    print("\n" + "="*60)
    print("  FINAL CLASSIFICATION REPORT (Validation Set)")
    print("="*60)
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0))

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    print("Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        row_sum = cm[i].sum()
        per_acc = cm[i, i] / row_sum if row_sum > 0 else 0.0
        print(f"  {name:22s}: {per_acc:.4f}  ({cm[i,i]:,} / {row_sum:,})")

    print(f"\n[train] Best epoch   : {best_epoch}")
    print(f"[train] Best val F1  : {best_val_f1:.4f}")
    print(f"[train] Model saved  : {MODEL_PATH}")

    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Log saved    : {LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN on NF-ToN-IoT-V2")
    parser.add_argument("--sample",      type=float, default=0.2,
                        help="Fraction of dataset (default=0.2)")
    parser.add_argument("--epochs",      type=int,   default=100,
                        help="Max training epochs (default=100)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate (default=1e-3)")
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early stopping patience (default=15)")
    parser.add_argument("--mini-batch",  type=int,   default=4,
                        help="Temporal graphs per gradient step (default=4)")
    parser.add_argument("--val-split",   type=float, default=0.2,
                        help="Fraction of batches for validation (default=0.2)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GNN TRAINING — NF-ToN-IoT-V2")
    print("="*60)
    print(f"  Sample fraction : {args.sample*100:.0f}%")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Val split       : {args.val_split*100:.0f}%")
    print("="*60 + "\n")

    train(
        sample_frac = args.sample,
        epochs      = args.epochs,
        lr          = args.lr,
        patience    = args.patience,
        mini_batch  = args.mini_batch,
        val_split   = args.val_split,
    )
