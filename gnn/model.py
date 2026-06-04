"""
model.py — Graph Attention Network (GAT) for IoT Attack Detection
=================================================================
Binary classifier: Normal vs DDoS

Architecture:
  GATConv(in=8,  out=32, heads=4, concat=True)  → 128 dims
  GATConv(in=128, out=32, heads=2, concat=False) → 32  dims
  Linear(32 → 2)
  LogSoftmax → 2 classes: Normal (0), DDoS (1)

Why GAT over GCN:
  - Attention mechanism focuses on suspicious neighbors (DDoS: many-to-one)
  - Handles heterogeneous IoT topology without uniform neighbor weighting
  - Per-edge attention reveals DDoS flood patterns
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


CLASS_NAMES = ["Normal", "DDoS"]
NUM_CLASSES  = len(CLASS_NAMES)


class GATDetector(torch.nn.Module):
    """
    Two-layer Graph Attention Network for binary IoT DDoS detection.

    Args:
        in_channels  : number of node input features (default 8)
        hidden_dim   : hidden dimension before heads (default 32)
        num_classes  : output classes (default 2)
        heads1       : attention heads in layer 1 (default 4)
        heads2       : attention heads in layer 2 (default 2)
        dropout      : dropout probability (default 0.3)
    """

    def __init__(
        self,
        in_channels: int  = 8,
        hidden_dim:  int  = 32,
        num_classes: int  = NUM_CLASSES,
        heads1:      int  = 4,
        heads2:      int  = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: concat=True → output = hidden_dim × heads1
        self.gat1 = GATConv(
            in_channels, hidden_dim,
            heads=heads1, concat=True, dropout=dropout
        )

        # Layer 2: concat=False → output = hidden_dim (averaged)
        self.gat2 = GATConv(
            hidden_dim * heads1, hidden_dim,
            heads=heads2, concat=False, dropout=dropout
        )

        # Binary classifier head
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        """
        Args:
            x          : node feature matrix  [N, in_channels]
            edge_index : edge connectivity     [2, E]
        Returns:
            log_probs  : log-softmax output    [N, num_classes]
        """
        # ── Layer 1 ─────────────────────────────────────────────────────
        x = self.gat1(x, edge_index)      # [N, hidden_dim * heads1]
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 2 ─────────────────────────────────────────────────────
        x = self.gat2(x, edge_index)      # [N, hidden_dim]
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Classifier ──────────────────────────────────────────────────
        x = self.classifier(x)            # [N, num_classes]
        return F.log_softmax(x, dim=1)

    def predict_proba(self, x, edge_index):
        """Return probabilities (not log) — for inference API."""
        with torch.no_grad():
            log_probs = self.forward(x, edge_index)
            return torch.exp(log_probs)


def build_model(in_channels: int = 8) -> GATDetector:
    """Factory function used by both train.py and inference_api.py."""
    return GATDetector(in_channels=in_channels)


if __name__ == "__main__":
    # Quick smoke test
    model = build_model()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Dummy forward pass
    N, E = 100, 200
    x          = torch.randn(N, 8)
    edge_index = torch.randint(0, N, (2, E))
    out = model(x, edge_index)
    print(f"Output shape: {out.shape}  (expected [{N}, 2])")
    print("Smoke test passed ✓")
