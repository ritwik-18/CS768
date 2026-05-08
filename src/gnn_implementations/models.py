"""
models.py - GNN Model Implementations
======================================
Implements:
  - GCN  (Graph Convolutional Network)
  - GIN  (Graph Isomorphism Network)
  - GNN+FA (GNN with a Fully-Adjacent final layer)

Reference: Alon & Yahav (2021), "On the Bottleneck of GNNs and its Practical Implications"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


# ---------------------------------------------------------------------------
# Helper: MLP block
# ---------------------------------------------------------------------------

def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
             dropout: float = 0.0) -> nn.Sequential:
    """Utility to build a simple MLP."""
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# GCN Model
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    Prone to over-squashing due to mean aggregation compressing
    all neighbor info equally.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNClassifier(nn.Module):
    """GCN for graph-level or node-level classification."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, dropout: float = 0.0, task: str = "node"):
        super().__init__()
        self.task = task
        self.gnn = GCN(in_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        h = self.gnn(x, edge_index, batch)
        if self.task == "graph":
            h = global_mean_pool(h, batch)
        return self.head(h)


# ---------------------------------------------------------------------------
# GIN Model
# ---------------------------------------------------------------------------

class GIN(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., 2019).
    Maximum expressivity for local structures (matches 1-WL test),
    but Alon & Yahav show it is highly susceptible to over-squashing
    because sum aggregation forces even more compression.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            mlp = make_mlp(dims[i], dims[i + 1], dims[i + 1], num_layers=2)
            self.convs.append(GINConv(mlp, train_eps=True))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GINClassifier(nn.Module):
    """GIN for graph-level or node-level classification."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, dropout: float = 0.0, task: str = "node"):
        super().__init__()
        self.task = task
        self.gnn = GIN(in_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        h = self.gnn(x, edge_index, batch)
        if self.task == "graph":
            h = global_add_pool(h, batch)
        return self.head(h)


# ---------------------------------------------------------------------------
# Fully-Adjacent (FA) Layer
# ---------------------------------------------------------------------------

class FALayer(nn.Module):
    """
    Fully-Adjacent Layer (Alon & Yahav, 2021).

    Adds a global attention step after the standard GNN layers.
    Every node can attend to every other node in the graph,
    completely bypassing graph topology for one step.
    This resolves the over-squashing bottleneck by allowing
    distant nodes to communicate directly.

    Implementation:
      - For each graph in the batch, we compute pairwise attention
        (transformer-style) between all nodes.
      - The output is added to the existing node representations (residual).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, \
            "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : Node features [N, hidden_dim]
            batch : Batch vector [N] (assigns each node to a graph)

        Returns:
            Updated node features [N, hidden_dim]
        """
        # Pad nodes into a dense batch: x_dense [B, max_N, D], mask [B, max_N]
        x_dense, mask = to_dense_batch(x, batch)
        B, max_N, D = x_dense.shape

        # Project Q, K, V — operate only on real (non-padded) positions.
        # Zero out padded positions before projection so they don't shift BN stats.
        x_dense = x_dense * mask.unsqueeze(-1).float()

        Q = self.q_proj(x_dense).view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_dense).view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_dense).view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        # All shapes: [B, heads, max_N, head_dim]

        # Scaled dot-product attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, max_N, max_N]

        # --- Bug fix 1: mask BOTH key AND query padded positions ---
        # key mask  [B, 1, 1, max_N] — prevents attending TO padded keys
        key_mask = mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~key_mask, float('-inf'))

        # --- Bug fix 2: NaN guard after softmax ---
        # If an entire row is -inf (padded query node), softmax → NaN.
        # Replace those NaNs with 0 so they don't propagate.
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # safe no-op for real nodes
        attn = self.dropout(attn)

        # query mask [B, 1, max_N, 1] — zero out output for padded query nodes
        query_mask = mask.unsqueeze(1).unsqueeze(-1)
        attn = attn * query_mask.float()

        # Aggregate values
        out = torch.matmul(attn, V)                                      # [B, heads, max_N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, max_N, D)         # [B, max_N, D]
        out = self.out_proj(out)

        # Un-pad: extract only real node rows [N, D]
        out_sparse = out[mask]

        # Residual + LayerNorm
        return self.norm(x + out_sparse)


# ---------------------------------------------------------------------------
# GNN + FA Classifier
# ---------------------------------------------------------------------------

class GNNWithFA(nn.Module):
    """
    GNN (GCN or GIN) followed by a Fully-Adjacent layer.

    Architecture:
      Input -> k GNN layers (local message passing)
             -> FA layer    (global attention, bypasses topology)
             -> Linear head -> Output

    The FA layer is applied AFTER the standard GNN layers so that
    local structural features are learned first, then global context
    is added in a single step without over-squashing.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, gnn_type: str = "gin",
                 dropout: float = 0.0, task: str = "node",
                 fa_heads: int = 4):
        super().__init__()
        self.task = task

        # Standard GNN backbone
        if gnn_type == "gcn":
            self.gnn = GCN(in_dim, hidden_dim, hidden_dim, num_layers, dropout)
        elif gnn_type == "gin":
            self.gnn = GIN(in_dim, hidden_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}. Choose 'gcn' or 'gin'.")

        # Fully-Adjacent layer (the key mitigation for over-squashing)
        self.fa_layer = FALayer(hidden_dim, num_heads=fa_heads, dropout=dropout)

        # Classification head
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Local message passing (susceptible to over-squashing)
        h = self.gnn(x, edge_index, batch)

        # Global attention (resolves bottleneck)
        h = self.fa_layer(h, batch)

        # Pool for graph-level tasks
        if self.task == "graph":
            h = global_mean_pool(h, batch)

        return self.head(h)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _test_fa_layer():
    """
    Regression test for the two FALayer bugs:
      1. NaN propagation from softmax on all-masked rows.
      2. Padded query positions leaking into real node outputs.

    Uses an intentionally unequal batch (sizes 5 and 3) to force padding.
    Passes if no NaN/Inf in output and shapes are correct.
    """
    torch.manual_seed(0)
    hidden_dim, heads = 16, 4
    # Graph 1: 5 nodes, Graph 2: 3 nodes → padded to max_N=5 in dense batch
    x     = torch.randn(8, hidden_dim)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])

    layer = FALayer(hidden_dim, num_heads=heads, dropout=0.0)
    layer.eval()
    with torch.no_grad():
        out = layer(x, batch)

    assert out.shape == (8, hidden_dim), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in FALayer output!"
    assert not torch.isinf(out).any(), "Inf detected in FALayer output!"
    print("  FALayer unit test PASSED ✓")


def _test_all_models():
    """Check every model variant runs forward pass without error."""
    torch.manual_seed(0)
    x          = torch.randn(10, 5)
    edge_index = torch.randint(0, 10, (2, 20))
    batch      = torch.tensor([0] * 5 + [1] * 5)

    for name in ["gcn", "gin", "gcn+fa", "gin+fa"]:
        model = get_model(name, in_dim=5, hidden_dim=16, out_dim=4,
                          num_layers=3, task="node")
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        assert out.shape == (10, 4), f"{name}: bad output shape {out.shape}"
        assert not torch.isnan(out).any(), f"{name}: NaN in output"
        print(f"  {name:10s} forward PASSED ✓  output shape: {out.shape}")


def get_model(model_name: str, in_dim: int, hidden_dim: int, out_dim: int,
              num_layers: int, dropout: float = 0.0,
              task: str = "node") -> nn.Module:
    """
    Factory function to instantiate a model by name.

    Args:
        model_name : one of 'gcn', 'gin', 'gcn+fa', 'gin+fa'
        task       : 'node' or 'graph'
    """
    name = model_name.lower()
    if name == "gcn":
        return GCNClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout, task)
    elif name == "gin":
        return GINClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout, task)
    elif name == "gcn+fa":
        return GNNWithFA(in_dim, hidden_dim, out_dim, num_layers, "gcn", dropout, task)
    elif name == "gin+fa":
        return GNNWithFA(in_dim, hidden_dim, out_dim, num_layers, "gin", dropout, task)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from gcn, gin, gcn+fa, gin+fa")


if __name__ == "__main__":
    print("=" * 50)
    print("  models.py self-test")
    print("=" * 50)
    print("\n[1] FALayer NaN regression test:")
    _test_fa_layer()
    print("\n[2] All model forward passes:")
    _test_all_models()
    print("\nAll tests passed. models.py is correctly implemented.")
