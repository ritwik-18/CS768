"""
dataset.py - TREE-NEIGHBORS-MATCH Synthetic Dataset
=====================================================
Implements the synthetic benchmark from Alon & Yahav (2021).

Task Description:
-----------------
Given a complete binary tree of depth r:
  - Each LEAF node is assigned a random integer label in {0, 1, ..., num_classes-1}.
  - One LEAF is designated as the "selected" leaf.
  - The ROOT node must predict the label of the selected leaf.

Why this is hard for GNNs:
  - The root is r hops away from any leaf.
  - A GNN needs AT LEAST r layers to propagate the label to the root.
  - The receptive field at depth r has 2^r leaf nodes.
  - All 2^r labels must be squashed into a fixed-size root embedding
    together with the "selection" signal — this is over-squashing.

Node Features:
  - Internal nodes: [0, 0, ..., 0] (zero vector of size num_classes+1)
  - Leaf nodes: one-hot encoding of their label (size num_classes)
                + 1 bit indicating if this leaf is "selected"
  - Total feature size: num_classes + 1

Label:
  - Single integer: the label of the selected leaf (for root node)
  - For non-root nodes: -1 (ignored in loss)
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from typing import List, Optional


def build_binary_tree_graph(depth: int) -> nx.DiGraph:
    """
    Build a balanced binary tree as a directed graph (edges go both ways).
    Uses NetworkX's balanced_binary_tree with branching factor 2.

    Returns:
        G      : NetworkX DiGraph
        root   : index of root node (always 0)
        leaves : list of leaf node indices
    """
    G_undirected = nx.balanced_binary_tree(2, depth)
    G = G_undirected.to_directed()
    root = 0
    leaves = [n for n in G.nodes() if G_undirected.degree(n) == 1 and n != root]
    # For depth=1, root has no parent — handle edge case
    if depth == 1:
        leaves = list(G_undirected.neighbors(root))
    return G, root, leaves


def create_single_instance(depth: int, num_classes: int,
                            rng: np.random.Generator) -> Data:
    """
    Create one TREE-NEIGHBORS-MATCH graph instance.

    Args:
        depth       : depth of the binary tree (= number of GNN layers needed)
        num_classes : number of distinct leaf labels
        rng         : numpy random generator for reproducibility

    Returns:
        PyG Data object with:
          x          : node features [N, num_classes+1]
          edge_index : graph edges [2, E]
          y          : target label (label of selected leaf) — scalar
          root_mask  : boolean mask identifying the root node
          depth      : tree depth (for logging)
    """
    G, root, leaves = build_binary_tree_graph(depth)
    num_nodes = G.number_of_nodes()
    feat_dim = num_classes + 1  # num_classes for label one-hot + 1 for selection bit

    # Initialize all node features to zero
    x = torch.zeros(num_nodes, feat_dim, dtype=torch.float)

    # Assign random labels to leaves
    leaf_labels = rng.integers(0, num_classes, size=len(leaves))

    # Select one leaf as the target
    selected_idx = rng.integers(0, len(leaves))
    selected_leaf = leaves[selected_idx]
    target_label = int(leaf_labels[selected_idx])

    # Encode leaf features
    for i, leaf in enumerate(leaves):
        label = int(leaf_labels[i])
        x[leaf, label] = 1.0          # one-hot label
        if leaf == selected_leaf:
            x[leaf, num_classes] = 1.0  # selection bit

    # Build edge_index from NetworkX
    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)

    # Root mask: only root node has a supervised label
    root_mask = torch.zeros(num_nodes, dtype=torch.bool)
    root_mask[root] = True

    # Node-level labels: target for root, -1 for all others (ignored)
    y_nodes = torch.full((num_nodes,), -1, dtype=torch.long)
    y_nodes[root] = target_label

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_nodes,
        root_mask=root_mask,
        num_nodes=num_nodes,
    )
    data.depth = depth
    data.target = target_label
    return data


class TreeNeighborsMatchDataset(Dataset):
    """
    Dataset of TREE-NEIGHBORS-MATCH instances for a fixed tree depth.

    Args:
        depth       : depth of binary tree (controls difficulty)
        num_samples : number of graph instances to generate
        num_classes : number of distinct leaf labels
        seed        : random seed for reproducibility
        split       : 'train', 'val', or 'test'
    """

    def __init__(self, depth: int, num_samples: int = 1000,
                 num_classes: int = 4, seed: int = 42, split: str = 'train'):
        super().__init__()
        self.depth = depth
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.feat_dim = num_classes + 1

        # Deterministic splits via different seeds
        split_offset = {"train": 0, "val": 10000, "test": 20000}
        rng_seed = seed + split_offset.get(split, 0)
        rng = np.random.default_rng(rng_seed)

        self.data_list: List[Data] = [
            create_single_instance(depth, num_classes, rng)
            for _ in range(num_samples)
        ]

    def len(self) -> int:
        return self.num_samples

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def get_datasets(depth: int, num_classes: int = 4,
                 train_size: int = 1000, val_size: int = 200,
                 test_size: int = 200, seed: int = 42):
    """
    Convenience function to get train/val/test splits.

    Returns:
        train_dataset, val_dataset, test_dataset
        feat_dim (int): input feature dimension
    """
    train_ds = TreeNeighborsMatchDataset(depth, train_size, num_classes, seed, 'train')
    val_ds = TreeNeighborsMatchDataset(depth, val_size, num_classes, seed, 'val')
    test_ds = TreeNeighborsMatchDataset(depth, test_size, num_classes, seed, 'test')
    feat_dim = num_classes + 1
    return train_ds, val_ds, test_ds, feat_dim


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TREE-NEIGHBORS-MATCH Dataset Sanity Check ===\n")
    for depth in [2, 3, 4, 5]:
        rng = np.random.default_rng(0)
        sample = create_single_instance(depth, num_classes=4, rng=rng)
        n_leaves = 2 ** depth
        print(f"Depth={depth}:")
        print(f"  Nodes      : {sample.num_nodes}  (tree has 2^{depth+1}-1 = {2**(depth+1)-1} nodes)")
        print(f"  Leaves     : {n_leaves}  (receptive field at root)")
        print(f"  Features   : {sample.x.shape}")
        print(f"  Edges      : {sample.edge_index.shape[1]}")
        print(f"  Target     : class {sample.target}")
        print(f"  Root label : {sample.y[0].item()}")
        print()
