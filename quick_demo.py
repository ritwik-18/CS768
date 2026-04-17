"""
quick_demo.py - Fast Demo (no GPU required, ~2 minutes)
=========================================================
Runs a minimal version of the experiment to verify everything works.
Tests depths 2, 3, 4 with 1 run each and fewer epochs.

Usage:
    python quick_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments', 'tree_neighbors_match'))

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from code.gnn_implementations.models import get_model
from code.gnn_implementations.train_utils import train_model, evaluate
from experiments.tree_neighbors_match.dataset import get_datasets

DEMO_CONFIG = {
    "depths":      [2, 3, 4],
    "models":      ["gin", "gin+fa"],
    "hidden_dim":  32,
    "num_classes": 4,
    "train_size":  300,
    "val_size":    100,
    "test_size":   100,
    "batch_size":  32,
    "lr":          1e-3,
    "num_epochs":  50,
    "patience":    15,
    "dropout":     0.0,
    "seed":        42,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
print("=" * 55)
print("  Quick Demo: Over-Squashing in GNNs")
print("  GIN vs GIN+FA on TREE-NEIGHBORS-MATCH")
print("=" * 55)

results = {}
for model_name in DEMO_CONFIG["models"]:
    results[model_name] = {}
    for depth in DEMO_CONFIG["depths"]:
        train_ds, val_ds, test_ds, feat_dim = get_datasets(
            depth=depth,
            num_classes=DEMO_CONFIG["num_classes"],
            train_size=DEMO_CONFIG["train_size"],
            val_size=DEMO_CONFIG["val_size"],
            test_size=DEMO_CONFIG["test_size"],
            seed=DEMO_CONFIG["seed"],
        )
        train_loader = DataLoader(train_ds, batch_size=DEMO_CONFIG["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=DEMO_CONFIG["batch_size"])
        test_loader  = DataLoader(test_ds,  batch_size=DEMO_CONFIG["batch_size"])

        model = get_model(
            model_name=model_name,
            in_dim=feat_dim,
            hidden_dim=DEMO_CONFIG["hidden_dim"],
            out_dim=DEMO_CONFIG["num_classes"],
            num_layers=depth,
            task="node",
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=DEMO_CONFIG["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        print(f"\nTraining {model_name.upper()} | Depth={depth} | "
              f"Nodes per tree: {2**(depth+1)-1}")

        history = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, device,
            num_epochs=DEMO_CONFIG["num_epochs"],
            patience=DEMO_CONFIG["patience"],
            verbose=False,
        )

        test_acc, _ = evaluate(model, test_loader, device, task="node")
        results[model_name][depth] = test_acc
        print(f"  Test Accuracy: {test_acc:.4f} "
              f"(Best val: {history['best_val_acc']:.4f} @ epoch {history['best_epoch']})")

# Summary
print("\n" + "=" * 55)
print("SUMMARY: Test Accuracy")
print(f"{'Depth':>8} | {'GIN':>10} | {'GIN+FA':>10} | {'FA Boost':>10}")
print("-" * 50)
for depth in DEMO_CONFIG["depths"]:
    gin_acc    = results.get("gin",    {}).get(depth, float('nan'))
    ginfa_acc  = results.get("gin+fa", {}).get(depth, float('nan'))
    boost      = ginfa_acc - gin_acc
    print(f"{depth:>8} | {gin_acc:>10.4f} | {ginfa_acc:>10.4f} | {boost:>+10.4f}")

print("=" * 55)
print("\nRun the full experiment:")
print("  cd experiments/tree_neighbors_match")
print("  python run_experiment.py --depths 2 3 4 5 6 7\n")
print("Generate plots:")
print("  python visualize.py --plot all")
