"""
run_experiment.py - Main TREE-NEIGHBORS-MATCH Experiment
=========================================================
Reproduces the core experiment from Alon & Yahav (2021):

  For each tree depth r in {2, 3, 4, 5, 6, 7, 8}:
    Train GCN, GIN, GCN+FA, GIN+FA with r layers.
    Record test accuracy.

Expected result:
  - GCN and GIN accuracy drops sharply as depth increases (over-squashing)
  - GCN+FA and GIN+FA maintain high accuracy across depths

Usage:
    python run_experiment.py
    python run_experiment.py --depths 2 3 4 5 --models gcn gin gin+fa
    python run_experiment.py --depth_single 5 --verbose
"""

import sys
import os
import argparse
import json
import time
import torch
import numpy as np
import pandas as pd

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from torch_geometric.loader import DataLoader
from code.gnn_implementations.models import get_model
from code.gnn_implementations.train_utils import train_model, evaluate, track_gradient_norms
from dataset import get_datasets


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "depths":       [2, 3, 4, 5, 6, 7],  # Tree depths to test
    "models":       ["gcn", "gin", "gcn+fa", "gin+fa"],
    "hidden_dim":   64,
    "num_classes":  4,      # Number of distinct leaf labels
    "train_size":   1000,
    "val_size":     200,
    "test_size":    200,
    "batch_size":   32,
    "lr":           1e-3,
    "num_epochs":   200,
    "patience":     30,
    "dropout":      0.0,
    "seed":         42,
    "num_runs":     3,      # Repeat each experiment for variance estimation
    "output_dir":   "results",
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(model_name: str, depth: int, config: dict,
               device: torch.device, verbose: bool = False) -> dict:
    """
    Train and evaluate one model on one tree depth.

    Returns dict with test_acc, val_acc, best_epoch, train_time.
    """
    # Data
    train_ds, val_ds, test_ds, feat_dim = get_datasets(
        depth=depth,
        num_classes=config["num_classes"],
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"])

    # Model: number of GNN layers = depth (minimum needed to reach leaves)
    model = get_model(
        model_name=model_name,
        in_dim=feat_dim,
        hidden_dim=config["hidden_dim"],
        out_dim=config["num_classes"],
        num_layers=depth,
        dropout=config["dropout"],
        task="node",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {model_name} | Depth: {depth} | Params: {n_params:,}")

    # Train
    t0 = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["num_epochs"],
        patience=config["patience"],
        task="node",
        verbose=verbose,
    )
    train_time = time.time() - t0

    # Evaluate on test set
    test_acc, test_loss = evaluate(model, test_loader, device, task="node")

    return {
        "test_acc":   test_acc,
        "val_acc":    history["best_val_acc"],
        "best_epoch": history["best_epoch"],
        "train_time": train_time,
        "history":    history,
    }


# ---------------------------------------------------------------------------
# Gradient norm experiment (demonstrates over-squashing mechanistically)
# ---------------------------------------------------------------------------

def run_gradient_experiment(depth: int, config: dict, device: torch.device):
    """
    For a given depth, compare gradient norms across layers for GIN vs GIN+FA.
    Smaller gradient norms in early layers = more over-squashing.
    """
    from dataset import get_datasets
    train_ds, _, _, feat_dim = get_datasets(depth=depth, num_classes=config["num_classes"],
                                             train_size=100, seed=config["seed"])
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))

    results = {}
    for model_name in ["gin", "gin+fa"]:
        model = get_model(model_name, feat_dim, config["hidden_dim"],
                          config["num_classes"], depth, task="node").to(device)
        # One step of training to initialize gradients
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        grad_norms = track_gradient_norms(model, batch, device)
        results[model_name] = grad_norms

    return results


# ---------------------------------------------------------------------------
# Full experiment sweep
# ---------------------------------------------------------------------------

def run_full_experiment(config: dict, device: torch.device,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Sweep over all (model, depth) combinations with multiple runs.

    Returns a DataFrame with columns:
      model, depth, run, test_acc, val_acc, best_epoch, train_time
    """
    os.makedirs(config["output_dir"], exist_ok=True)
    records = []

    total = len(config["models"]) * len(config["depths"]) * config["num_runs"]
    done = 0

    for model_name in config["models"]:
        for depth in config["depths"]:
            accs = []
            for run in range(config["num_runs"]):
                # Vary seed per run for variance estimation
                run_config = {**config, "seed": config["seed"] + run * 100}

                print(f"\n[{done+1}/{total}] Model={model_name} | Depth={depth} | Run={run+1}")
                result = run_single(model_name, depth, run_config, device, verbose)

                print(f"  -> Test Acc: {result['test_acc']:.4f} | "
                      f"Best Val Acc: {result['val_acc']:.4f} | "
                      f"Best Epoch: {result['best_epoch']} | "
                      f"Time: {result['train_time']:.1f}s")

                accs.append(result["test_acc"])
                records.append({
                    "model":      model_name,
                    "depth":      depth,
                    "run":        run,
                    "test_acc":   result["test_acc"],
                    "val_acc":    result["val_acc"],
                    "best_epoch": result["best_epoch"],
                    "train_time": result["train_time"],
                })
                done += 1

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"\n  >>> {model_name} @ depth={depth}: "
                  f"Mean Acc = {mean_acc:.4f} ± {std_acc:.4f}")

    df = pd.DataFrame(records)
    out_path = os.path.join(config["output_dir"], "results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    return df


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame):
    """Print a clean mean ± std table across runs."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY: Test Accuracy (Mean ± Std over runs)")
    print("=" * 70)

    summary = df.groupby(["model", "depth"])["test_acc"].agg(["mean", "std"]).reset_index()
    pivot = summary.pivot(index="depth", columns="model", values=["mean", "std"])

    models = df["model"].unique()
    depths = sorted(df["depth"].unique())

    # Header
    header = f"{'Depth':>6} | " + " | ".join(f"{m:>12}" for m in models)
    print(header)
    print("-" * len(header))

    for depth in depths:
        row = f"{depth:>6} | "
        parts = []
        for model in models:
            sub = df[(df["model"] == model) & (df["depth"] == depth)]["test_acc"]
            if len(sub) > 0:
                parts.append(f"{sub.mean():.3f}±{sub.std():.3f}")
            else:
                parts.append("     N/A    ")
        row += " | ".join(f"{p:>12}" for p in parts)
        print(row)

    print("=" * 70)
    print("Higher is better. Over-squashing causes GCN/GIN to degrade with depth.")
    print("GCN+FA and GIN+FA should maintain accuracy due to the FA layer bypass.\n")


# ---------------------------------------------------------------------------
# Argument parser + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TREE-NEIGHBORS-MATCH: empirical over-squashing benchmark"
    )
    parser.add_argument("--depths", nargs="+", type=int,
                        default=DEFAULT_CONFIG["depths"],
                        help="Tree depths to sweep over")
    parser.add_argument("--models", nargs="+", type=str,
                        default=DEFAULT_CONFIG["models"],
                        choices=["gcn", "gin", "gcn+fa", "gin+fa"],
                        help="Models to evaluate")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--num_runs", type=int, default=DEFAULT_CONFIG["num_runs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Print epoch-level logs")
    parser.add_argument("--depth_single", type=int, default=None,
                        help="Run a single depth only (quick test)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    config = {**DEFAULT_CONFIG}
    config.update({
        "depths":     [args.depth_single] if args.depth_single else args.depths,
        "models":     args.models,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "num_runs":   args.num_runs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "seed":       args.seed,
        "output_dir": args.output_dir,
    })

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print("\n" + "=" * 50)
    print("  TREE-NEIGHBORS-MATCH Experiment")
    print("  Over-Squashing in GNNs (Alon & Yahav 2021)")
    print("=" * 50)
    print(f"  Depths  : {config['depths']}")
    print(f"  Models  : {config['models']}")
    print(f"  Runs    : {config['num_runs']}")
    print(f"  Epochs  : {config['num_epochs']}")
    print("=" * 50)

    df = run_full_experiment(config, device, verbose=args.verbose)
    print_summary_table(df)

    # Save config
    config_path = os.path.join(config["output_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump({k: v for k, v in config.items() if isinstance(v, (int, float, str, list))}, f, indent=2)
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
