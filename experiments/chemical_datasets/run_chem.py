"""
run_chem.py - Chemical Dataset Experiment (ZINC Benchmark)
============================================================
Demonstrates the over-squashing bottleneck on real-world molecular graphs.

Experiment:
  Train GCN and GCN+FA on the ZINC dataset (molecular property prediction).
  ZINC is a standard graph regression benchmark where nodes are atoms and
  edges are chemical bonds.  The task is to predict the penalized logP
  score of each molecule — a scalar regression target.

Why ZINC?
  Molecules are graphs where long-range dependencies matter.
  For example, the chemical activity of a functional group can depend on
  atoms several bonds away.  Standard GCNs suffer from over-squashing when
  this long-range information must be propagated through many hops.
  The FA layer bypasses the bottleneck by letting every atom attend globally.

Expected result:
  GCN+FA achieves lower MAE than plain GCN, confirming that the FA layer
  alleviates over-squashing on real chemical data — not just synthetic trees.

Usage:
    python run_chem.py
    python run_chem.py --models gcn gcn+fa --num_layers 4 --epochs 100
    python run_chem.py --quick          # fast sanity-check run
    python run_chem.py --output_dir my_results
"""

import sys
import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.gnn_implementations.models import GCN, GNNWithFA


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "models":       ["gcn", "gcn+fa"],
    "num_layers":   [4],            # GNN depths to evaluate
    "hidden_dim":   64,
    "atom_types":   21,             # ZINC has 21 distinct atom-type integers
    "batch_size":   128,
    "lr":           1e-3,
    "num_epochs":   200,
    "patience":     30,
    "dropout":      0.0,
    "seed":         42,
    "num_runs":     3,
    "output_dir":   "results_chem",
    "data_root":    "./data",       # where PyG will cache the ZINC dataset
}


# ---------------------------------------------------------------------------
# Regression wrappers
# (The backbone models output logits; we add a scalar head for regression.)
# ---------------------------------------------------------------------------

class GCNRegressor(nn.Module):
    """
    GCN for graph-level regression.

    Architecture:
      Atom embedding -> k GCNConv layers -> global mean pool -> Linear(1)

    An embedding layer converts integer atom-type indices into dense
    vectors before they enter the GCN, matching standard practice on
    ZINC (where x is a [N, 1] integer tensor).
    """

    def __init__(self, atom_types: int, hidden_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(atom_types, hidden_dim)
        # GCN backbone: in_dim == hidden_dim because embedding maps there
        self.gnn = GCN(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        # x is [N, 1] integer tensor in ZINC — squeeze and embed
        h = self.embedding(x.squeeze(-1))           # [N, hidden_dim]
        h = self.gnn(h, edge_index, batch)
        h = global_mean_pool(h, batch)              # [B, hidden_dim]
        return self.head(h).squeeze(-1)             # [B]


class GCNWithFARegressor(nn.Module):
    """
    GCN+FA for graph-level regression.

    Same as GCNRegressor but with a Fully-Adjacent layer appended after
    the GCN stack.  The FA layer lets every atom attend to every other
    atom in its molecule, bypassing the topological bottleneck.
    """

    def __init__(self, atom_types: int, hidden_dim: int,
                 num_layers: int, dropout: float = 0.0, fa_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(atom_types, hidden_dim)
        self.gnn_fa = GNNWithFA(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,     # keep hidden_dim; head is below
            num_layers=num_layers,
            gnn_type="gcn",
            dropout=dropout,
            task="graph",           # enables global_mean_pool inside GNNWithFA
            fa_heads=fa_heads,
        )
        # GNNWithFA already applies global_mean_pool and a Linear head
        # when task="graph". We want to intercept BEFORE that head so we
        # can apply our own two-layer regression head.
        # Workaround: set task="node" and pool manually here.
        self.gnn_fa = _GCNFABackbone(hidden_dim, num_layers, dropout, fa_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        h = self.embedding(x.squeeze(-1))
        h = self.gnn_fa(h, edge_index, batch)      # [N, hidden_dim]
        h = global_mean_pool(h, batch)              # [B, hidden_dim]
        return self.head(h).squeeze(-1)             # [B]


class _GCNFABackbone(nn.Module):
    """
    GCN followed by an FA layer, returning node-level features.
    Used internally by GCNWithFARegressor so we can pool with our own head.
    """

    def __init__(self, hidden_dim: int, num_layers: int,
                 dropout: float, fa_heads: int):
        super().__init__()
        from src.gnn_implementations.models import GCN, FALayer
        self.gnn = GCN(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.fa  = FALayer(hidden_dim, num_heads=fa_heads, dropout=dropout)

    def forward(self, x, edge_index, batch):
        h = self.gnn(x, edge_index, batch)
        h = self.fa(h, batch)
        return h


def build_model(model_name: str, config: dict) -> nn.Module:
    """Instantiate the correct model for the given name."""
    name = model_name.lower()
    if name == "gcn":
        return GCNRegressor(
            atom_types=config["atom_types"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers_current"],
            dropout=config["dropout"],
        )
    elif name == "gcn+fa":
        return GCNWithFARegressor(
            atom_types=config["atom_types"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers_current"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose: gcn, gcn+fa")


# ---------------------------------------------------------------------------
# Training / evaluation loops (regression — MAE metric)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device) -> float:
    """One training pass, returns mean MAE over the epoch."""
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.l1_loss(pred, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Returns mean absolute error (MAE) on the given loader."""
    model.eval()
    total_mae, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        mae = F.l1_loss(pred, batch.y.float(), reduction="sum")
        total_mae += mae.item()
        n += batch.num_graphs
    return total_mae / n


def train_model(model, train_loader, val_loader, config, device,
                verbose=False) -> dict:
    """
    Full training loop with early stopping.
    Returns history dict with best_val_mae and train curve.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5,
    )

    best_val_mae = float("inf")
    best_state   = None
    patience_ctr = 0
    history      = {"train_mae": [], "val_mae": [], "best_val_mae": float("inf")}

    for epoch in range(1, config["num_epochs"] + 1):
        train_mae = train_epoch(model, train_loader, optimizer, device)
        val_mae   = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            history["best_val_mae"] = best_val_mae
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | Train MAE: {train_mae:.4f} | "
                  f"Val MAE: {val_mae:.4f} | Best: {best_val_mae:.4f}")

        if patience_ctr >= config["patience"]:
            if verbose:
                print(f"    Early stopping at epoch {epoch}.")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_zinc(data_root: str, batch_size: int):
    """
    Download (or load from cache) the ZINC dataset via PyTorch Geometric.

    ZINC subset=True gives the standard 12k/1k/1k train/val/test split
    used in the benchmarking literature (Dwivedi et al., 2020).

    Returns (train_loader, val_loader, test_loader).
    """
    print(f"  Loading ZINC dataset (cache: {data_root}) ...")
    train_ds = ZINC(root=data_root, subset=True, split="train")
    val_ds   = ZINC(root=data_root, subset=True, split="val")
    test_ds  = ZINC(root=data_root, subset=True, split="test")

    print(f"  Dataset sizes — Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Sample graph: {train_ds[0]}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(model_name: str, num_layers: int, config: dict,
               train_loader, val_loader, test_loader,
               device: torch.device, verbose: bool = False) -> dict:
    """Train + evaluate one (model, depth) configuration."""

    run_cfg = {**config, "num_layers_current": num_layers}

    torch.manual_seed(run_cfg["seed"])
    model = build_model(model_name, run_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"    Params: {n_params:,}")

    t0 = time.time()
    history = train_model(model, train_loader, val_loader, run_cfg, device, verbose)
    train_time = time.time() - t0

    test_mae = evaluate(model, test_loader, device)

    return {
        "test_mae":    test_mae,
        "best_val_mae": history["best_val_mae"],
        "train_time":  train_time,
        "n_params":    n_params,
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_full_experiment(config: dict, device: torch.device,
                        verbose: bool = False) -> pd.DataFrame:
    """
    Sweep over all (model, num_layers) combinations for `num_runs` seeds.
    Returns a tidy DataFrame.
    """
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load data once (shared across all runs)
    train_loader, val_loader, test_loader = load_zinc(
        config["data_root"], config["batch_size"]
    )

    records = []
    total = len(config["models"]) * len(config["num_layers"]) * config["num_runs"]
    done  = 0

    for model_name in config["models"]:
        for num_layers in config["num_layers"]:
            maes = []
            for run in range(config["num_runs"]):
                run_config = {**config, "seed": config["seed"] + run * 100}
                done += 1

                print(f"\n[{done}/{total}] Model={model_name} | "
                      f"Layers={num_layers} | Run={run + 1}")

                result = run_single(
                    model_name, num_layers, run_config,
                    train_loader, val_loader, test_loader,
                    device, verbose,
                )

                print(f"  -> Test MAE: {result['test_mae']:.4f} | "
                      f"Best Val MAE: {result['best_val_mae']:.4f} | "
                      f"Time: {result['train_time']:.1f}s")

                maes.append(result["test_mae"])
                records.append({
                    "model":        model_name,
                    "num_layers":   num_layers,
                    "run":          run,
                    "test_mae":     result["test_mae"],
                    "best_val_mae": result["best_val_mae"],
                    "train_time":   result["train_time"],
                    "n_params":     result["n_params"],
                })

            print(f"\n  >>> {model_name} | layers={num_layers}: "
                  f"MAE = {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    df = pd.DataFrame(records)
    out_path = os.path.join(config["output_dir"], "chem_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame):
    """Print a clean mean ± std MAE table (lower is better)."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY: Test MAE on ZINC (Mean ± Std, lower is better)")
    print("=" * 70)

    models     = df["model"].unique()
    layer_vals = sorted(df["num_layers"].unique())

    header = f"{'Layers':>8} | " + " | ".join(f"{m:>14}" for m in models)
    print(header)
    print("-" * len(header))

    for nl in layer_vals:
        row_parts = []
        for model in models:
            sub = df[(df["model"] == model) & (df["num_layers"] == nl)]["test_mae"]
            if len(sub) > 0:
                row_parts.append(f"{sub.mean():.4f}±{sub.std():.4f}")
            else:
                row_parts.append("      N/A     ")
        row = f"{nl:>8} | " + " | ".join(f"{p:>14}" for p in row_parts)
        print(row)

    print("=" * 70)
    improvement = _compute_improvement(df)
    if improvement is not None:
        print(f"FA layer improvement: {improvement:+.4f} MAE "
              f"({'better' if improvement < 0 else 'worse'} than plain GCN)")
    print("\nInterpretation:")
    print("  Lower MAE is better.  GCN+FA should outperform plain GCN because")
    print("  the FA layer lets each atom attend globally, bypassing the")
    print("  over-squashing bottleneck that limits standard message passing.\n")


def _compute_improvement(df: pd.DataFrame):
    """Return (gcn+fa MAE) - (gcn MAE). Negative = FA is better."""
    gcn    = df[df["model"] == "gcn"]["test_mae"].mean()
    gcn_fa = df[df["model"] == "gcn+fa"]["test_mae"].mean()
    if np.isnan(gcn) or np.isnan(gcn_fa):
        return None
    return gcn_fa - gcn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chemical dataset experiment: GCN vs GCN+FA on ZINC"
    )
    parser.add_argument("--models", nargs="+", type=str,
                        default=DEFAULT_CONFIG["models"],
                        choices=["gcn", "gcn+fa"],
                        help="Models to compare")
    parser.add_argument("--num_layers", nargs="+", type=int,
                        default=DEFAULT_CONFIG["num_layers"],
                        help="Number of GNN layers (depths) to sweep")
    parser.add_argument("--hidden_dim", type=int,
                        default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--epochs", type=int,
                        default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--num_runs", type=int,
                        default=DEFAULT_CONFIG["num_runs"])
    parser.add_argument("--batch_size", type=int,
                        default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--dropout", type=float,
                        default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--output_dir", type=str,
                        default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--data_root", type=str,
                        default=DEFAULT_CONFIG["data_root"],
                        help="Directory where ZINC will be downloaded/cached")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Print epoch-level training logs")
    parser.add_argument("--quick", action="store_true",
                        help="Fast sanity-check: 1 layer, 1 run, 20 epochs")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    config = {**DEFAULT_CONFIG}
    config.update({
        "models":     args.models,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.epochs,
        "num_runs":   args.num_runs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "dropout":    args.dropout,
        "seed":       args.seed,
        "output_dir": args.output_dir,
        "data_root":  args.data_root,
    })

    # Quick-run override for sanity checks
    if args.quick:
        print("  [quick mode] Overriding to 1 layer, 1 run, 20 epochs.")
        config.update({"num_layers": [1], "num_runs": 1, "num_epochs": 20})

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print("\n" + "=" * 55)
    print("  Chemical Dataset Experiment: ZINC Regression")
    print("  GCN vs GCN+FA (Alon & Yahav, 2021)")
    print("=" * 55)
    print(f"  Models     : {config['models']}")
    print(f"  Layers     : {config['num_layers']}")
    print(f"  Hidden dim : {config['hidden_dim']}")
    print(f"  Runs       : {config['num_runs']}")
    print(f"  Epochs     : {config['num_epochs']}")
    print("=" * 55)

    df = run_full_experiment(config, device, verbose=args.verbose)
    print_summary_table(df)

    # Save config alongside results
    cfg_path = os.path.join(config["output_dir"], "chem_config.json")
    with open(cfg_path, "w") as f:
        json.dump(
            {k: v for k, v in config.items() if isinstance(v, (int, float, str, list))},
            f, indent=2,
        )
    print(f"Config saved to: {cfg_path}")


if __name__ == "__main__":
    main()