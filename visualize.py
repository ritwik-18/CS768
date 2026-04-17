"""
visualize.py - Plot Experiment Results
=======================================
Generates figures:
  1. Test accuracy vs. tree depth (over-squashing bottleneck curve)
  2. Gradient norm by layer (shows vanishing gradients = over-squashing)
  3. Training curves for a selected model

Usage:
    python visualize.py                          # reads results/results.csv
    python visualize.py --results_path my/path/results.csv
    python visualize.py --plot gradient --depth 5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works on servers)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Color/style config
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "gcn":    "#e74c3c",
    "gin":    "#e67e22",
    "gcn+fa": "#2ecc71",
    "gin+fa": "#3498db",
}
MODEL_MARKERS = {
    "gcn":    "o",
    "gin":    "s",
    "gcn+fa": "^",
    "gin+fa": "D",
}
MODEL_LABELS = {
    "gcn":    "GCN (baseline)",
    "gin":    "GIN (baseline)",
    "gcn+fa": "GCN + FA layer",
    "gin+fa": "GIN + FA layer",
}


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs Depth
# ---------------------------------------------------------------------------

def plot_accuracy_vs_depth(df: pd.DataFrame, output_dir: str = "results"):
    """
    The key experiment figure. Shows how accuracy drops with depth
    for standard GNNs (over-squashing) and stays stable for +FA variants.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    models = df["model"].unique()
    depths = sorted(df["depth"].unique())

    for model in models:
        sub = df[df["model"] == model]
        means = []
        stds = []
        for d in depths:
            vals = sub[sub["depth"] == d]["test_acc"]
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            stds.append(vals.std()  if len(vals) > 1 else 0.0)

        means = np.array(means)
        stds  = np.array(stds)
        color  = MODEL_COLORS.get(model, "gray")
        marker = MODEL_MARKERS.get(model, "o")
        label  = MODEL_LABELS.get(model, model)
        ls     = "--" if "fa" not in model else "-"

        ax.plot(depths, means, color=color, marker=marker,
                linestyle=ls, linewidth=2.2, markersize=8, label=label)
        ax.fill_between(depths, means - stds, means + stds,
                        color=color, alpha=0.15)

    # Random-chance baseline
    num_classes = 4  # default
    ax.axhline(y=1/num_classes, color="gray", linestyle=":", linewidth=1.5,
               label=f"Random chance (1/{num_classes})")

    ax.set_xlabel("Tree Depth (r)  ·  GNN Layers = r", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title("Over-Squashing: Test Accuracy vs Tree Depth\n"
                 "FA layer preserves accuracy; baseline GNNs collapse", fontsize=13)
    ax.set_xticks(depths)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "accuracy_vs_depth.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Gradient norms by layer
# ---------------------------------------------------------------------------

def plot_gradient_norms(grad_results: dict, depth: int, output_dir: str = "results"):
    """
    Plot gradient norm per GNN layer for GIN vs GIN+FA.
    Expected: GIN gradients vanish for early layers (over-squashing).
              GIN+FA maintains gradients via the FA layer bypass.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (model_name, grad_norms) in zip(axes, grad_results.items()):
        # Group by layer index (conv layers)
        layer_norms = {}
        for name, norm in grad_norms.items():
            # Extract layer number from parameter name
            if "convs." in name:
                try:
                    layer_idx = int(name.split("convs.")[1].split(".")[0])
                    layer_norms[f"Conv Layer {layer_idx+1}"] = \
                        layer_norms.get(f"Conv Layer {layer_idx+1}", []) + [norm]
                except Exception:
                    pass
            elif "fa_layer" in name:
                layer_norms["FA Layer"] = \
                    layer_norms.get("FA Layer", []) + [norm]

        names  = list(layer_norms.keys())
        values = [np.mean(v) for v in layer_norms.values()]

        color = MODEL_COLORS.get(model_name, "steelblue")
        bars  = ax.bar(names, values, color=color, alpha=0.85, edgecolor="black")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-5,
                    f"{val:.2e}", ha="center", va="bottom", fontsize=8, rotation=30)

        ax.set_yscale("log")
        ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)}\n"
                     f"(Depth={depth})", fontsize=12)
        ax.set_ylabel("Gradient L2 Norm (log scale)", fontsize=10)
        ax.set_xlabel("Parameter Group", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Gradient Norms: GIN vs GIN+FA\n"
        "Vanishing gradients in early GIN layers → over-squashing",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"gradient_norms_depth{depth}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(histories: dict, depth: int, output_dir: str = "results"):
    """
    Plot training loss and validation accuracy over epochs for each model.

    Args:
        histories : dict mapping model_name -> history dict (from train_model)
        depth     : tree depth (for title)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, history in histories.items():
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        ls    = "--" if "fa" not in model_name else "-"

        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], color=color,
                 linestyle=ls, linewidth=1.8, label=label)
        ax2.plot(epochs, history["val_acc"], color=color,
                 linestyle=ls, linewidth=1.8, label=label)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title(f"Training Loss (Depth={depth})", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    ax2.set_title(f"Validation Accuracy (Depth={depth})", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"training_curves_depth{depth}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Receptive field growth (theoretical)
# ---------------------------------------------------------------------------

def plot_receptive_field_growth(max_depth: int = 10, output_dir: str = "results"):
    """
    Theoretical plot showing exponential growth of receptive field
    and the resulting bottleneck (Eq. 2 from the report).
    """
    depths = np.arange(1, max_depth + 1)
    receptive_field = 2 ** depths  # binary tree: 2^r leaves

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale
    ax1.bar(depths, receptive_field, color="#3498db", alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Tree Depth r", fontsize=12)
    ax1.set_ylabel("Nodes in Receptive Field |Nᵥᴷ|", fontsize=12)
    ax1.set_title("Receptive Field Growth (Linear Scale)\n"
                  "|Nᵥᴷ| = O(exp(r))", fontsize=12)
    ax1.set_xticks(depths)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Log scale showing linear growth in log = exponential in original
    ax2.plot(depths, receptive_field, color="#e74c3c", marker="o",
             linewidth=2.2, markersize=8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Tree Depth r", fontsize=12)
    ax2.set_ylabel("|Nᵥᴷ| (log scale)", fontsize=12)
    ax2.set_title("Receptive Field Growth (Log Scale)\n"
                  "Exponential growth → fixed d cannot hold all info", fontsize=12)
    ax2.set_xticks(depths)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Annotate bottleneck
    ax2.annotate("Over-squashing:\nexponential info\ninto fixed-size vector",
                 xy=(max_depth, 2 ** max_depth),
                 xytext=(max_depth - 4, 2 ** (max_depth - 2)),
                 fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="black"),
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "receptive_field_growth.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/results.csv")
    parser.add_argument("--output_dir",   type=str, default="results")
    parser.add_argument("--plot",         type=str, default="all",
                        choices=["all", "accuracy", "receptive_field", "gradient"])
    parser.add_argument("--depth",        type=int, default=5,
                        help="Depth for gradient norm plot")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot in ("all", "receptive_field"):
        print("Plotting receptive field growth (theoretical)...")
        plot_receptive_field_growth(max_depth=10, output_dir=args.output_dir)

    if args.plot in ("all", "accuracy"):
        if not os.path.exists(args.results_path):
            print(f"Results file not found: {args.results_path}")
            print("Run run_experiment.py first to generate results.")
        else:
            print("Plotting accuracy vs depth...")
            df = pd.read_csv(args.results_path)
            plot_accuracy_vs_depth(df, output_dir=args.output_dir)

    if args.plot in ("all", "gradient"):
        print(f"Running gradient norm experiment at depth={args.depth}...")
        try:
            import torch
            from run_experiment import run_gradient_experiment, DEFAULT_CONFIG
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grad_results = run_gradient_experiment(args.depth, DEFAULT_CONFIG, device)
            plot_gradient_norms(grad_results, args.depth, output_dir=args.output_dir)
        except ImportError as e:
            print(f"Could not run gradient experiment: {e}")

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
