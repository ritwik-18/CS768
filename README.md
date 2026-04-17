# CS768 – On the Bottleneck of GNNs and its Practical Implications

**Authors:** Vijaya Raghavendra S (23B1042) · Ritwik Bavurupudi (23B0954) · Tharun Tej Banoth (23B0918)

---

## Project Overview

This repository empirically investigates the **over-squashing** bottleneck in Graph Neural Networks, reproducing and extending the experiments from:

> Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR 2021.

---

## Repository Structure

```
CS768/
├── requirements.txt
├── quick_demo.py                          ← Start here! (~2 min run)
│
├── code/
│   └── gnn_implementations/
│       ├── models.py                      ← GCN, GIN, GCN+FA, GIN+FA
│       └── train_utils.py                 ← Training loop, eval, gradient tracking
│
└── experiments/
    └── tree_neighbors_match/
        ├── dataset.py                     ← Synthetic dataset generator
        ├── run_experiment.py              ← Full experiment sweep
        └── visualize.py                   ← Plot results
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 2. Install PyTorch (pick one based on your system)
```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric
```bash
pip install torch-geometric
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Code

All commands should be run from the **CS768/** root directory.

---

### Quick Demo (recommended first run, ~2 minutes on CPU)

Tests depths 2–4 with GIN vs GIN+FA to verify everything works:

```bash
python quick_demo.py
```

Expected output:
```
SUMMARY: Test Accuracy
   Depth |        GIN |     GIN+FA |   FA Boost
--------------------------------------------------
       2 |     0.9800 |     0.9900 |    +0.0100
       3 |     0.7500 |     0.9700 |    +0.2200
       4 |     0.2800 |     0.9400 |    +0.6600
```

---

### Full Experiment Sweep

Reproduces the main result from the paper (depths 2–7, all 4 models):

```bash
cd experiments/tree_neighbors_match
python run_experiment.py
```

With custom settings:
```bash
# Specific depths and models
python run_experiment.py --depths 2 3 4 5 --models gcn gin gin+fa

# Verbose (see per-epoch logs)
python run_experiment.py --verbose

# Quick single-depth test
python run_experiment.py --depth_single 4 --num_runs 1 --num_epochs 100

# GPU run (auto-detected, or force CPU)
python run_experiment.py --no_cuda
```

Results are saved to `experiments/tree_neighbors_match/results/results.csv`.

---

### Visualize Results

```bash
cd experiments/tree_neighbors_match

# All plots
python visualize.py --plot all

# Only the main accuracy curve (needs results.csv first)
python visualize.py --plot accuracy

# Theoretical receptive field growth (no experiment needed)
python visualize.py --plot receptive_field

# Gradient norm comparison (runs a mini experiment)
python visualize.py --plot gradient --depth 5
```

Plots are saved to `experiments/tree_neighbors_match/results/`.

---

## What the Code Does

### Models (`code/gnn_implementations/models.py`)

| Model    | Description |
|----------|-------------|
| `gcn`    | Graph Convolutional Network — mean aggregation, prone to over-squashing |
| `gin`    | Graph Isomorphism Network — sum aggregation, maximum 1-WL expressivity, but **most** susceptible to over-squashing (Alon & Yahav 2021) |
| `gcn+fa` | GCN + Fully-Adjacent final layer — global attention bypasses topology |
| `gin+fa` | GIN + Fully-Adjacent final layer — best of both worlds |

### Experiment (`experiments/tree_neighbors_match/`)

**TREE-NEIGHBORS-MATCH** benchmark:
- Binary tree of depth `r`
- Each leaf gets a random label ∈ {0, 1, 2, 3}
- One leaf is "selected" (encoded via a special bit)
- Root must predict the selected leaf's label
- Requires exactly `r` message-passing steps
- As `r` grows, 2ʳ leaf values must pass through fixed-size bottleneck nodes

**Key expected results:**
- GCN/GIN accuracy → ~25% (random) as depth increases (**over-squashing**)
- GCN+FA/GIN+FA maintain high accuracy across all depths (**bottleneck resolved**)

### Gradient Norm Plot

Mechanistic evidence of over-squashing:
- In plain GIN at large depth, gradients in early layers → 0
- Information from leaves never reaches the root (gradient doesn't flow back either)
- In GIN+FA, the FA layer provides a direct gradient path, preventing this

---

## Key Files Explained

### `models.py` — FALayer class
The `FALayer` implements multi-head self-attention over **all nodes in a graph**:
```
Q, K, V projections → Scaled dot-product attention → Residual + LayerNorm
```
This is analogous to the Transformer attention (Vaswani et al., 2017) applied
globally on the graph, completely bypassing graph topology for one step.

### `dataset.py` — TreeNeighborsMatchDataset
Generates synthetic binary trees using NetworkX and converts them to
PyTorch Geometric `Data` objects. The dataset is generated on-the-fly
with a fixed random seed for reproducibility.

### `train_utils.py` — train_model()
Standard training loop with:
- Adam optimizer
- ReduceLROnPlateau scheduler
- Early stopping
- Gradient clipping (to isolate over-squashing from exploding gradients)

---

## Hyperparameters

| Parameter     | Default | Notes |
|---------------|---------|-------|
| `hidden_dim`  | 64      | Increase for deeper trees |
| `num_layers`  | = depth | Must equal tree depth to reach leaves |
| `lr`          | 1e-3    | Adam learning rate |
| `num_epochs`  | 200     | Max epochs (early stopping applies) |
| `patience`    | 30      | Early stopping patience |
| `num_runs`    | 3       | For variance estimation |
| `batch_size`  | 32      | Graphs per batch |
| `num_classes` | 4       | Leaf label cardinality |

---

## References

1. Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR.
2. Gori, M., Monfardini, G., & Scarselli, F. (2005). *A new model for learning in graph domains.* IEEE IJCNN.
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR.
4. Wu, Z. et al. (2020). *A Comprehensive Survey on Graph Neural Networks.* IEEE TNNLS.
5. Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* arXiv.
