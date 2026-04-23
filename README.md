# CS768 – On the Bottleneck of GNNs and its Practical Implications

**Authors:** Vijaya Raghavendra S (23B1042) · Ritwik Bavurupudi (23B0954) · Tharun Tej Banoth (23B0918)
**Course:** CS768 – Graph Representation Learning

---

## Project Overview

This repository empirically investigates the **over-squashing** bottleneck in Graph Neural
Networks, reproducing and extending the core experiments from:

> Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR 2021.

The key claim we validate: standard GNNs (GCN, GIN) degrade to random-chance accuracy
as tree depth increases because exponentially many leaf values are compressed into a
fixed-size bottleneck. A single **Fully-Adjacent (FA) layer** appended after the GNN
layers resolves this completely by giving every node direct attention access to every
other node, bypassing graph topology entirely for one step.

---

## Repository Structure

```
CS768/
├── requirements.txt                       ← pip dependencies (read before installing)
├── quick_demo.py                          ← Start here (~2 min, no GPU needed)
│
├── code/
│   ├── __init__.py
│   └── gnn_implementations/
│       ├── __init__.py
│       ├── models.py                      ← GCN, GIN, FALayer, GCN+FA, GIN+FA, get_model()
│       └── train_utils.py                 ← train_epoch, evaluate, track_gradient_norms,
│                                             train_model, EarlyStopping
│
└── experiments/
    └── tree_neighbors_match/
        ├── dataset.py                     ← TreeNeighborsMatchDataset + get_datasets()
        ├── run_experiment.py              ← Full sweep: depths × models × runs → results.csv
        └── visualize.py                   ← 4 publication-ready figures
```

---

## Setup

### 1. Create a virtual environment (strongly recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows
```

Python 3.9 – 3.12 all work. Python 3.8 is not supported (lacks `numpy.random.default_rng`).

---

### 2. Install PyTorch

Pick exactly one command based on your hardware:

```bash
# CPU only (works everywhere, sufficient for depths ≤ 6)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU — CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU — CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric
```

> **⚠️ Do NOT install `torch-scatter` or `torch-sparse` manually.**
> These are legacy PyG 1.x packages. PyG ≥ 2.3 (which this project requires) ships its
> own C++ backend and will use pure-Python fallbacks automatically. Installing those
> packages by hand causes version-conflict errors with no obvious fix.

---

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

The full list with justification is in `requirements.txt`. Removed from the original list:
- `torch-scatter`, `torch-sparse` — legacy, causes conflicts (see above)
- `seaborn` — not imported anywhere in the codebase
- `scikit-learn` — not imported anywhere in the codebase
- `tqdm` — not imported anywhere in the codebase

---

### 5. Verify the installation (run this before anything else)

```bash
python code/gnn_implementations/models.py
```

Expected output:
```
==================================================
  models.py self-test
==================================================

[1] FALayer NaN regression test:
  FALayer unit test PASSED ✓

[2] All model forward passes:
  gcn        forward PASSED ✓  output shape: torch.Size([10, 4])
  gin        forward PASSED ✓  output shape: torch.Size([10, 4])
  gcn+fa     forward PASSED ✓  output shape: torch.Size([10, 4])
  gin+fa     forward PASSED ✓  output shape: torch.Size([10, 4])

All tests passed. models.py is correctly implemented.
```

If you see an `ImportError` here, your PyTorch / PyG install did not complete correctly.
Re-read steps 2–4.

---

## Running the Experiments

All commands run from the **CS768/** root directory unless stated otherwise.

---

### Quick Demo (~2 min on CPU)

Tests depths 2–4 with GIN vs GIN+FA:

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

The FA Boost column is the empirical signature of over-squashing: GIN collapses, GIN+FA stays high.

---

### Full Experiment Sweep

Reproduces the main paper result (all 4 models, depths 2–7, 3 runs each):

```bash
cd experiments/tree_neighbors_match
python run_experiment.py
```

Common options:

```bash
# Specific depths and models only
python run_experiment.py --depths 2 3 4 5 --models gcn gin gin+fa

# See per-epoch training logs
python run_experiment.py --verbose

# Fast single-depth check
python run_experiment.py --depth_single 4 --num_runs 1 --num_epochs 100

# Force CPU even if CUDA is available
python run_experiment.py --no_cuda
```

Results are written to `experiments/tree_neighbors_match/results/results.csv`.

---

### Generate Plots

```bash
cd experiments/tree_neighbors_match

# All four figures at once
python visualize.py --plot all

# Figure 1: accuracy vs depth (needs results.csv)
python visualize.py --plot accuracy

# Figure 2: theoretical receptive field growth (no experiment needed)
python visualize.py --plot receptive_field

# Figure 3: gradient norms per layer — GIN vs GIN+FA (runs a mini experiment)
python visualize.py --plot gradient --depth 5
```

All figures are saved to `experiments/tree_neighbors_match/results/`.

---

## Code Reference

### `models.py` — All model classes

| Symbol | Type | Description |
|--------|------|-------------|
| `GCN` | `nn.Module` | Graph Convolutional Network backbone (mean aggregation) |
| `GIN` | `nn.Module` | Graph Isomorphism Network backbone (sum aggregation, max 1-WL expressivity) |
| `GCNClassifier` | `nn.Module` | GCN + linear head for node/graph tasks |
| `GINClassifier` | `nn.Module` | GIN + linear head for node/graph tasks |
| `FALayer` | `nn.Module` | Fully-Adjacent layer — multi-head global self-attention |
| `GNNWithFA` | `nn.Module` | GCN or GIN backbone + FALayer + linear head |
| `get_model(name, ...)` | factory fn | Returns one of `gcn`, `gin`, `gcn+fa`, `gin+fa` |
| `_test_fa_layer()` | test fn | NaN regression test for FALayer |
| `_test_all_models()` | test fn | Forward-pass smoke test for all four variants |

**FALayer — implementation note:**
The FA layer implements scaled multi-head dot-product attention (Vaswani et al., 2017)
over all nodes in a graph simultaneously, completely ignoring graph edges. Two bugs
present in naive implementations are explicitly fixed:

1. **NaN from softmax on padded rows** — when a batch contains graphs of different sizes,
   `to_dense_batch` pads the smaller graphs. Padded query rows have all key scores set to
   `-inf`, causing `softmax → NaN`. Fixed with `torch.nan_to_num(attn, nan=0.0)` after
   softmax.

2. **Missing query-dimension mask** — masking only the key axis leaves padded query nodes
   producing non-zero attention outputs. Fixed by multiplying attention weights by a query
   mask before aggregating values.

---

### `train_utils.py` — Training utilities

| Function | Description |
|----------|-------------|
| `train_epoch(model, loader, optimizer, device, task)` | One pass over the dataset, returns avg loss |
| `evaluate(model, loader, device, task)` | Returns `(accuracy, avg_loss)` |
| `track_gradient_norms(model, batch, device)` | Returns dict of `{param_name: grad_L2_norm}` |
| `train_model(...)` | Full loop with early stopping, returns history dict |
| `EarlyStopping` | Standalone class tracking best metric + patience counter |

Gradient clipping (`clip_grad_norm_`, max=1.0) is applied inside `train_epoch` to isolate
over-squashing (vanishing gradients) from the confound of exploding gradients.

---

### `dataset.py` — TREE-NEIGHBORS-MATCH

| Class / Function | Description |
|-----------------|-------------|
| `build_binary_tree_graph(depth)` | Returns `(NetworkX DiGraph, root_idx, leaf_indices)` |
| `create_single_instance(depth, num_classes, rng)` | Returns one PyG `Data` object |
| `TreeNeighborsMatchDataset` | PyG `Dataset` of N instances at a fixed depth |
| `get_datasets(depth, ...)` | Returns `(train, val, test, feat_dim)` |

**Task:** A balanced binary tree of depth `r` is created. Each leaf receives a random
label in `{0, …, num_classes-1}`. One leaf is marked as "selected". The root must
predict the label of the selected leaf. This requires exactly `r` GNN layers and forces
2ʳ values through the bottleneck — the signature of over-squashing.

**Node features (size = num_classes + 1):**
- Leaf nodes: one-hot label + selection bit
- Internal nodes: all zeros

---

## Hyperparameters

| Parameter | Default | Where set | Notes |
|-----------|---------|-----------|-------|
| `hidden_dim` | 64 | `run_experiment.py` | Increase for depths > 7 |
| `num_layers` | = depth | `run_single()` | Must equal depth to reach leaves |
| `fa_heads` | 4 | `GNNWithFA.__init__` | `hidden_dim` must be divisible by this |
| `lr` | 1e-3 | `run_single()` | Adam |
| `num_epochs` | 200 | `run_experiment.py` | Early stopping applies |
| `patience` | 30 | `run_experiment.py` | Val-accuracy patience |
| `num_runs` | 3 | `run_experiment.py` | Seed = base + run×100 |
| `batch_size` | 32 | `run_experiment.py` | Graphs per batch |
| `num_classes` | 4 | `run_experiment.py` | Leaf label cardinality |
| `train_size` | 1000 | `run_experiment.py` | Graphs in training set |

---

## References

1. Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR.
2. Gori, M., Monfardini, G., & Scarselli, F. (2005). *A new model for learning in graph domains.* IEEE IJCNN.
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR.
4. Wu, Z. et al. (2020). *A Comprehensive Survey on Graph Neural Networks.* IEEE TNNLS.
5. Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* arXiv.
6. Vaswani, A. et al. (2017). *Attention is All You Need.* NeurIPS. ← FALayer attention mechanism
