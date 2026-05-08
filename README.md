# CS768 – On the Bottleneck of GNNs and its Practical Implications

**Authors:** Vijaya Raghavendra S (23B1042) · Ritwik Bavurupudi (23B0954) · Tharun Tej Banoth (23B0918)  
**Course:** CS768 – Graph Representation Learning

---

## Project Overview

This repository empirically investigates the **over-squashing** bottleneck in Graph Neural Networks, reproducing and extending the core experiments from:

> Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR 2021.

The key claim we validate is that standard GNNs such as **GCN** and **GIN** degrade toward random-chance accuracy as tree depth increases, because exponentially many leaf values are compressed into a fixed-size bottleneck. A single **Fully-Adjacent (FA) layer** appended after the GNN layers resolves this by giving every node direct attention access to every other node for one step, bypassing graph topology.

We also extend the analysis to the **ZINC chemical dataset**, showing that the FA layer improves molecular property prediction when properly regularized.

---

## Repository Structure

```text
root/
├── requirements.txt                       # pip dependencies (read before installing)
├── quick_demo.py                          # Start here (~2 min, no GPU needed)
│
├── src/
│   ├── __init__.py
│   └── gnn_implementations/
│       ├── __init__.py
│       ├── models.py                      # GCN, GIN, FALayer, GCN+FA, GIN+FA, get_model()
│       └── train_utils.py                 # train_epoch, evaluate, track_gradient_norms, etc.
│
└── experiments/
    ├── tree_neighbors_match/
    │   ├── dataset.py                     # TreeNeighborsMatchDataset + get_datasets()
    │   ├── run_experiment.py              # Full sweep: depths × models × runs → results.csv
    │   └── visualize.py                   # Publication-ready figures
    │
    └── chemical_datasets/
        ├── __init__.py
        └── run_chem.py                    # ZINC regression task (GCN vs GCN+FA)
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux / Mac
# venv\\Scripts\\activate      # Windows
```

Python 3.9–3.12 are supported. Python 3.8 is not supported.

### 2. Install PyTorch and PyTorch Geometric

Install the PyTorch build that matches your hardware:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU — CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU — CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install PyTorch Geometric:

```bash
pip install torch-geometric
```

> **Note:** Do not install `torch-scatter` or `torch-sparse` manually. Those are legacy packages and can cause version conflicts.

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

All commands below should be executed from the `root/` directory.

### Phase 1: Synthetic Proof (`TREE-NEIGHBORS-MATCH`)

Run the quick demo:

```bash
python quick_demo.py
```

Run the full sweep:

```bash
cd experiments/tree_neighbors_match
python run_experiment.py
```

Generate plots:

```bash
python visualize.py --plot all
```

You can also run specific plots:

```bash
python visualize.py --plot accuracy
python visualize.py --plot receptive_field
python visualize.py --plot gradient --depth 5
```

### Phase 2: Practical Implications (`ZINC` Regression)

This phase tests the FA layer on real-world molecular graphs for penalized logP prediction.

Run the regularized sweep:

```bash
python experiments/chemical_datasets/run_chem.py --models gcn gcn+fa --num_layers 4 --epochs 200 --dropout 0.5 --output_dir results_chem
```

Expected outcome: the **GCN+FA** model should achieve a lower MAE than the baseline **GCN**, demonstrating reduced topological bottleneck effects.

---

## Code Reference

### `models.py`

- `GCN`: Graph Convolutional Network backbone
- `GIN`: Graph Isomorphism Network backbone
- `FALayer`: Fully-Adjacent global self-attention layer
- `GCN+FA`, `GIN+FA`: Backbone models augmented with FA
- `get_model(name, ...)`: Factory for model creation

### `train_utils.py`

- `train_epoch(model, loader, optimizer, device, task)`
- `evaluate(model, loader, device, task)`
- `track_gradient_norms(model, batch, device)`
- `train_model(...)`
- `EarlyStopping`

### `dataset.py`

The TREE-NEIGHBORS-MATCH task builds a balanced binary tree of depth `r`. Each leaf receives a random label, one leaf is selected, and the root must predict the selected leaf’s label. This setup forces information through a narrow message-passing bottleneck and makes over-squashing visible.

---

## References

1. Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR.
2. Gori, M., Monfardini, G., & Scarselli, F. (2005). *A new model for learning in graph domains.* IEEE IJCNN.
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR.
4. Wu, Z. et al. (2020). *A Comprehensive Survey on Graph Neural Networks.* IEEE TNNLS.
5. Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* arXiv.
6. Vaswani, A. et al. (2017). *Attention is All You Need.* NeurIPS. ← FALayer attention mechanism
