# CS768 Mid-Semester Project
## On the Bottleneck of Graph Neural Networks and its Practical Implications

### Authors
- Vijaya Raghavendra S (23B1042)
- Ritwik Bavurupudi (23B0954)
- Tharun Tej Banoth (23B0918)

---

## Project Overview

Graph Neural Networks (GNNs) are widely used for learning from graph-structured data such as social networks, molecular graphs, and program structures. However, training deep GNNs to capture **long-range dependencies** remains challenging.

This project studies a fundamental limitation in message-passing GNN architectures known as **over-squashing**, where an exponentially growing amount of information from distant nodes is compressed into fixed-size node embeddings.

Our goal is to analyze the theoretical causes of this bottleneck and explore architectural approaches that mitigate its effects.

---

## Problem Statement

In a standard message-passing Graph Neural Network (GNN), each node updates its representation by aggregating information from its neighbors.

For a graph **G = (V, E)**, the representation of node **v** at layer **k** is computed using:
- the previous representation of node **v**
- the representations of its neighboring nodes
- a learnable neural network function

In simple terms, each layer allows information to travel **one hop further in the graph**.

Therefore, if two nodes are **r hops apart**, the network must have **at least r layers** for information to propagate between them.

However, the number of nodes that influence a given node grows **exponentially** with the number of layers. This expanding region is called the **receptive field**.

As the receptive field grows, the model must compress a large amount of information into a fixed-size vector representation. This compression creates a structural bottleneck known as **over-squashing**.

---

## Key Concepts

### Over-Squashing
Occurs when information from exponentially many nodes must pass through a small set of edges or nodes, forcing excessive compression.

### Over-Smoothing
Occurs when repeated message passing causes node representations to become indistinguishable.

These are **distinct problems**, though both affect deep GNN performance.

---

## Literature Review

Our study is based on the following important works:

- **Alon & Yahav (2021)** – Identified the over-squashing bottleneck in GNNs.
- **Xu et al. (2019)** – Introduced Graph Isomorphism Networks (GIN) and analyzed GNN expressiveness.
- **Wu et al. (2020)** – Provided a comprehensive survey of GNN architectures.
- **Bahdanau et al. (2014)** – Introduced attention mechanisms in neural machine translation, inspiring similar solutions for GNN bottlenecks.

---

## Proposed Approach

We analyze the structural bottleneck in standard GNN architectures and study possible mitigation strategies, including:

- Topological rewiring
- Virtual nodes
- Fully-Adjacent (FA) layers enabling global information flow

---

## Future Work

For the final stage of the project, we plan to:

- Reproduce **TREE-NEIGHBORS-MATCH experiments** to analyze gradient degradation caused by over-squashing.
- Implement **Fully-Adjacent (FA) layers** to enable global message passing.
- Evaluate performance improvements on graph-based datasets such as chemical molecular graphs.

---

## References

1. Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications*. ICLR.  
2. Gori, M., Monfardini, G., & Scarselli, F. (2005). *A new model for learning in graph domains*. IEEE IJCNN.  
3. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?*. ICLR.  
4. Wu, Z. et al. (2020). *A Comprehensive Survey on Graph Neural Networks*. IEEE TNNLS.  
5. Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv.

---

## Repository Structure
CS768/
│
├── README.md
├── report/
│ └── midsem_report.pdf
├── experiments/
│ └── tree_neighbors_match/
└── code/
└── gnn_implementations/


---

## Course

**CS768 – Graph Representation Learning**

