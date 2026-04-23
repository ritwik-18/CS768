"""
train_utils.py - Training Loop and Evaluation Utilities
=========================================================
Provides:
  - train_epoch()   : one training pass over the dataset
  - evaluate()      : accuracy evaluation on a dataset
  - track_gradients(): tracks gradient norms per layer (to visualize over-squashing)
  - EarlyStopping   : simple early stopping callback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                task: str = "node") -> float:
    """
    Run one training epoch.

    Args:
        model     : GNN model
        loader    : DataLoader for training data
        optimizer : optimizer (e.g. Adam)
        device    : cpu or cuda
        task      : 'node' (root prediction) or 'graph'

    Returns:
        Average cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        if task == "node":
            # Only compute loss at root nodes (root_mask == True)
            root_out = out[batch.root_mask]        # [num_graphs, num_classes]
            root_y = batch.y[batch.root_mask]      # [num_graphs]
            loss = F.cross_entropy(root_out, root_y)
            num_samples += root_y.size(0)
        else:
            # Graph-level prediction
            loss = F.cross_entropy(out, batch.y)
            num_samples += batch.y.size(0)

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * (root_y.size(0) if task == "node" else batch.y.size(0))

    return total_loss / num_samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             task: str = "node") -> Tuple[float, float]:
    """
    Evaluate model accuracy and loss.

    Returns:
        (accuracy, avg_loss) tuple
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)

        if task == "node":
            root_out = out[batch.root_mask]
            root_y = batch.y[batch.root_mask]
            loss = F.cross_entropy(root_out, root_y)
            preds = root_out.argmax(dim=-1)
            correct += (preds == root_y).sum().item()
            num_samples += root_y.size(0)
            total_loss += loss.item() * root_y.size(0)
        else:
            loss = F.cross_entropy(out, batch.y)
            preds = out.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            num_samples += batch.y.size(0)
            total_loss += loss.item() * batch.y.size(0)

    accuracy = correct / num_samples
    avg_loss = total_loss / num_samples
    return accuracy, avg_loss


# ---------------------------------------------------------------------------
# Gradient norm tracking (to measure over-squashing)
# ---------------------------------------------------------------------------

def track_gradient_norms(model: nn.Module,
                         batch,
                         device: torch.device) -> Dict[str, float]:
    """
    Compute gradient norms for each parameter group.
    Used to empirically measure how gradients vanish due to over-squashing.

    A sharp drop in gradient norm for early layers indicates that
    information from distant nodes is not flowing back — a symptom of
    over-squashing.

    Returns:
        Dict mapping layer name to its gradient L2 norm.
    """
    model.train()
    batch = batch.to(device)

    # Forward + backward on a single batch
    out = model(batch.x, batch.edge_index, batch.batch)
    root_out = out[batch.root_mask]
    root_y = batch.y[batch.root_mask]
    loss = F.cross_entropy(root_out, root_y)
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0

    model.zero_grad()
    return grad_norms


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                device: torch.device,
                num_epochs: int = 100,
                patience: int = 20,
                task: str = "node",
                verbose: bool = True) -> Dict:
    """
    Full training loop with early stopping.

    Returns:
        history dict with keys: train_loss, val_loss, val_acc, best_val_acc
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, task)
        val_acc, val_loss = evaluate(model, val_loader, device, task)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}.")
            break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


# ---------------------------------------------------------------------------
# Early Stopping (standalone class version)
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Tracks validation metric and signals when to stop training."""

    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False
        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
