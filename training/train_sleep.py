"""
SleepLSTM training script.

Trains the SleepLSTM model on mock sleep epoch data for verification.
Uses FocalLoss + oversampling to handle class imbalance (especially the awake class).

Usage:
    python training/train_sleep.py --mock --epochs 10
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.sleep.model import FocalLoss, SleepLSTM

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("models/sleep/checkpoints")
MOCK_TRAIN_PATH = Path("data/mock/sleep/sleep_epochs_train.npz")
MOCK_VAL_PATH = Path("data/mock/sleep/sleep_epochs_val.npz")


class SleepDataset(torch.utils.data.Dataset):
    """Dataset for sleep stage classification.

    Each sample is one night of epoch features.
    Returns (features, labels) of shape (N_epochs, 6) and (N_epochs,).
    """

    def __init__(self, data_path: Path) -> None:
        loaded = np.load(data_path)
        self.features = torch.tensor(loaded["features"], dtype=torch.float32)
        self.labels = torch.tensor(loaded["labels"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def compute_sleep_metrics(
    probs: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Compute accuracy and per-class F1 for sleep stage classification."""
    preds = torch.argmax(probs, dim=-1)
    flat_preds = preds.flatten().detach().cpu()
    flat_labels = labels.flatten().detach().cpu()

    correct = (flat_preds == flat_labels).sum().item()
    accuracy = correct / len(flat_labels)

    f1s = {}
    for cls_idx, cls_name in enumerate(["awake", "light", "deep"]):
        tp = ((flat_preds == cls_idx) & (flat_labels == cls_idx)).sum().item()
        fp = ((flat_preds == cls_idx) & (flat_labels != cls_idx)).sum().item()
        fn = ((flat_preds != cls_idx) & (flat_labels == cls_idx)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s[cls_name] = f1

    return {"accuracy": accuracy, **{f"f1_{k}": v for k, v in f1s.items()}, "macro_f1": np.mean(list(f1s.values()))}


def train_epoch(
    model: SleepLSTM,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        probs = model(features)  # (batch, N_epochs, 3)
        probs_flat = probs.reshape(-1, 3)
        labels_flat = labels.reshape(-1)
        loss = loss_fn(probs_flat, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_sleep_metrics(torch.cat(all_probs), torch.cat(all_labels))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: SleepLSTM,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        probs = model(features)
        probs_flat = probs.reshape(-1, 3)
        labels_flat = labels.reshape(-1)
        loss = loss_fn(probs_flat, labels_flat)
        total_loss += loss.item() * features.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_sleep_metrics(torch.cat(all_probs), torch.cat(all_labels))
    metrics["loss"] = avg_loss
    return metrics


def train_sleep(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: SleepLSTM,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
) -> SleepLSTM:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")

    best_f1 = 0.0
    best_state = None

    logger.info(f"Training SleepLSTM on {device}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Loss: FocalLoss (gamma=2.0)")

    for epoch in range(epochs):
        start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, loss_fn)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val macro F1: {val_metrics['macro_f1']:.3f} | "
            f"Val acc: {val_metrics['accuracy']:.3f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            path = CHECKPOINT_DIR / "sleep_lstm_best.pth"
            torch.save(best_state, path)
            logger.info(f"  Saved best checkpoint (macro F1={best_f1:.3f})")

    if best_state:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SleepLSTM")
    parser.add_argument("--mock", action="store_true", help="Use mock sleep data")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)

    train_path = Path(args.train_path) if args.train_path else MOCK_TRAIN_PATH
    val_path = Path(args.val_path) if args.val_path else MOCK_VAL_PATH

    if args.mock:
        from training.mock_data import generate_sleep_dataset
        fall_train, fall_val = generate_sleep_dataset()
        train_path, val_path = fall_train, fall_val

    if not train_path.exists():
        logger.warning(f"Sleep data not found at {train_path}. Generating mock data...")
        from training.mock_data import generate_sleep_dataset
        fall_train, fall_val = generate_sleep_dataset()
        train_path, val_path = fall_train, fall_val

    logger.info(f"Train: {train_path}\nVal: {val_path}")

    train_ds = SleepDataset(train_path)
    val_ds = SleepDataset(val_path)

    # Compute class frequencies for oversampling and focal loss alpha
    flat_train_labels = train_ds.labels.flatten()
    n_classes = 3
    class_counts = torch.zeros(n_classes, dtype=torch.float32)
    for cls in range(n_classes):
        class_counts[cls] = (flat_train_labels == cls).sum().float()
    logger.info(f"Per-class epoch counts (train): awake={int(class_counts[0])}, light={int(class_counts[1])}, deep={int(class_counts[2])}")

    # Inverse frequency weights for oversampling
    sample_weights = torch.zeros(len(flat_train_labels), dtype=torch.float32)
    for cls in range(n_classes):
        mask = flat_train_labels == cls
        sample_weights[mask] = 1.0 / class_counts[cls]

    # Per-night sample weights (average of per-epoch weights)
    epochs_per_night = train_ds.features.shape[1]
    night_weights = sample_weights.view(len(train_ds), -1).mean(dim=1)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=night_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Alpha for FocalLoss: inverse frequency, normalized to sum to n_classes
    alpha = n_classes * class_counts / class_counts.sum()
    logger.info(f"FocalLoss alpha weights: awake={alpha[0]:.3f}, light={alpha[1]:.3f}, deep={alpha[2]:.3f}")

    model = SleepLSTM(n_features=6, hidden_dim=64, n_layers=2, n_classes=3, dropout=0.3).to(device)

    model = train_sleep(
        train_loader, val_loader, model, device,
        epochs=args.epochs, lr=args.lr, class_weights=alpha,
    )

    # Final evaluation
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0, reduction="mean")
    final = evaluate(model, val_loader, device, loss_fn)
    logger.info(
        f"Final val: loss={final['loss']:.4f}, "
        f"F1(awake)={final['f1_awake']:.3f}, "
        f"F1(light)={final['f1_light']:.3f}, "
        f"F1(deep)={final['f1_deep']:.3f}, "
        f"macro_f1={final['macro_f1']:.3f}"
    )

    logger.info("SleepLSTM training verification complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
