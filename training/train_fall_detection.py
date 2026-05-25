"""
CSI-FallNet training script for fall detection.

Pipeline: CSI-Bench pre-training -> ElderAL-CSI fine-tuning -> in-situ data

For quick verification during development, uses synthetic CSI data.
Replace with real CSI-Bench/ElderAL-CSI data for actual training.

Usage:
    # Quick verification (synthetic data, 2 epochs)
    python training/train_fall_detection.py --quick

    # Full training with real data
    python training/train_fall_detection.py \
        --dataset data/processed/csibench \
        --epochs 50 \
        --output models/fall_detection/checkpoints/
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fall_detection.model import FallDetector
from training.dataset import SyntheticCSIDataset, get_dataloaders

logger = logging.getLogger(__name__)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 from logits."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train_epoch(
    model: FallDetector,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    fall_weight: float = 10.0,
) -> dict[str, float]:
    """Train one epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        # Apply data augmentation during training
        if model.training:
            data = augment_batch(data)

        logits, _ = model(data)

        # Weighted loss for class imbalance
        weights = torch.where(labels == 1, fall_weight, 1.0).to(device)
        loss = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss * weights).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: FallDetector,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        logits, _ = model(data)
        loss = criterion(logits, labels)
        total_loss += loss.item() * data.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = avg_loss
    return metrics


def augment_batch(data: torch.Tensor) -> torch.Tensor:
    """Apply CSI data augmentation: time shift, Gaussian noise, subcarrier dropout."""
    batch, T, C = data.shape

    # Random time shift (+/-5 frames)
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-5, 6, (1,)).item()
        if shift > 0:
            data = torch.cat([data[:, shift:], data[:, :shift]], dim=1)
        elif shift < 0:
            shift = -shift
            data = torch.cat([data[:, -shift:], data[:, :-shift]], dim=1)

    # Gaussian noise injection
    if torch.rand(1).item() < 0.5:
        noise_std = torch.rand(1).item() * 0.05
        data = data + torch.randn_like(data) * noise_std

    # Subcarrier dropout (randomly zero 10% of subcarriers)
    if torch.rand(1).item() < 0.3:
        mask = torch.bernoulli(torch.ones(C) * 0.9).to(data.device)
        data = data * mask.view(1, 1, C)

    return data


def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: FallDetector,
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    output_dir: Optional[str] = None,
) -> FallDetector:
    """Train CSI-FallNet. Returns trained model."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None

    logger.info(f"Training CSI-FallNet on {device}")
    logger.info(f"  Parameters: {model.get_parameter_count():,}")
    logger.info(f"  Epochs: {epochs}, LR: {lr}, Batch: {train_loader.batch_size}")

    for epoch in range(epochs):
        start = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.3f} | "
            f"Val acc: {val_metrics['accuracy']:.3f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, "csi_fallnet_best.pth")
                torch.save(best_state, path)
                logger.info(f"  Saved best checkpoint to {path} (F1={best_f1:.3f})")

    if best_state:
        model.load_state_dict(best_state)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSI-FallNet for fall detection")
    parser.add_argument("--dataset", default=None, help="Path to preprocessed dataset directory (optional)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/fall_detection/checkpoints/", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Quick verification with synthetic data (2 epochs)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples for --quick")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.dataset and os.path.exists(args.dataset):
        logger.info(f"Loading real dataset from {args.dataset}")
        logger.info("Real dataset loading not yet implemented. Use --quick for synthetic.")
        return
    else:
        if not args.dataset:
            logger.info("No dataset specified. Using synthetic CSI data for quick verification.")
        else:
            logger.warning(f"Dataset not found: {args.dataset}. Using synthetic data.")

        train_loader, val_loader = get_dataloaders(
            n_train=int(args.samples * 0.8),
            n_val=int(args.samples * 0.2),
            batch_size=args.batch_size,
        )

    model = FallDetector(
        n_subcarriers=52,
        sequence_length=100,
        dropout=0.3,  # lower dropout for quick verification
    ).to(device)

    n_params = model.get_parameter_count()
    logger.info(f"Model parameters: {n_params:,}")

    model = train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output,
    )

    # Final evaluation
    final_metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    logger.info(f"Final validation: loss={final_metrics['loss']:.4f}, F1={final_metrics['f1']:.3f}, acc={final_metrics['accuracy']:.3f}")

    # Smoke test: run inference on a single sample
    sample, label = val_loader.dataset[0]
    pred_class, pred_conf = model.predict(sample.unsqueeze(0).to(device))
    logger.info(f"Smoke test: true={label}, predicted={pred_class}, confidence={pred_conf:.3f}")

    logger.info("Training verification complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
