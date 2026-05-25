"""
CSI-FallNet training script.

Supports four dataset types:
  --quick / --dataset-type synthetic  : synthetic data (fast verification)
  --dataset-type mock                 : mock ElderAL-CSI .npz from data/mock/
  --dataset-type csibench             : CSI-Bench .npz format
  --dataset-type elderal              : ElderAL-CSI CSV directory
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
    model, loader, optimizer, criterion, device, fall_weight=10.0,
) -> dict:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        if model.training:
            data = augment_batch(data)
        logits, _ = model(data)
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
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
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
    batch, T, C = data.shape
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-5, 6, (1,)).item()
        if shift > 0:
            data = torch.cat([data[:, shift:], data[:, :shift]], dim=1)
        elif shift < 0:
            shift = -shift
            data = torch.cat([data[:, -shift:], data[:, :-shift]], dim=1)
    if torch.rand(1).item() < 0.5:
        noise_std = torch.rand(1).item() * 0.05
        data = data + torch.randn_like(data) * noise_std
    if torch.rand(1).item() < 0.3:
        mask = torch.bernoulli(torch.ones(C) * 0.9).to(data.device)
        data = data * mask.view(1, 1, C)
    return data


def train(
    train_loader, val_loader, model, device, epochs=2, lr=1e-3, weight_decay=1e-4, output_dir=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_state = None
    logger.info(f"Training CSI-FallNet on {device} | {model.get_parameter_count():,} params | {epochs} epochs")

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
                logger.info(f"  Saved best checkpoint (F1={best_f1:.3f})")

    if best_state:
        model.load_state_dict(best_state)
    return model


def _resolve_dataloaders(args) -> tuple:
    """Determine dataset source and return (train_loader, val_loader)."""

    if args.dataset_type == "mock" or (args.dataset and Path(args.dataset).match("data/mock/fall/*.npz")):
        from training.mock_data import generate_fall_dataset
        train_p, val_p = generate_fall_dataset()
        from training.dataset_mappers.csibench import get_csibench_dataloaders
        return get_csibench_dataloaders(str(train_p), str(val_p), batch_size=args.batch_size)

    if args.dataset_type == "csibench" and args.dataset and os.path.exists(args.dataset):
        from training.dataset_mappers.csibench import get_csibench_dataloaders
        return get_csibench_dataloaders(args.dataset, batch_size=args.batch_size)

    if args.dataset_type == "elderal" and args.dataset and os.path.exists(args.dataset):
        from training.dataset_mappers import get_elderal_dataloaders
        return get_elderal_dataloaders(args.dataset, batch_size=args.batch_size)

    if args.dataset and os.path.exists(args.dataset) and args.dataset_type == "auto":
        path = Path(args.dataset)
        if path.suffix == ".npz":
            from training.dataset_mappers.csibench import get_csibench_dataloaders
            logger.info("Auto-detected CSI-Bench .npz format")
            return get_csibench_dataloaders(str(path), batch_size=args.batch_size)
        if path.is_dir() and any(path.glob("*.csv")):
            from training.dataset_mappers import get_elderal_dataloaders
            logger.info("Auto-detected ElderAL-CSI CSV directory")
            return get_elderal_dataloaders(str(path), batch_size=args.batch_size)

    # Fallback: synthetic
    if args.dataset and not os.path.exists(args.dataset):
        logger.warning("Dataset not found: %s", args.dataset)
    logger.info("Using synthetic CSI data")
    return get_dataloaders(
        n_train=int(args.samples * 0.8), n_val=int(args.samples * 0.2),
        batch_size=args.batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSI-FallNet for fall detection")
    parser.add_argument("--dataset", default=None, help="Path to dataset file or directory")
    parser.add_argument("--dataset-type", default="auto",
                       choices=["auto", "synthetic", "csibench", "elderal", "mock"],
                       help="Dataset type")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/fall_detection/checkpoints/", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Quick verification with synthetic data (2 epochs)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples for --quick")
    args = parser.parse_args()

    if args.quick:
        args.dataset_type = "synthetic"

    device = torch.device(args.device)
    train_loader, val_loader = _resolve_dataloaders(args)

    model = FallDetector(n_subcarriers=52, sequence_length=100, dropout=0.3).to(device)
    n_params = model.get_parameter_count()
    logger.info("Model parameters: %s", f"{n_params:,}")

    model = train(train_loader, val_loader, model, device, epochs=args.epochs, lr=args.lr, output_dir=args.output)

    final_metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    logger.info(
        "Final validation: loss=%.4f, F1=%.3f, acc=%.3f",
        final_metrics["loss"], final_metrics["f1"], final_metrics["accuracy"],
    )

    sample, label = val_loader.dataset[0]
    pred_class, pred_conf = model.predict(sample.unsqueeze(0).to(device))
    logger.info("Smoke test: true=%d, predicted=%d, confidence=%.3f", label, pred_class, pred_conf)
    logger.info("Training verification complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
