"""
Model evaluation for ElderCare.

Evaluates all models on mock datasets and generates a performance report.
Saves results to data/evaluation/report.json.

Usage:
    python -m pipeline.evaluate
    python -m pipeline.evaluate --output data/evaluation/report.json
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/evaluation/report.json")


def evaluate_fall_model(
    checkpoint_path: str = "models/fall_detection/checkpoints/csi_fallnet_best.pth",
) -> dict[str, Any]:
    """Evaluate FallDetector on mock ElderAL-CSI validation set."""
    from models.fall_detection.model import FallDetector

    logger.info("--- Fall Detection Evaluation ---")

    model = FallDetector(n_subcarriers=52, sequence_length=100)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        logger.warning(f"No checkpoint at {checkpoint_path}, using untrained model")

    model.eval()

    val_path = Path("data/mock/fall/elderal_csi_val.npz")
    if not val_path.exists():
        logger.warning(f"Mock dataset not found at {val_path}. Running synthetic eval.")
        from training.dataset import SyntheticCSIDataset
        ds = SyntheticCSIDataset(n_samples=400, seed=9999)
        data, labels = ds.data, ds.labels
    else:
        loaded = np.load(val_path)
        data = torch.tensor(loaded["data"], dtype=torch.float32)
        labels = torch.tensor(loaded["labels"], dtype=torch.long)

    all_preds = []
    all_confs = []
    with torch.no_grad():
        for i in range(len(data)):
            csi = data[i].unsqueeze(0)
            pred, conf = model.predict(csi)
            all_preds.append(pred)
            all_confs.append(conf)

    all_preds = np.array(all_preds)
    labels_np = labels.numpy()

    tp = int(np.sum((all_preds == 1) & (labels_np == 1)))
    fp = int(np.sum((all_preds == 1) & (labels_np == 0)))
    fn = int(np.sum((all_preds == 0) & (labels_np == 1)))
    tn = int(np.sum((all_preds == 0) & (labels_np == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(labels_np)

    n_fall_confident = int(np.sum(np.array(all_confs) >= 0.85))
    n_fall_actual = int(np.sum(labels_np == 1))

    result = {
        "model": "CSI-FallNet",
        "parameters": model.get_parameter_count(),
        "samples": len(labels_np),
        "fall_samples": n_fall_actual,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "high_confidence_detections": n_fall_confident,
        "checkpoint": checkpoint_path,
    }

    logger.info(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | Recall: {rec:.3f} | Precision: {prec:.3f}")
    return result


def evaluate_sleep_model(
    checkpoint_path: str = "models/sleep/checkpoints/sleep_lstm_best.pth",
) -> dict[str, Any]:
    """Evaluate SleepLSTM on mock sleep validation set."""
    from models.sleep.model import SleepLSTM

    logger.info("--- Sleep Model Evaluation ---")

    model = SleepLSTM(n_features=5, hidden_dim=64, n_layers=2, n_classes=3, dropout=0.3)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        logger.warning(f"No checkpoint at {checkpoint_path}, using untrained model")

    model.eval()

    val_path = Path("data/mock/sleep/sleep_epochs_val.npz")
    if not val_path.exists():
        logger.warning("No sleep mock data found, skipping evaluation")
        return {"model": "SleepLSTM", "status": "no_data"}

    loaded = np.load(val_path)
    features = torch.tensor(loaded["features"], dtype=torch.float32)
    labels = torch.tensor(loaded["labels"], dtype=torch.long)

    all_preds = []
    with torch.no_grad():
        for i in range(len(features)):
            x = features[i].unsqueeze(0)
            probs = model(x)
            preds = torch.argmax(probs, dim=-1)
            all_preds.append(preds)

    all_preds = torch.cat(all_preds)
    flat_preds = all_preds.flatten().numpy()
    flat_labels = labels.flatten().numpy()

    correct = int(np.sum(flat_preds == flat_labels))
    acc = correct / len(flat_labels)

    f1_per_class = {}
    for cls_idx, cls_name in enumerate(["awake", "light", "deep"]):
        tp = int(np.sum((flat_preds == cls_idx) & (flat_labels == cls_idx)))
        fp = int(np.sum((flat_preds == cls_idx) & (flat_labels != cls_idx)))
        fn = int(np.sum((flat_preds != cls_idx) & (flat_labels == cls_idx)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class[cls_name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    macro_f1 = round(np.mean([v["f1"] for v in f1_per_class.values()]), 4)

    result = {
        "model": "SleepLSTM",
        "parameters": sum(p.numel() for p in model.parameters()),
        "nights": len(features),
        "epochs": int(labels.numel()),
        "accuracy": round(acc, 4),
        "macro_f1": macro_f1,
        "per_class": f1_per_class,
        "checkpoint": checkpoint_path,
    }

    logger.info(f"  Accuracy: {acc:.3f} | Macro F1: {macro_f1:.3f}")
    for cls_name, m in f1_per_class.items():
        logger.info(f"  {cls_name:6s} F1: {m['f1']:.3f}")
    return result


def evaluate_vital_signs() -> dict[str, Any]:
    """Verify vital signs adapter is functional."""
    logger.info("--- Vital Signs Evaluation ---")

    try:
        from models.vital_signs.estimator import VitalsAdapter
        adapter = VitalsAdapter(n_subcarriers=52, sample_rate=50.0,
                                respiration_window_secs=5.0, heart_rate_window_secs=5.0)
        for _ in range(300):
            adapter.feed_frame(np.random.randn(52).astype(np.float32) * 0.1 + 1.0)
        result = {
            "model": "VitalsAdapter (wifi_densepose)",
            "status": "functional",
            "respiration_bpm": adapter.respiration_bpm,
            "heart_rate_bpm": adapter.heart_rate_bpm,
        }
        logger.info(f"  Respiration: {adapter.respiration_bpm} | Heart rate: {adapter.heart_rate_bpm}")
        return result
    except ImportError as e:
        logger.warning(f"wifi_densepose not available: {e}")
        return {"model": "VitalsAdapter", "status": "wifi_densepose not installed"}
    except Exception as e:
        logger.error(f"VitalsAdapter error: {e}")
        return {"model": "VitalsAdapter", "status": f"error: {e}"}


def evaluate_activity() -> dict[str, Any]:
    """Verify activity detector scenarios."""
    logger.info("--- Activity Detection Evaluation ---")
    from models.activity.detector import ActivityDetector

    detector = ActivityDetector(
        threshold_active=0.5, threshold_still=0.15,
        window_seconds=30.0, sample_rate=50.0,
    )

    # Active movement
    active_amp = np.random.randn(1500, 52) * 0.3 + 1.0
    state_a, _ = detector.update(active_amp, 10)
    logger.info(f"  Active movement ({10}:00) -> '{state_a}'")

    # Still (breathing only)
    still_amp = np.random.randn(1500, 52) * 0.05 + 1.0
    state_s, _ = detector.update(still_amp, 14)
    logger.info(f"  Still/breathing ({14}:00) -> '{state_s}'")

    # Inactivity (very low variance)
    inactivity_amp = np.random.randn(1500, 52) * 0.02 + 1.0
    state_i, _ = detector.update(inactivity_amp, 16)
    logger.info(f"  Low variance ({16}:00) -> '{state_i}'")

    # Inactivity at night should not trigger alert
    state_n, alert_n = detector.update(inactivity_amp, 2)
    logger.info(f"  Night inactivity ({2}:00) -> '{state_n}', alert='{alert_n}'")

    return {
        "model": "ActivityDetector",
        "status": "functional",
        "scenarios": {
            "active_day": {"state": state_a, "expected": "active"},
            "still_day": {"state": state_s, "expected": "still"},
            "inactive_day": {"state": state_i, "expected": "inactivity"},
            "inactive_night": {"state": state_n, "alert": str(alert_n), "expected_alert": "None"},
        },
    }


def run_all(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run all evaluations and save report."""
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {},
    }

    results["models"]["fall_detection"] = evaluate_fall_model()
    results["models"]["vital_signs"] = evaluate_vital_signs()
    results["models"]["activity"] = evaluate_activity()

    # Sleep eval only if checkpoint exists
    if Path("models/sleep/checkpoints/sleep_lstm_best.pth").exists():
        results["models"]["sleep"] = evaluate_sleep_model()
    else:
        results["models"]["sleep"] = {"model": "SleepLSTM", "status": "no_checkpoint_yet"}

    # Summary
    results["summary"] = {
        "fall_f1": results["models"]["fall_detection"].get("f1_score", "N/A"),
        "sleep_macro_f1": results["models"].get("sleep", {}).get("macro_f1", "N/A"),
        "vitals_status": results["models"]["vital_signs"].get("status"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nEvaluation report saved to {output_path}")
    return results


def calibrate_fall_model(
    checkpoint_path: str = "models/fall_detection/checkpoints/csi_fallnet_best.pth",
) -> dict[str, Any]:
    """Calibrate FallDetector confidence via temperature scaling.

    Loads model, runs on validation data, prints pre/post calibration ECE.
    """
    from models.calibration import TemperatureScaling
    from models.fall_detection.model import FallDetector

    logger.info("--- Fall Model Calibration ---")

    model = FallDetector(n_subcarriers=52, sequence_length=100)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        logger.warning(f"No checkpoint at {checkpoint_path}, using untrained model")

    model.eval()

    val_path = Path("data/mock/fall/elderal_csi_val.npz")
    if not val_path.exists():
        logger.warning(f"Mock dataset not found at {val_path}. Running synthetic eval.")
        from training.dataset import SyntheticCSIDataset
        ds = SyntheticCSIDataset(n_samples=400, seed=9999)
        data = torch.tensor(ds.data, dtype=torch.float32)
        labels = torch.tensor(ds.labels, dtype=torch.long)
    else:
        loaded = np.load(val_path)
        data = torch.tensor(loaded["data"], dtype=torch.float32)
        labels = torch.tensor(loaded["labels"], dtype=torch.long)

    # Collect raw logits (bypass temperature scaling for calibration)
    all_logits = []
    with torch.no_grad():
        for i in range(len(data)):
            csi = data[i].unsqueeze(0)
            # Forward through conv + lstm + fc to get raw logits
            x = csi.permute(0, 2, 1)
            x = torch.nn.functional.relu(model.bn1(model.conv1(x)))
            x = torch.nn.functional.relu(model.bn2(model.conv2(x)))
            x = model.pool(x)
            x = torch.nn.functional.relu(model.bn3(model.conv3(x)))
            x = model.pool(x)
            x = x.permute(0, 2, 1)
            lstm_out, _ = model.lstm(x)
            pooled = model.attention(lstm_out)
            pooled = model.dropout(pooled)
            pooled = torch.nn.functional.relu(model.fc1(pooled))
            pooled = torch.nn.functional.relu(model.fc2(pooled))
            logits = model.fc3(pooled)
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)

    # Pre-calibration ECE
    pre_calibrator = TemperatureScaling()  # T=1.0 (identity)
    pre_ece = pre_calibrator.compute_ece(all_logits, labels)
    logger.info(f"  Pre-calibration ECE:  {pre_ece:.4f}")

    # Calibrate
    nll = model.calibrate(all_logits, labels)

    # Post-calibration ECE
    post_ece = model.temperature_scaling.compute_ece(all_logits, labels)
    logger.info(f"  Post-calibration ECE: {post_ece:.4f}")
    logger.info(f"  Learned temperature:  {model.temperature_scaling.temperature.item():.4f}")
    logger.info(f"  Final NLL:            {nll:.4f}")

    result = {
        "pre_calibration_ece": round(pre_ece, 4),
        "post_calibration_ece": round(post_ece, 4),
        "temperature": round(model.temperature_scaling.temperature.item(), 4),
        "nll": round(nll, 4),
        "checkpoint": checkpoint_path,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ElderCare model evaluation")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    run_all(Path(args.output))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
