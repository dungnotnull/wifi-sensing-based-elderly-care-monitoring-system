"""
Mock dataset generator for ElderCare training verification.

Generates two datasets:
  1. ElderAL-CSI style fall detection data (T=100, C=52, binary labels)
  2. Sleep epoch data (N_epochs, 4 features + stage labels)

Saves to data/mock/ for training verification without requiring
heavy downloads of CSI-Bench or ElderAL-CSI.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/mock")


def generate_fall_dataset(
    n_samples: int = 2000,
    n_subcarriers: int = 52,
    sequence_length: int = 100,
    fall_ratio: float = 0.1,
    noise_std: float = 0.05,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Generate mock ElderAL-CSI fall dataset and save to disk.

    Returns (train_path, val_path).
    """
    output_dir = output_dir or (DATA_DIR / "fall")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    n_falls = max(1, int(n_samples * fall_ratio))
    data = np.zeros((n_samples, sequence_length, n_subcarriers), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        is_fall = i < n_falls
        labels[i] = 1 if is_fall else 0

        if is_fall:
            data[i] = _make_fall_window(rng, sequence_length, n_subcarriers)
        else:
            kind = rng.choice(["idle", "breathing", "walking"])
            if kind == "idle":
                data[i] = _make_idle_window(rng, sequence_length, n_subcarriers)
            elif kind == "breathing":
                data[i] = _make_breathing_window(rng, sequence_length, n_subcarriers)
            else:
                data[i] = _make_walking_window(rng, sequence_length, n_subcarriers)
        data[i] += rng.normal(0, noise_std, (sequence_length, n_subcarriers))

    idx = rng.permutation(n_samples)
    data, labels = data[idx], labels[idx]

    n_train = int(n_samples * 0.8)
    split_data = {"train": (data[:n_train], labels[:n_train]), "val": (data[n_train:], labels[n_train:])}

    paths = {}
    for split, (d, lab) in split_data.items():
        path = output_dir / f"elderal_csi_{split}.npz"
        np.savez_compressed(path, data=d, labels=lab)
        paths[split] = path
        label_counts = dict(zip(*np.unique(lab, return_counts=True)))
        logger.info(f"  {split}: {len(d)} samples, label dist={label_counts}")

    return paths["train"], paths["val"]


def generate_sleep_dataset(
    n_nights: int = 100,
    epochs_per_night: int = 480,  # 480 epochs = 8 hours
    n_features: int = 6,
    seed: int = 99,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Generate mock sleep data (epoch features + stage labels) and save to disk.

    Sleep stages: 0=awake, 1=light, 2=deep
    Returns (train_path, val_path).
    """
    output_dir = output_dir or (DATA_DIR / "sleep")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    all_features = []
    all_labels = []

    for _ in range(n_nights):
        features, labels = _make_sleep_night(rng, epochs_per_night, n_features)
        all_features.append(features)
        all_labels.append(labels)

    features = np.stack(all_features, axis=0).astype(np.float32)  # (n_nights, 480, 4)
    labels = np.stack(all_labels, axis=0).astype(np.int64)

    n_train = int(n_nights * 0.8)
    split_data = {
        "train": (features[:n_train], labels[:n_train]),
        "val": (features[n_train:], labels[n_train:]),
    }

    paths = {}
    for split, (f, lab) in split_data.items():
        path = output_dir / f"sleep_epochs_{split}.npz"
        np.savez_compressed(path, features=f, labels=lab)
        paths[split] = path
        # Flatten to count per-stage
        flat = lab.flatten()
        counts = dict(zip(*np.unique(flat, return_counts=True)))
        logger.info(f"  {split}: {len(f)} nights, {lab.size} epochs, stage dist={counts}")

    return paths["train"], paths["val"]


def _make_fall_window(rng: np.random.Generator, T: int, C: int) -> np.ndarray:
    """Sharp amplitude drop + recovery pattern."""
    window = np.ones((T, C))
    fall_start = int(rng.integers(20, 40))
    fall_magnitude = float(rng.uniform(0.3, 0.6))
    t = np.arange(T)

    for c in range(C):
        phase = float(rng.uniform(0, 2 * np.pi))
        scale = float(rng.uniform(0.8, 1.2))
        envelope = np.ones(T)
        mask = t >= fall_start
        envelope[mask] = fall_magnitude + (1.0 - fall_magnitude) * (1.0 - np.exp(-(t[mask].astype(float) - fall_start) * 0.12))
        envelope[:fall_start] *= (0.9 + 0.1 * np.sin(t[:fall_start].astype(float) * 0.3 + phase))
        window[:, c] = envelope * scale

    return window


def _make_idle_window(rng: np.random.Generator, T: int, C: int) -> np.ndarray:
    window = np.ones((T, C))
    window += rng.normal(0, 0.02, (T, C))
    return window


def _make_breathing_window(rng: np.random.Generator, T: int, C: int) -> np.ndarray:
    t = np.arange(T) / 50.0
    window = np.ones((T, C))
    freq = float(rng.uniform(0.15, 0.35))
    amp = float(rng.uniform(0.03, 0.08))
    for c in range(C):
        phase = float(rng.uniform(0, 2 * np.pi))
        scale = float(rng.uniform(0.8, 1.2))
        window[:, c] += amp * scale * np.sin(2 * np.pi * freq * t + phase)
    return window


def _make_walking_window(rng: np.random.Generator, T: int, C: int) -> np.ndarray:
    t = np.arange(T) / 50.0
    window = np.ones((T, C))
    for c in range(C):
        phase = float(rng.uniform(0, 2 * np.pi))
        freq = float(rng.uniform(1.0, 4.0))
        amp = float(rng.uniform(0.15, 0.35))
        scale = float(rng.uniform(0.8, 1.2))
        window[:, c] += amp * scale * np.sin(2 * np.pi * freq * t + phase)
    return window


def _make_sleep_night(
    rng: np.random.Generator, n_epochs: int, n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one night of sleep data with balanced stage distribution.

    Features are designed to be realistically discriminative:
      - Awake: high movement variance (0.10-0.40), high burst count (3-15),
               high wakefulness index (0.4-0.9), elevated respiration (16-22 bpm)
      - Light:  moderate movement (0.02-0.12), low burst count (0-3),
               wakefulness index (0.05-0.30), moderate respiration (13-17 bpm)
      - Deep:   minimal movement (0.005-0.04), zero bursts normally,
               wakefulness index (0.0-0.10), slow respiration (11-15 bpm)

    Target distribution: awake ~22%, light ~40%, deep ~38%
    """
    features = np.zeros((n_epochs, n_features), dtype=np.float32)
    labels = np.zeros(n_epochs, dtype=np.int64)

    state_chains = [
        [0] * int(rng.integers(12, 25)),
        [1] * int(rng.integers(20, 50)),
        [2] * int(rng.integers(40, 80)),
        [1] * int(rng.integers(15, 30)),
        [0] * int(rng.integers(6, 15)),
        [1] * int(rng.integers(20, 40)),
        [2] * int(rng.integers(30, 60)),
        [1] * int(rng.integers(15, 30)),
        [0] * int(rng.integers(3, 10)),
        [1] * int(rng.integers(15, 25)),
        [2] * int(rng.integers(20, 40)),
        [1] * int(rng.integers(20, 40)),
        [0] * int(rng.integers(8, 20)),
    ]
    all_stages: list[int] = []
    for chain in state_chains:
        all_stages.extend(chain)
    if len(all_stages) > n_epochs:
        all_stages = all_stages[:n_epochs]
    else:
        all_stages.extend([rng.integers(0, 3) for _ in range(n_epochs - len(all_stages))])
    all_stages = all_stages[:n_epochs]

    prev_movement = 0.0
    for t, stage in enumerate(all_stages):
        labels[t] = stage

        if stage == 0:  # awake
            resp = rng.normal(18, 2.5)
            resp_std = rng.uniform(1.0, 3.5)
            movement = rng.uniform(0.10, 0.40)
            bursts = float(rng.integers(3, 15))
            wakefulness = rng.uniform(0.40, 0.90)
        elif stage == 1:  # light
            resp = rng.normal(15, 1.5)
            resp_std = rng.uniform(0.5, 1.5)
            movement = rng.uniform(0.02, 0.12)
            bursts = float(rng.integers(0, 3))
            wakefulness = rng.uniform(0.05, 0.30)
        else:  # deep
            resp = rng.normal(13, 1.0)
            resp_std = rng.uniform(0.2, 0.8)
            movement = rng.uniform(0.005, 0.04)
            bursts = float(rng.integers(0, 2))
            wakefulness = rng.uniform(0.0, 0.10)

        movement_rate_of_change = movement - prev_movement
        prev_movement = movement

        features[t] = [resp, resp_std, movement, bursts, movement_rate_of_change, wakefulness]

    return features, labels


def all_datasets() -> dict:
    """Generate all mock datasets. Returns dict of paths."""
    logger.info("Generating mock ElderAL-CSI fall dataset...")
    fall_paths = generate_fall_dataset()

    logger.info("Generating mock sleep epoch dataset...")
    sleep_paths = generate_sleep_dataset()

    return {
        "fall_train": fall_paths[0],
        "fall_val": fall_paths[1],
        "sleep_train": sleep_paths[0],
        "sleep_val": sleep_paths[1],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = all_datasets()
    print("\nMock datasets created at data/mock/")
    for k, p in paths.items():
        print(f"  {k}: {p}")
