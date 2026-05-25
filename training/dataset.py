"""
Synthetic CSI dataset generator for quick verification.

Generates CSI amplitude windows matching CSI-Bench format:
  - Shape: (N, T=100, C=52)
  - Labels: 0 (non-fall), 1 (fall)
  - Class imbalance ~1:10 (realistic)
  - Falls: sharp amplitude drop + recovery pattern
  - Non-falls: normal movement patterns + breathing + idle
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SyntheticCSIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset of synthetic CSI windows for fall detection."""

    def __init__(
        self,
        n_samples: int = 1000,
        sequence_length: int = 100,
        n_subcarriers: int = 52,
        fall_ratio: float = 0.09,  # ~1:10 class imbalance
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.n_subcarriers = n_subcarriers
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

        self.data, self.labels = self._generate(n_samples, fall_ratio, noise_std)

    def _generate(self, n: int, fall_ratio: float, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.zeros((n, self.sequence_length, self.n_subcarriers), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)

        n_falls = max(1, int(n * fall_ratio))

        for i in range(n):
            is_fall = i < n_falls
            labels[i] = 1 if is_fall else 0

            if is_fall:
                data[i] = self._generate_fall_window()
            else:
                c = self.rng.choice(["idle", "breathing", "moving"])
                if c == "idle":
                    data[i] = self._generate_idle_window()
                elif c == "breathing":
                    data[i] = self._generate_breathing_window()
                else:
                    data[i] = self._generate_movement_window()

            data[i] += self.rng.normal(0, noise_std, (self.sequence_length, self.n_subcarriers))

        labels[:n_falls] = 1
        self.rng.shuffle(labels)
        # Re-apply correct class split after shuffle
        fall_indices = np.where(labels == 1)[0]
        for idx in fall_indices:
            data[idx] = self._generate_fall_window()
            data[idx] += self.rng.normal(0, noise_std, (self.sequence_length, self.n_subcarriers))

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def _generate_fall_window(self) -> np.ndarray:
        """Generate a fall CSI window: sharp amplitude drop then slow recovery."""
        T, C = self.sequence_length, self.n_subcarriers
        window = np.ones((T, C))

        fall_start = self.rng.randint(20, 40)
        fall_duration = self.rng.randint(10, 25)
        fall_magnitude = self.rng.uniform(0.4, 0.7)

        t_axis = np.arange(T)[:, np.newaxis]

        for i in range(C):
            subcarrier_phase = self.rng.uniform(0, 2 * np.pi)
            subcarrier_scale = self.rng.uniform(0.8, 1.2)
            envelope = np.ones(T)

            mask = t_axis.flatten() >= fall_start
            envelope[mask] = fall_magnitude + (1.0 - fall_magnitude) * (
                1.0 - np.exp(-(t_axis.flatten()[mask] - fall_start) * 0.15)
            )
            envelope[:fall_start] *= (0.9 + 0.1 * np.sin(t_axis[:fall_start].flatten() * 0.3 + subcarrier_phase))

            window[:, i] = envelope * subcarrier_scale

        return window

    def _generate_idle_window(self) -> np.ndarray:
        T, C = self.sequence_length, self.n_subcarriers
        window = np.ones((T, C))
        window += self.rng.normal(0, 0.02, (T, C))
        return window

    def _generate_breathing_window(self) -> np.ndarray:
        T, C = self.sequence_length, self.n_subcarriers
        t_axis = np.arange(T)[:, np.newaxis] / 50.0
        window = np.ones((T, C))

        breathing_freq = self.rng.uniform(0.15, 0.35)
        breathing_amp = self.rng.uniform(0.03, 0.08)

        for i in range(C):
            phase = self.rng.uniform(0, 2 * np.pi)
            scale = self.rng.uniform(0.8, 1.2)
            window[:, i] += breathing_amp * scale * np.sin(2 * np.pi * breathing_freq * t_axis.flatten() + phase)

        return window

    def _generate_movement_window(self) -> np.ndarray:
        T, C = self.sequence_length, self.n_subcarriers
        t_axis = np.arange(T)[:, np.newaxis] / 50.0
        window = np.ones((T, C))

        for i in range(C):
            phase = self.rng.uniform(0, 2 * np.pi)
            freq = self.rng.uniform(1.0, 4.0)
            amp = self.rng.uniform(0.15, 0.35)
            scale = self.rng.uniform(0.8, 1.2)
            window[:, i] += amp * scale * np.sin(2 * np.pi * freq * t_axis.flatten() + phase)

        return window

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def get_dataloaders(
    n_train: int = 800,
    n_val: int = 200,
    batch_size: int = 32,
    sequence_length: int = 100,
    n_subcarriers: int = 52,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val dataloaders with synthetic CSI data."""

    train_ds = SyntheticCSIDataset(
        n_samples=n_train,
        sequence_length=sequence_length,
        n_subcarriers=n_subcarriers,
        seed=seed,
    )
    val_ds = SyntheticCSIDataset(
        n_samples=n_val,
        sequence_length=sequence_length,
        n_subcarriers=n_subcarriers,
        seed=seed + 1000,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Log class distribution
    train_labels = train_ds.labels.numpy()
    val_labels = val_ds.labels.numpy()
    logger.info(f"Train: {len(train_ds)} samples, {np.sum(train_labels)} falls ({100*np.mean(train_labels):.1f}%)")
    logger.info(f"Val:   {len(val_ds)} samples, {np.sum(val_labels)} falls ({100*np.mean(val_labels):.1f}%)")

    return train_loader, val_loader
