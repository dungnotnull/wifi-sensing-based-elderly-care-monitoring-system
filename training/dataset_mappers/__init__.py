"""ElderAL-CSI dataset mapper.

Converts ElderAL-CSI CSV files (MIMO format: 2 TX x 3 RX, 512 subcarriers)
into the model-compatible tensor format: (N, T=100, C=52) with binary labels.

File structure (as observed from sample):
  action{N}_activity_name/
    user{M}_position{P}_activity{A}/
      YYYYMMDD=HHMMSS_mimo5s_partN.csv

CSV columns: 4 metadata + 2TX * 3RX * 512subcarriers = 3076 total
  [0] activityID, [1] sujectID, [2] positionID, [3] timestamp
  [4..515]   amp_tx0_rx0_sub0..sub511
  [516..1027]  amp_tx0_rx1_sub0..sub511
  [1028..1539] amp_tx0_rx2_sub0..sub511
  [1540..2051] amp_tx1_rx0_sub0..sub511
  [2052..2563] amp_tx1_rx1_sub0..sub511
  [2564..3075] amp_tx1_rx2_sub0..sub511

Subcarrier selection: 512 subcarriers -> 52 (take every N-th subcarrier or first 52)
Label: derived from file path (action2_fall_new -> label=1 for fall, otherwise 0)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default subcarrier indices for 512 -> 52 downsampling (take every ~10th)
_DEFAULT_CSI_BENCH_SUBCARRIERS = 52

# Activity-to-label mapping — extensible for multi-class in the future
DEFAULT_LABEL_MAP = {
    "fall": 1,
    "fall_new": 1,
    "walk": 0,
    "idle": 0,
    "sit": 0,
    "stand": 0,
    "lie": 0,
    "bend": 0,
}

# Column indices for amplitude extraction
TX_RX_PAIRS = [
    ("tx0_rx0", range(4, 516)),
    ("tx0_rx1", range(516, 1028)),
    ("tx0_rx2", range(1028, 1540)),
    ("tx1_rx0", range(1540, 2052)),
    ("tx1_rx1", range(2052, 2564)),
    ("tx1_rx2", range(2564, 3076)),
]


def extract_label_from_path(file_path: str, label_map: Optional[dict] = None) -> int:
    """Extract binary label from file path.

    Example: action2_fall_new/user2_position6/... -> label=1
    """
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    path_lower = Path(file_path).as_posix().lower()
    for activity, label in label_map.items():
        if activity in path_lower:
            return label
    return 0


def load_elderal_csv(
    csv_path: str,
    n_subcarriers: int = _DEFAULT_CSI_BENCH_SUBCARRIERS,
    tx_rx_pair: str = "tx0_rx0",
    label_map: Optional[dict] = None,
) -> tuple[np.ndarray, int]:
    """Load a single ElderAL-CSI CSV file and return (amplitude_matrix, label).

    Args:
        csv_path: Path to the CSV file
        n_subcarriers: Number of subcarriers to keep (default 52)
        tx_rx_pair: Which TX-RX pair to extract (e.g., "tx0_rx0", "tx0_rx1")
        label_map: Activity name -> integer label mapping

    Returns:
        (amplitude, label) where amplitude.shape = (N_timesteps, n_subcarriers)
    """
    label = extract_label_from_path(csv_path, label_map)
    pair_key = tx_rx_pair.lower()
    col_range = None
    for key, rng in TX_RX_PAIRS:
        if key == pair_key:
            col_range = rng
            break
    if col_range is None:
        raise ValueError(f"Unknown tx_rx_pair: {tx_rx_pair}. Options: {[k for k,_ in TX_RX_PAIRS]}")

    data = []
    subcarrier_count = len(col_range)
    step = max(1, subcarrier_count // n_subcarriers)

    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split(",")
            if len(cols) < col_range.stop - 1:
                continue
            amps = [float(cols[i]) for i in col_range]
            # Downsample: take every step-th subcarrier, then first n_subcarriers
            selected = amps[::step][:n_subcarriers]
            if len(selected) < n_subcarriers:
                continue
            data.append(selected)

    return np.array(data, dtype=np.float32), label


def sliding_windows(
    amplitude: np.ndarray,
    window_size: int = 100,
    stride: int = 50,
    label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sliding windows from a long CSI trace.

    Args:
        amplitude: shape (N_timesteps, C_subcarriers)
        window_size: number of timesteps per window
        stride: stride between windows
        label: label to assign to all windows

    Returns:
        (windows, labels) where windows.shape = (N_windows, window_size, C)
    """
    n_timesteps, n_subcarriers = amplitude.shape
    if n_timesteps < window_size:
        return np.empty((0, window_size, n_subcarriers), dtype=np.float32), np.empty((0,), dtype=np.int64)

    starts = range(0, n_timesteps - window_size + 1, stride)
    n_windows = len(starts)
    windows = np.zeros((n_windows, window_size, n_subcarriers), dtype=np.float32)
    for i, s in enumerate(starts):
        windows[i] = amplitude[s : s + window_size]
    labels = np.full(n_windows, label, dtype=np.int64)
    return windows, labels


def load_elderal_directory(
    data_dir: str,
    n_subcarriers: int = _DEFAULT_CSI_BENCH_SUBCARRIERS,
    window_size: int = 100,
    stride: int = 50,
    tx_rx_pair: str = "tx0_rx0",
    label_map: Optional[dict] = None,
    recursive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all ElderAL-CSI CSV files from a directory tree.

    Returns:
        (data, labels) where data.shape = (N_samples, window_size, n_subcarriers)
    """
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    pattern = "**/*.csv" if recursive else "*.csv"
    total_files = 0

    for csv_path in Path(data_dir).glob(pattern):
        try:
            amp, label = load_elderal_csv(str(csv_path), n_subcarriers, tx_rx_pair, label_map)
            windows, wlabels = sliding_windows(amp, window_size, stride, label)
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(wlabels)
                total_files += 1
        except Exception:
            logger.debug("Skipping unreadable file: %s", csv_path, exc_info=True)

    if not all_windows:
        logger.warning("No valid CSV files found in %s", data_dir)
        return np.empty((0, window_size, n_subcarriers), dtype=np.float32), np.empty((0,), dtype=np.int64)

    data = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    idx = np.random.permutation(len(data))
    data, labels = data[idx], labels[idx]

    # Class distribution
    n_fall = int(np.sum(labels == 1))
    logger.info(
        "Loaded %d files -> %d windows | %d falls (%.1f%%)",
        total_files, len(data), n_fall, 100 * n_fall / max(1, len(data)),
    )
    return data, labels


class ElderALDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for ElderAL-CSI."""

    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def get_elderal_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    **kwargs,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load ElderAL-CSI data and create train/val dataloaders."""
    data, labels = load_elderal_directory(data_dir, **kwargs)
    if len(data) == 0:
        raise FileNotFoundError(f"No valid ElderAL-CSI data found in {data_dir}")

    n_val = max(1, int(len(data) * val_ratio))
    idx = np.random.permutation(len(data))

    train_ds = ElderALDataset(data[idx[n_val:]], labels[idx[n_val:]])
    val_ds = ElderALDataset(data[idx[:n_val]], labels[idx[:n_val]])

    logger.info("ElderAL-CSI: train=%d val=%d", len(train_ds), len(val_ds))
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
