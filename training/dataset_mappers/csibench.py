"""CSI-Bench dataset mapper.

Converts CSI-Bench format (likely .npz with data/labels) into
model-compatible tensors: (N, T=100, C=52) with binary labels.

The CSI-Bench dataset from Kaggle comes as .mat files with
complex CSI matrices. This mapper expects preprocessed .npz
files. An adapter for .mat raw files is also provided with
clear documentation about required preprocessing steps.

Expected .npz format:
  data: (N, 100, 52) float32 CSI amplitude windows
  labels: (N,) int64 binary {0=non-fall, 1=fall}

For .mat raw files, the preprocessing pipeline is:
  raw CSI -> subcarrier selection (52) -> amplitude extraction
  -> sliding windows (T=100, stride=50) -> label assignment
  -> save as .npz
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 100
DEFAULT_SUBCARRIERS = 52


def load_csibench_npz(
    npz_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed CSI-Bench .npz file.

    Args:
        npz_path: Path to .npz file with keys 'data' and 'labels'

    Returns:
        (data, labels) where data.shape = (N, 100, 52)
    """
    loaded = np.load(npz_path)
    data = loaded["data"]
    labels = loaded["labels"]
    logger.info(
        "Loaded CSI-Bench: %d samples, shape=%s, %d falls (%.1f%%)",
        len(data), data.shape, int(np.sum(labels)),
        100 * float(np.mean(labels)),
    )
    return data, labels


def preprocess_csibench_mat(
    mat_path: str,
    n_subcarriers: int = DEFAULT_SUBCARRIERS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = 50,
    label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess raw CSI-Bench .mat file into model-ready tensors.

    This is a documented adapter for the raw .mat format. The actual
    .mat parsing depends on the specific MATLAB struct layout used by
    CSI-Bench and should be customized when the real data is available.

    Expected .mat structure (CSI-Bench convention):
      csi_data['amplitude']: (N_timesteps, N_subcarriers) complex CSI
      csi_data['label']: scalar activity label

    For now, this function provides the preprocessing pipeline contract.
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required for .mat file support. pip install scipy")

    mat = sio.loadmat(mat_path)
    available_keys = [k for k in mat.keys() if not k.startswith("__")]
    logger.info("CSI-Bench .mat keys available: %s", available_keys)

    csi = mat.get("amplitude")
    if csi is None:
        raise KeyError(
            f"Key 'amplitude' not found in {mat_path}. "
            f"Available keys: {available_keys}. "
            f"Update the key name in training/dataset_mappers/csibench.py to match."
        )

    # Total amplitude from real/imag
    if np.iscomplexobj(csi):
        amplitude = np.abs(csi)
    else:
        amplitude = csi.astype(np.float32)

    # Subcarrier selection: take first N or downsample
    if amplitude.shape[1] > n_subcarriers:
        step = max(1, amplitude.shape[1] // n_subcarriers)
        amplitude = amplitude[:, ::step][:, :n_subcarriers]

    # Sliding windows
    n_timesteps = amplitude.shape[0]
    if n_timesteps < window_size:
        logger.warning("CSI trace too short: %d < %d window", n_timesteps, window_size)
        return np.empty((0, window_size, n_subcarriers), dtype=np.float32), np.empty((0,), dtype=np.int64)

    starts = range(0, n_timesteps - window_size + 1, stride)
    n_windows = len(starts)
    data = np.zeros((n_windows, window_size, n_subcarriers), dtype=np.float32)
    for i, s in enumerate(starts):
        data[i] = amplitude[s : s + window_size]

    labels_arr = np.full(n_windows, label, dtype=np.int64)
    logger.info("Preprocessed %d windows from %s", n_windows, mat_path)
    return data, labels_arr


class CSIBenchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for CSI-Bench."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def get_csibench_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    is_mat: bool = False,
    **kwargs,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create CSI-Bench train/val dataloaders.

    Supports both .npz (preprocessed) and .mat (raw) formats.
    If val_path is None, splits off the given train data.

    Args:
        train_path: Path to .npz or directory of .mat files
        val_path: Optional separate validation path
        batch_size: Dataloader batch size
        val_ratio: Validation split ratio if val_path is None
        is_mat: If True, treat train_path as directory of .mat files
    """
    if is_mat:
        train_dir = Path(train_path)
        all_data = []
        all_labels = []
        for mat_file in train_dir.glob("*.mat"):
            try:
                d, lbl = preprocess_csibench_mat(str(mat_file), **kwargs)
                if len(d) > 0:
                    all_data.append(d)
                    all_labels.append(lbl)
            except Exception:
                logger.debug("Skipping %s", mat_file, exc_info=True)
        if not all_data:
            raise FileNotFoundError(f"No valid .mat files found in {train_path}")
        data = np.concatenate(all_data, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0)
    else:
        data, labels = load_csibench_npz(train_path)

    if val_path and not is_mat:
        val_data, val_labels = load_csibench_npz(val_path)
        n_val = len(val_data)
    else:
        n_val = max(1, int(len(data) * val_ratio))

    idx = np.random.permutation(len(data))

    if val_path and not is_mat:
        train_ds = CSIBenchDataset(data, labels)
        val_ds = CSIBenchDataset(val_data, val_labels)
    else:
        train_ds = CSIBenchDataset(data[idx[n_val:]], labels[idx[n_val:]])
        val_ds = CSIBenchDataset(data[idx[:n_val]], labels[idx[:n_val]])

    logger.info("CSI-Bench: train=%d val=%d", len(train_ds), len(val_ds))
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
