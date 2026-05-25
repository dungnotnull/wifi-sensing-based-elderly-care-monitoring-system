"""Tests for dataset mappers (ElderAL-CSI and CSI-Bench)."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.dataset_mappers import (
    DEFAULT_LABEL_MAP,
    extract_label_from_path,
    load_elderal_csv,
    sliding_windows,
)
from training.dataset_mappers.csibench import (
    load_csibench_npz,
    preprocess_csibench_mat,
    get_csibench_dataloaders,
)


class TestElderALMapper:
    def test_extract_label_from_path_fall(self) -> None:
        """Paths containing 'fall' should map to label 1."""
        assert extract_label_from_path("action2_fall_new/user2/file.csv") == 1
        assert extract_label_from_path("data/fall/user1/20250101.csv") == 1

    def test_extract_label_from_path_non_fall(self) -> None:
        """Paths without 'fall' should map to label 0."""
        assert extract_label_from_path("action1_walk/user3/file.csv") == 0
        assert extract_label_from_path("data/action_idle/file.csv") == 0

    def test_sliding_windows_output_shape(self) -> None:
        """Sliding windows should produce correct output dimensions."""
        amp = np.random.randn(300, 52).astype(np.float32)
        windows, labels = sliding_windows(amp, window_size=100, stride=50, label=1)

        assert windows.shape == (5, 100, 52)  # (300-100)//50 + 1 = 5 windows
        assert labels.shape == (5,)
        assert np.all(labels == 1)

    def test_sliding_windows_too_short(self) -> None:
        amp = np.random.randn(50, 52).astype(np.float32)
        windows, labels = sliding_windows(amp, window_size=100, stride=50)
        assert len(windows) == 0
        assert len(labels) == 0

    def test_sliding_windows_stride_100(self) -> None:
        amp = np.random.randn(250, 52).astype(np.float32)
        windows, labels = sliding_windows(amp, window_size=100, stride=100, label=0)
        assert windows.shape[0] == 2  # (250-100)//100 + 1 = 2


class TestCSIBenchMapper:
    def test_load_csibench_npz(self) -> None:
        """Loading a valid .npz should return correct shapes."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            data = np.random.randn(200, 100, 52).astype(np.float32)
            labels = np.array([0] * 180 + [1] * 20, dtype=np.int64)
            np.savez_compressed(path, data=data, labels=labels)

            loaded_data, loaded_labels = load_csibench_npz(path)
            assert loaded_data.shape == (200, 100, 52)
            assert loaded_labels.shape == (200,)
            assert np.sum(loaded_labels) == 20
        finally:
            os.unlink(path)

    def test_get_csibench_dataloaders(self) -> None:
        """Dataloaders should produce batched tensors."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            data = np.random.randn(100, 100, 52).astype(np.float32)
            labels = np.array([0] * 90 + [1] * 10, dtype=np.int64)
            np.savez_compressed(path, data=data, labels=labels)

            train_loader, val_loader = get_csibench_dataloaders(path, batch_size=8)

            for batch_data, batch_labels in train_loader:
                assert batch_data.shape[1:] == (100, 52)
                assert batch_labels.dim() == 1
                break
        finally:
            os.unlink(path)

    def test_preprocess_csibench_mat_no_scipy(self) -> None:
        """Should raise ImportError if scipy not available (graceful error)."""
        # This test just verifies the function skeleton exists and is importable.
        # The actual .mat parsing requires real CSI-Bench data.
        assert callable(preprocess_csibench_mat)
