"""Tests for CSI preprocessing pipeline."""
import numpy as np
import pytest
from pipeline.preprocessor import Preprocessor


class TestPreprocessor:
    def test_hampel_filter_removes_outliers(self) -> None:
        preprocessor = Preprocessor(n_subcarriers=1, hampel_window=7, hampel_threshold=3.0)
        x = np.ones(100)
        x[50] = 100.0  # spike
        filtered = preprocessor._hampel_filter(x)
        assert abs(filtered[50] - 1.0) < 0.1  # spike should be removed

    def test_sanitize_phase(self) -> None:
        n_time, n_sub = 100, 5
        phase = np.cumsum(np.random.randn(n_time, n_sub) * 0.1, axis=0)
        sanitized = Preprocessor.sanitize_phase(phase)
        assert sanitized.shape == phase.shape
        # Sanitized phase should have near-zero mean after detrend
        assert np.abs(np.mean(sanitized)) < 1.0

    def test_normalize_amplitude_zero_mean_unit_variance(self) -> None:
        preprocessor = Preprocessor(n_subcarriers=5)
        amp = np.random.randn(200, 5) * 0.5 + 2.0  # mean=2.0, std=0.5
        normalized = preprocessor.normalize_amplitude(amp)
        # After normalization, columns should be roughly zero-mean, unit-variance
        col_means = np.mean(normalized, axis=0)
        col_stds = np.std(normalized, axis=0)
        assert np.all(np.abs(col_means) < 1.0)
        assert np.all(np.abs(col_stds - 1.0) < 0.5)

    def test_remove_edge_subcarriers(self) -> None:
        preprocessor = Preprocessor(n_subcarriers=10)
        matrix = np.ones((50, 10))
        trimmed = preprocessor.remove_edge_subcarriers(matrix, n_remove=2)
        assert trimmed.shape == (50, 6)

    def test_process_returns_correct_shapes(self) -> None:
        preprocessor = Preprocessor(n_subcarriers=52)
        amp = np.random.randn(100, 52) * 0.1 + 1.0
        phase = np.cumsum(np.random.randn(100, 52) * 0.05, axis=0)
        proc_amp, proc_phase = preprocessor.process(amp, phase, remove_edges=True)
        # After removing 2 edge subcarriers: 52 - 2*2 = 48
        assert proc_amp.shape == (100, 48)
        assert proc_phase.shape == (100, 48)
