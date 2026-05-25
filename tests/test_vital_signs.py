"""Tests for vital signs estimation."""
import numpy as np
import pytest
from models.vital_signs.estimator import RespirationEstimator, HeartRateEstimator


class TestRespirationEstimator:
    def test_returns_valid_bpm_range(self) -> None:
        estimator = RespirationEstimator()
        # Generate a sinusoidal phase signal at 0.25 Hz (15 BPM)
        sample_rate = 50.0
        duration = 35.0
        t = np.arange(0, duration, 1.0 / sample_rate)
        signal = np.sin(2 * np.pi * 0.25 * t)[:, np.newaxis]
        phase = np.tile(signal, (1, 52))

        bpm, conf = estimator.estimate(phase)
        assert 10 <= bpm <= 20  # should be near 15 BPM
        assert conf > 0.3

    def test_insufficient_data_returns_zero(self) -> None:
        estimator = RespirationEstimator()
        phase = np.random.randn(10, 52)
        bpm, conf = estimator.estimate(phase)
        assert bpm == 0.0
        assert conf == 0.0


class TestHeartRateEstimator:
    def test_returns_valid_range(self) -> None:
        estimator = HeartRateEstimator()
        sample_rate = 50.0
        duration = 35.0
        t = np.arange(0, duration, 1.0 / sample_rate)
        signal = np.sin(2 * np.pi * 1.2 * t)[:, np.newaxis]  # 1.2 Hz = 72 BPM
        phase = np.tile(signal, (1, 52))

        bpm, conf = estimator.estimate(phase)
        assert 50 <= bpm <= 100
