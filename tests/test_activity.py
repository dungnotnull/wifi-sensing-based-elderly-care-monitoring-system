"""Tests for activity detector."""
import numpy as np
import pytest
from models.activity.detector import ActivityDetector, ActivityState


class TestActivityDetector:
    def test_active_detection(self) -> None:
        detector = ActivityDetector(threshold_active=0.5)
        # High per-frame amplitude variance across 500 frames
        rng = np.random.RandomState(42)
        per_frame = rng.randn(500) * 2.0  # each frame very different
        amp = np.tile(per_frame[:, np.newaxis], (1, 52))
        state = detector.classify(amp)
        assert state == ActivityState.ACTIVE

    def test_still_detection(self) -> None:
        detector = ActivityDetector(threshold_active=0.5, threshold_still=0.15)
        rng = np.random.RandomState(42)
        per_frame = rng.randn(500) * 0.7 + 1.0  # variance ~0.49 per column
        amp = np.tile(per_frame[:, np.newaxis], (1, 52))
        state = detector.classify(amp)
        assert state == ActivityState.STILL

    def test_inactivity_detection(self) -> None:
        detector = ActivityDetector(threshold_still=0.15)
        amp = np.ones((500, 52)) * 1.0  # zero variance
        state = detector.classify(amp)
        assert state == ActivityState.INACTIVITY

    def test_is_daytime(self) -> None:
        detector = ActivityDetector(daytime_start_hour=6, daytime_end_hour=22)
        assert detector.is_daytime(7.0) is True
        assert detector.is_daytime(14.0) is True
        assert detector.is_daytime(3.0) is False
        assert detector.is_daytime(23.0) is False
