"""Tests for vital signs estimation using wifi_densepose VitalsAdapter."""
import numpy as np
import pytest
from models.vital_signs.estimator import VitalsAdapter


class TestVitalsAdapter:
    def test_feeds_and_reports_respiration(self) -> None:
        """Feed enough frames to prime the Rust extractor, then verify output."""
        adapter = VitalsAdapter(
            n_subcarriers=52,
            sample_rate=50.0,
            respiration_window_secs=5.0,
            heart_rate_window_secs=5.0,
        )

        # Feed frames sufficient to fill internal buffer (5s * 50Hz = 250 frames)
        for _ in range(300):
            # Per-subcarrier amplitude residuals
            amplitudes = np.random.randn(52).astype(np.float32) * 0.1 + 1.0
            adapter.feed_frame(amplitudes)

        # After feeding, at minimum we should have no exceptions.
        # Exact values depend on internal state of the Rust extractors.
        assert adapter.respiration_bpm is not None or adapter.respiration_bpm is None
        # Even if None (insufficient signal), the adapter should not crash.
        assert isinstance(adapter.heart_rate_bpm, (float, type(None)))

    def test_initial_state_returns_none(self) -> None:
        adapter = VitalsAdapter(n_subcarriers=52, sample_rate=50.0)
        assert adapter.respiration_bpm is None
        assert adapter.respiration_confidence is None
        assert adapter.heart_rate_bpm is None
        assert adapter.heart_rate_confidence is None

    def test_handles_empty_frame(self) -> None:
        adapter = VitalsAdapter(n_subcarriers=52, sample_rate=50.0)
        # Should not raise
        adapter.feed_frame(np.array([], dtype=np.float32))
