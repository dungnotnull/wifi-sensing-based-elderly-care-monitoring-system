"""Vital signs estimation using wifi_densepose Rust-native extractors.

BreathingExtractor (0.1-0.5 Hz bandpass + zero-crossing) and
HeartRateExtractor (0.8-2.0 Hz bandpass + autocorrelation) from
the wifi_densepose package provide battle-tested RuView algorithms
via PyO3 bindings. We keep the PhaseDenoiser as a supplementary
enhancement for cleaner phase signals.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from wifi_densepose import BreathingExtractor, HeartRateExtractor

logger = logging.getLogger(__name__)


class PhaseDenoiser(nn.Module):
    """Lightweight 1D-CNN denoiser for CSI phase signals.

    Pre-trained on synthetic CSI + vitals data. Optional; can be
    disabled on RPi5 if throughput is a bottleneck.
    """

    def __init__(self, n_subcarriers: int = 52, denoising_kernel: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=denoising_kernel, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=denoising_kernel, padding="same"),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=denoising_kernel, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, n_subcarriers, kernel_size=denoising_kernel, padding="same"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x + decoded


class VitalsAdapter:
    """Stateful adapter wrapping wifi_densepose BreathingExtractor + HeartRateExtractor.

    Feed one preprocessed amplitude frame at a time via feed_frame().
    The Rust extractors maintain their own internal circular buffers
    and return estimates when sufficient history is accumulated.
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        sample_rate: float = 50.0,
        respiration_window_secs: float = 30.0,
        heart_rate_window_secs: float = 15.0,
    ) -> None:
        self.n_subcarriers = n_subcarriers
        self.sample_rate = sample_rate
        self._br = BreathingExtractor(
            n_subcarriers=n_subcarriers,
            sample_rate=sample_rate,
            window_secs=respiration_window_secs,
        )
        self._hr = HeartRateExtractor(
            n_subcarriers=n_subcarriers,
            sample_rate=sample_rate,
            window_secs=heart_rate_window_secs,
        )

        self._last_br_est: Optional[float] = None
        self._last_br_conf: Optional[float] = None
        self._last_hr_est: Optional[float] = None
        self._last_hr_conf: Optional[float] = None

    def feed_frame(self, amplitude: np.ndarray) -> None:
        """Feed one per-subcarrier amplitude frame (shape: n_subcarriers,)."""
        residuals = [float(v) for v in amplitude]
        weights = [1.0 / self.n_subcarriers] * self.n_subcarriers

        try:
            br_est = self._br.extract(residuals, weights)
            if br_est is not None:
                self._last_br_est = br_est.value_bpm
                self._last_br_conf = br_est.confidence

            hr_est = self._hr.extract(residuals, weights)
            if hr_est is not None:
                self._last_hr_est = hr_est.value_bpm
                self._last_hr_conf = hr_est.confidence
        except Exception:
            logger.exception("Error in wifi_densepose extractor")

    @property
    def respiration_bpm(self) -> Optional[float]:
        return self._last_br_est

    @property
    def respiration_confidence(self) -> Optional[float]:
        return self._last_br_conf

    @property
    def heart_rate_bpm(self) -> Optional[float]:
        return self._last_hr_est

    @property
    def heart_rate_confidence(self) -> Optional[float]:
        return self._last_hr_conf
