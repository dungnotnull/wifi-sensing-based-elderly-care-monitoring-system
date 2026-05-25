"""
Vital Signs Estimation

Respiration rate (FFT-based, primary) and heart rate (experimental).
Includes a lightweight 1D-CNN denoiser for phase signal enhancement.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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
        # x: (batch, n_subcarriers, T)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x + decoded  # residual connection


class RespirationEstimator:
    """FFT-based respiration rate detector.

    Input: 30-second CSI phase segment (per subcarrier)
    Output: Respiration rate in breaths per minute (BPM)
    """

    def __init__(
        self,
        sample_rate: float = 50.0,
        window_seconds: float = 30.0,
        bandpass_low: float = 0.1,
        bandpass_high: float = 0.5,
        n_subcarriers: int = 52,
        top_k_subcarriers: int = 10,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.bandpass_low = bandpass_low    # 0.1 Hz = 6 BPM
        self.bandpass_high = bandpass_high  # 0.5 Hz = 30 BPM
        self.n_subcarriers = n_subcarriers
        self.top_k = top_k_subcarriers

    def estimate(self, phase_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Args:
            phase_matrix: shape (N_time, n_subcarriers), sanitized phase data

        Returns:
            (respiration_bpm, confidence)
        """
        if phase_matrix.shape[0] < self.window_samples // 2:
            return 0.0, 0.0

        # Use most recent window_samples frames
        if phase_matrix.shape[0] > self.window_samples:
            phase_matrix = phase_matrix[-self.window_samples:]

        n_time = phase_matrix.shape[0]

        # Select top-K subcarriers by signal variance
        variances = np.var(phase_matrix, axis=0)
        top_indices = np.argsort(variances)[-self.top_k:]

        # Average selected subcarriers
        phase_signal = np.mean(phase_matrix[:, top_indices], axis=1)

        # FFT
        fft = np.fft.rfft(phase_signal)
        freqs = np.fft.rfftfreq(n_time, d=1.0 / self.sample_rate)
        power = np.abs(fft) ** 2

        # Bandpass mask
        mask = (freqs >= self.bandpass_low) & (freqs <= self.bandpass_high)
        if not np.any(mask):
            return 0.0, 0.0

        band_freqs = freqs[mask]
        band_power = power[mask]

        # Dominant frequency
        peak_idx = np.argmax(band_power)
        peak_freq = band_freqs[peak_idx]
        bpm = peak_freq * 60.0

        # Simple confidence: ratio of peak power to mean band power
        mean_power = np.mean(band_power)
        confidence = min(band_power[peak_idx] / (mean_power + 1e-10) / 5.0, 1.0)

        return bpm, confidence


class HeartRateEstimator:
    """FFT-based heart rate detector (experimental).

    Operates in 0.8–2.0 Hz band. Accuracy limited by ESP32 hardware.
    """

    def __init__(
        self,
        sample_rate: float = 50.0,
        window_seconds: float = 30.0,
        bandpass_low: float = 0.8,
        bandpass_high: float = 2.0,
        n_subcarriers: int = 52,
        top_k_subcarriers: int = 10,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.bandpass_low = bandpass_low    # 0.8 Hz = 48 BPM
        self.bandpass_high = bandpass_high  # 2.0 Hz = 120 BPM
        self.n_subcarriers = n_subcarriers
        self.top_k = top_k_subcarriers

    def estimate(self, phase_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Args:
            phase_matrix: shape (N_time, n_subcarriers), sanitized phase data

        Returns:
            (heart_rate_bpm, confidence)
        """
        if phase_matrix.shape[0] < self.window_samples // 2:
            return 0.0, 0.0

        if phase_matrix.shape[0] > self.window_samples:
            phase_matrix = phase_matrix[-self.window_samples:]

        n_time = phase_matrix.shape[0]

        variances = np.var(phase_matrix, axis=0)
        top_indices = np.argsort(variances)[-self.top_k:]

        phase_signal = np.mean(phase_matrix[:, top_indices], axis=1)

        fft = np.fft.rfft(phase_signal)
        freqs = np.fft.rfftfreq(n_time, d=1.0 / self.sample_rate)
        power = np.abs(fft) ** 2

        mask = (freqs >= self.bandpass_low) & (freqs <= self.bandpass_high)
        if not np.any(mask):
            return 0.0, 0.0

        band_freqs = freqs[mask]
        band_power = power[mask]

        peak_idx = np.argmax(band_power)
        peak_freq = band_freqs[peak_idx]
        bpm = peak_freq * 60.0

        mean_power = np.mean(band_power)
        confidence = min(band_power[peak_idx] / (mean_power + 1e-10) / 3.0, 1.0)

        return bpm, confidence
