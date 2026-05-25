"""Pure-Python vital signs extractors using scipy/numpy.

Drop-in replacement for wifi_densepose Rust extractors when the native
package is unavailable. Provides the same feed-per-frame interface with
FFT-based respiration estimation and autocorrelation-based heart rate
estimation.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


@dataclass
class VitalEstimate:
    """Single vital-sign measurement from the Python fallback."""

    value_bpm: float
    confidence: float


class PythonBreathingExtractor:
    """Estimate respiration rate from CSI amplitude residuals.

    Accumulates per-frame amplitude in a circular buffer, then on each
    extraction call:
      1. Compute residuals (amplitude - running mean across subcarriers).
      2. Bandpass filter 0.1-0.5 Hz (6-30 breaths/min).
      3. Run FFT and find the peak frequency.
      4. Convert peak frequency to BPM and estimate confidence from SNR.

    Typical breathing range: 6-30 breaths/min (0.1-0.5 Hz).
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        sample_rate: float = 50.0,
        window_secs: float = 30.0,
    ) -> None:
        self._n_subcarriers = n_subcarriers
        self._sample_rate = sample_rate
        self._window_size = int(window_secs * sample_rate)

        self._buffer: deque = deque(maxlen=self._window_size)
        self._running_mean = np.zeros(n_subcarriers, dtype=np.float64)

        # Pre-compute bandpass filter coefficients (0.1-0.5 Hz)
        nyquist = sample_rate / 2.0
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        # Guard against invalid normalized frequencies
        high = min(high, 0.99)
        self._b, self._a = butter(2, [low, high], btype="band")

    def feed(self, residuals: list[float], weights: list[float]) -> None:
        """Store one frame of per-subcarrier amplitude residuals."""
        weighted = np.array(residuals, dtype=np.float64) * np.array(weights, dtype=np.float64)
        # Exponential moving average for running mean
        alpha = 0.01
        self._running_mean = (1 - alpha) * self._running_mean + alpha * np.array(residuals, dtype=np.float64)
        # Store the weighted sum as a single scalar per frame
        self._buffer.append(float(np.sum(weighted)))

    def extract(self, residuals: list[float], weights: list[float]) -> Optional[VitalEstimate]:
        """Feed one frame and attempt extraction. Returns None if buffer is insufficient."""
        self.feed(residuals, weights)

        if len(self._buffer) < self._sample_rate * 2:
            # Need at least 2 seconds of data
            return None

        signal = np.array(self._buffer, dtype=np.float64)

        # Remove DC component
        signal = signal - np.mean(signal)

        # Bandpass filter (0.1-0.5 Hz for breathing)
        try:
            padlen = min(3 * max(len(self._b), len(self._a)) - 1, len(signal) - 1)
            filtered = filtfilt(self._b, self._a, signal, padlen=padlen)
        except ValueError:
            # filtfilt can fail on very short or degenerate signals
            return None

        # FFT-based frequency estimation
        n = len(filtered)
        fft_vals = np.fft.rfft(filtered)
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / self._sample_rate)
        fft_magnitude = np.abs(fft_vals)

        # Restrict search to breathing band
        mask = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
        if not np.any(mask):
            return None

        magnitudes_in_band = fft_magnitude[mask]
        freqs_in_band = fft_freqs[mask]

        peak_idx = np.argmax(magnitudes_in_band)
        peak_freq = freqs_in_band[peak_idx]
        peak_magnitude = magnitudes_in_band[peak_idx]

        # Confidence from SNR: ratio of peak to mean of non-peak magnitudes
        mean_magnitude = np.mean(magnitudes_in_band)
        if mean_magnitude < 1e-10:
            return None
        snr = peak_magnitude / mean_magnitude
        confidence = float(np.clip((snr - 1.0) / 4.0, 0.0, 1.0))

        bpm = peak_freq * 60.0  # Convert Hz to breaths-per-minute

        # Sanity check: breathing must be in physiological range
        if bpm < 6.0 or bpm > 30.0:
            return None

        return VitalEstimate(value_bpm=bpm, confidence=confidence)


class PythonHeartRateExtractor:
    """Estimate heart rate from CSI amplitude residuals using autocorrelation.

    Heart rate estimation from CSI is experimental. Uses autocorrelation
    on a bandpass-filtered signal (0.8-2.0 Hz, i.e. 48-120 BPM) to find
    the dominant periodicity.

    Confidence is capped lower than breathing to reflect the experimental
    nature of CSI-based heart rate estimation.
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        sample_rate: float = 50.0,
        window_secs: float = 15.0,
    ) -> None:
        self._n_subcarriers = n_subcarriers
        self._sample_rate = sample_rate
        self._window_size = int(window_secs * sample_rate)

        self._buffer: deque = deque(maxlen=self._window_size)
        self._running_mean = np.zeros(n_subcarriers, dtype=np.float64)

        # Bandpass filter coefficients (0.8-2.0 Hz for heart rate)
        nyquist = sample_rate / 2.0
        low = 0.8 / nyquist
        high = 2.0 / nyquist
        high = min(high, 0.99)
        low = max(low, 0.01)
        self._b, self._a = butter(3, [low, high], btype="band")

    def feed(self, residuals: list[float], weights: list[float]) -> None:
        """Store one frame of per-subcarrier amplitude residuals."""
        weighted = np.array(residuals, dtype=np.float64) * np.array(weights, dtype=np.float64)
        alpha = 0.01
        self._running_mean = (1 - alpha) * self._running_mean + alpha * np.array(residuals, dtype=np.float64)
        self._buffer.append(float(np.sum(weighted)))

    def extract(self, residuals: list[float], weights: list[float]) -> Optional[VitalEstimate]:
        """Feed one frame and attempt extraction. Returns None if buffer is insufficient."""
        self.feed(residuals, weights)

        # Heart rate needs more data for reliable autocorrelation
        if len(self._buffer) < self._sample_rate * 5:
            return None

        signal = np.array(self._buffer, dtype=np.float64)
        signal = signal - np.mean(signal)

        # Bandpass filter (0.8-2.0 Hz)
        try:
            padlen = min(3 * max(len(self._b), len(self._a)) - 1, len(signal) - 1)
            filtered = filtfilt(self._b, self._a, signal, padlen=padlen)
        except ValueError:
            return None

        # Autocorrelation-based peak detection
        n = len(filtered)
        # Compute autocorrelation using FFT for efficiency
        fft_signal = np.fft.fft(filtered, n=2 * n)
        autocorr = np.fft.ifft(fft_signal * np.conj(fft_signal)).real[:n]
        # Normalize
        if autocorr[0] < 1e-10:
            return None
        autocorr = autocorr / autocorr[0]

        # Search for first peak in the heart-rate lag range
        min_lag = int(self._sample_rate / 2.0)   # 2.0 Hz = 120 BPM
        max_lag = int(self._sample_rate / 0.8)   # 0.8 Hz = 48 BPM
        max_lag = min(max_lag, n - 1)

        if min_lag >= max_lag:
            return None

        autocorr_segment = autocorr[min_lag:max_lag]
        if len(autocorr_segment) == 0:
            return None

        peak_idx = np.argmax(autocorr_segment)
        best_lag = min_lag + peak_idx
        peak_value = autocorr_segment[peak_idx]

        # Convert lag to frequency to BPM
        bpm = (self._sample_rate / best_lag) * 60.0

        # Confidence: autocorrelation peak strength, capped to reflect
        # the experimental nature of CSI heart rate estimation
        confidence = float(np.clip(peak_value * 0.6, 0.0, 0.7))

        # Sanity check: heart rate must be in physiological range
        if bpm < 48.0 or bpm > 120.0:
            return None

        return VitalEstimate(value_bpm=bpm, confidence=confidence)
