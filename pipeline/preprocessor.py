"""
CSI Preprocessing Pipeline

Signal processing chain for raw CSI data:
  1. Hampel filter — outlier removal
  2. Butterworth bandpass filter
  3. Phase sanitization — unwrapping + linear detrend
  4. Amplitude normalization — per-subcarrier z-score
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


class Preprocessor:
    """CSI signal preprocessing with configurable filter parameters."""

    def __init__(
        self,
        n_subcarriers: int = 52,
        hampel_window: int = 5,
        hampel_threshold: float = 3.0,
        bandpass_low: float = 0.1,
        bandpass_high: float = 20.0,
        sample_rate: float = 50.0,
    ) -> None:
        self.n_subcarriers = n_subcarriers
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold

        # Design Butterworth bandpass filter
        nyquist = sample_rate / 2.0
        low = bandpass_low / nyquist if bandpass_low > 0 else 0.01 / nyquist
        high = bandpass_high / nyquist if bandpass_high < nyquist else 0.99
        self._b, self._a = butter(N=4, Wn=[low, high], btype="band")

        # Running stats for z-score normalization (per subcarrier)
        self._amp_mean: np.ndarray = np.ones(n_subcarriers)
        self._amp_std: np.ndarray = np.ones(n_subcarriers)
        self._stats_decay: float = 0.99  # EMA decay for running stats
        self._stats_initialized: bool = False

    def _hampel_filter(self, x: np.ndarray) -> np.ndarray:
        """Apply Hampel filter for impulse noise removal.

        Replaces outliers (values > threshold * MAD from local median) with the local median.
        """
        k = self.hampel_window // 2
        n = len(x)
        filtered = np.copy(x)

        for i in range(k, n - k):
            window = x[i - k : i + k + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            mad = mad if mad > 1e-10 else 1e-10

            if abs(x[i] - median) > self.hampel_threshold * mad:
                filtered[i] = median

        return filtered

    def _butterworth_filter(self, x: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        if len(x) < 15:
            return x  # Not enough samples for meaningful filtering
        return filtfilt(self._b, self._a, x)

    @staticmethod
    def sanitize_phase(phase_matrix: np.ndarray) -> np.ndarray:
        """Phase unwrapping + linear detrend over time.

        Args:
            phase_matrix: shape (N_time, n_subcarriers)

        Returns:
            Sanitized phase matrix, same shape.
        """
        if phase_matrix.shape[0] < 2:
            return phase_matrix

        sanitized = np.zeros_like(phase_matrix)
        for i in range(phase_matrix.shape[1]):
            col = phase_matrix[:, i]
            unwrapped = np.unwrap(col)
            t = np.arange(len(unwrapped))
            trend = np.polyfit(t, unwrapped, 1)
            detrended = unwrapped - np.polyval(trend, t)
            sanitized[:, i] = detrended

        return sanitized

    def normalize_amplitude(self, amplitude_matrix: np.ndarray) -> np.ndarray:
        """Per-subcarrier z-score normalization with running statistics.

        Args:
            amplitude_matrix: shape (N_time, n_subcarriers)

        Returns:
            Normalized amplitude matrix, same shape.
        """
        col_means = np.mean(amplitude_matrix, axis=0)
        col_stds = np.std(amplitude_matrix, axis=0)
        col_stds = np.where(col_stds < 1e-10, 1.0, col_stds)

        if self._stats_initialized:
            self._amp_mean = self._stats_decay * self._amp_mean + (1 - self._stats_decay) * col_means
            self._amp_std = self._stats_decay * self._amp_std + (1 - self._stats_decay) * col_stds
        else:
            self._amp_mean = col_means
            self._amp_std = col_stds
            self._stats_initialized = True

        return (amplitude_matrix - self._amp_mean) / self._amp_std

    def remove_edge_subcarriers(self, matrix: np.ndarray, n_remove: int = 2) -> np.ndarray:
        """Remove edge subcarriers with poor SNR.

        Args:
            matrix: shape (N_time, n_subcarriers)
            n_remove: number of subcarriers to remove from each edge

        Returns:
            Trimmed matrix with n_subcarriers - 2*n_remove subcarriers.
        """
        return matrix[:, n_remove:-n_remove]

    def process(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        remove_edges: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full preprocessing pipeline.

        Args:
            amplitude: shape (N_time, n_subcarriers)
            phase: shape (N_time, n_subcarriers)
            remove_edges: whether to strip edge subcarriers

        Returns:
            (processed_amplitude, processed_phase) — same shape (or trimmed if remove_edges=True)
        """
        # 1. Hampel filter per subcarrier
        for i in range(self.n_subcarriers):
            amplitude[:, i] = self._hampel_filter(amplitude[:, i])

        # 2. Butterworth bandpass per subcarrier
        for i in range(self.n_subcarriers):
            amplitude[:, i] = self._butterworth_filter(amplitude[:, i])

        # 3. Phase sanitization
        phase = self.sanitize_phase(phase)

        # 4. Amplitude normalization
        amplitude = self.normalize_amplitude(amplitude)

        # 5. Remove noisy edge subcarriers
        if remove_edges:
            amplitude = self.remove_edge_subcarriers(amplitude)
            phase = self.remove_edge_subcarriers(phase)

        return amplitude, phase

    def process_from_ring_buffer(
        self, packets: list[dict], n_subcarriers: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method: extract amplitude/phase matrices from CSI packet list and preprocess.

        Returns:
            (amplitude_matrix, phase_matrix) of shape (N_packets, n_subcarriers)
        """
        if n_subcarriers is None:
            n_subcarriers = self.n_subcarriers

        n_packets = len(packets)
        amp = np.zeros((n_packets, n_subcarriers), dtype=np.float32)
        phase = np.zeros((n_packets, n_subcarriers), dtype=np.float32)

        for i, pkt in enumerate(packets):
            amp[i, :] = pkt["csi_amplitude"][:n_subcarriers]
            phase[i, :] = pkt["csi_phase"][:n_subcarriers]

        return self.process(amp, phase)
