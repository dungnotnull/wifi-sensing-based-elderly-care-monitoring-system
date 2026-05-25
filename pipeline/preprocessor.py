"""
CSI Preprocessing Pipeline

Signal processing chain for raw CSI data:
  1. Hampel filter -- outlier removal (vectorized across all subcarriers)
  2. Butterworth bandpass filter (vectorized with axis=0)
  3. Phase sanitization -- unwrapping + linear detrend (vectorized)
  4. Amplitude normalization -- per-subcarrier z-score
"""

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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

    # ------------------------------------------------------------------
    # Vectorized Hampel filter
    # ------------------------------------------------------------------

    def _hampel_filter_matrix(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Hampel filter applied to the full (N_time, N_subcarriers) matrix.

        Uses sliding_window_view to extract exact sliding windows for all
        subcarriers simultaneously, then computes per-window median and MAD
        to identify and replace outliers -- matching the scalar implementation
        exactly.

        Args:
            x: input matrix of shape (N_time, N_subcarriers)

        Returns:
            Filtered matrix with outliers replaced by local medians.
        """
        n_time, n_sub = x.shape
        window_size = self.hampel_window
        k = window_size // 2

        # Not enough samples for a meaningful window
        if n_time <= window_size:
            return x.copy()

        filtered = x.copy()

        # Extract sliding windows along the time axis.
        # Result shape: (n_time - window_size + 1, n_sub, window_size)
        windows = sliding_window_view(x, window_shape=window_size, axis=0)

        # Per-window median and MAD across the window dimension (axis=2).
        # Both have shape (n_time - window_size + 1, n_sub).
        window_medians = np.median(windows, axis=2)
        window_mads = np.median(np.abs(windows - window_medians[:, :, np.newaxis]), axis=2)
        window_mads = np.maximum(window_mads, 1e-10)

        # The sliding windows start at index 0 and end at n_time - window_size.
        # In the scalar loop, index i corresponds to the window centered at i,
        # which starts at i - k. So the scalar range is [k, n_time - k).
        # The sliding windows cover indices [k, n_time - k) because
        # n_windows = n_time - window_size + 1 = n_time - 2k.
        # Window j (0-indexed) corresponds to scalar index i = j + k.
        interior_slice = slice(k, n_time - k)

        # Original values at interior points
        interior_vals = x[interior_slice, :]

        # Outlier detection at interior points
        outlier_mask = np.abs(interior_vals - window_medians) > self.hampel_threshold * window_mads

        # Replace outliers with the window median
        filtered[interior_slice, :] = np.where(
            outlier_mask,
            window_medians,
            interior_vals,
        )

        return filtered

    # ------------------------------------------------------------------
    # Scalar Hampel filter (kept for backward compatibility / testing)
    # ------------------------------------------------------------------

    def _hampel_filter(self, x: np.ndarray) -> np.ndarray:
        """Apply Hampel filter for impulse noise removal (scalar, per-subcarrier).

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

    # ------------------------------------------------------------------
    # Vectorized Butterworth filter
    # ------------------------------------------------------------------

    def _butterworth_filter(self, x: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter.

        Accepts either 1-D (single subcarrier) or 2-D (full matrix).
        For 2-D input, filtfilt is applied along axis=0 to filter all
        subcarriers simultaneously.
        """
        if x.ndim == 1:
            if len(x) < 15:
                return x
            return filtfilt(self._b, self._a, x)
        else:
            if x.shape[0] < 15:
                return x
            return filtfilt(self._b, self._a, x, axis=0)

    # ------------------------------------------------------------------
    # Scalar Butterworth filter (alias for backward compatibility)
    # ------------------------------------------------------------------

    def _butterworth_filter_scalar(self, x: np.ndarray) -> np.ndarray:
        """Apply Butterworth bandpass filter to a single 1-D signal."""
        if len(x) < 15:
            return x
        return filtfilt(self._b, self._a, x)

    # ------------------------------------------------------------------
    # Vectorized phase sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_phase(phase_matrix: np.ndarray) -> np.ndarray:
        """Phase unwrapping + linear detrend over time (vectorized).

        Uses np.unwrap with axis=0 to unwrap all subcarriers simultaneously,
        then performs vectorized linear detrending per column.

        Args:
            phase_matrix: shape (N_time, n_subcarriers)

        Returns:
            Sanitized phase matrix, same shape.
        """
        if phase_matrix.shape[0] < 2:
            return phase_matrix

        # Unwrap all columns at once along the time axis
        unwrapped = np.unwrap(phase_matrix, axis=0)

        # Vectorized linear detrend per column.
        # polyfit with deg=1 along each column, then subtract the trend.
        n_time = unwrapped.shape[0]
        t = np.arange(n_time, dtype=np.float64)

        # Compute linear fit coefficients for every column simultaneously.
        # np.polyfit does not support an axis parameter, so we use the
        # closed-form solution for least-squares linear fit:
        #   slope = (n*sum(t*y) - sum(t)*sum(y)) / (n*sum(t^2) - sum(t)^2)
        #   intercept = mean(y) - slope * mean(t)
        n = float(n_time)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()

        y_means = unwrapped.mean(axis=0)
        # covariance of t and each column
        ty_cov = ((t[:, np.newaxis] - t_mean) * (unwrapped - y_means[np.newaxis, :])).sum(axis=0)
        slopes = ty_cov / t_var

        # Subtract the linear trend: detrended = y - (slope * t + intercept)
        trend = slopes[np.newaxis, :] * t[:, np.newaxis] + (y_means - slopes * t_mean)[np.newaxis, :]
        sanitized = unwrapped - trend

        return sanitized

    # ------------------------------------------------------------------
    # Scalar phase sanitization (kept for backward compatibility / testing)
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_phase_scalar(phase_matrix: np.ndarray) -> np.ndarray:
        """Phase unwrapping + linear detrend over time (scalar, per-column loop).

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
        """Run the full preprocessing pipeline (vectorized).

        Args:
            amplitude: shape (N_time, n_subcarriers)
            phase: shape (N_time, n_subcarriers)
            remove_edges: whether to strip edge subcarriers

        Returns:
            (processed_amplitude, processed_phase) -- same shape (or trimmed if remove_edges=True)
        """
        # 1. Hampel filter -- vectorized across all subcarriers
        amplitude = self._hampel_filter_matrix(amplitude)

        # 2. Butterworth bandpass -- vectorized with axis=0
        amplitude = self._butterworth_filter(amplitude)

        # 3. Phase sanitization -- vectorized
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
