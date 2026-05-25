"""
Activity / Inactivity Detection

Rule-based primary detector for MVP. Uses CSI amplitude variance
over a sliding window to classify: ACTIVE / STILL / INACTIVITY.

DL enhancement (post-MVP): Fine-grained activity classification
using ResNet-style 1D-CNN.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ActivityState:
    ACTIVE = "active"
    STILL = "still"
    INACTIVITY = "inactivity"


class ActivityDetector:
    """Rule-based activity detector using CSI amplitude variance."""

    def __init__(
        self,
        threshold_active: float = 0.5,
        threshold_still: float = 0.15,
        window_seconds: float = 30.0,
        sample_rate: float = 50.0,
        inactivity_timeout_seconds: float = 7200.0,  # 2 hours
        daytime_start_hour: int = 6,
        daytime_end_hour: int = 22,
    ) -> None:
        self.threshold_active = threshold_active
        self.threshold_still = threshold_still
        self.window_frames = int(window_seconds * sample_rate)
        self.inactivity_timeout = inactivity_timeout_seconds
        self.daytime_start = daytime_start_hour
        self.daytime_end = daytime_end_hour

        self._inactivity_start: Optional[float] = None  # timestamp
        self._state = ActivityState.ACTIVE
        self._last_state_change: float = 0.0

    def classify(self, csi_amplitude: np.ndarray) -> str:
        """Classify activity state from recent CSI amplitude data.

        Args:
            csi_amplitude: shape (N_time, n_subcarriers)

        Returns:
            One of: "active", "still", "inactivity"
        """
        if csi_amplitude.shape[0] == 0:
            return self._state

        # Compute mean amplitude variance across subcarriers
        per_frame_amp = np.mean(np.abs(csi_amplitude), axis=1)
        variance = np.var(per_frame_amp)

        if variance > self.threshold_active:
            return ActivityState.ACTIVE
        elif variance > self.threshold_still:
            return ActivityState.STILL
        else:
            return ActivityState.INACTIVITY

    def update(self, csi_amplitude: np.ndarray, timestamp_hour: float) -> Tuple[str, Optional[str]]:
        """Update state and check for alert conditions.

        Args:
            csi_amplitude: recent CSI amplitude frames
            timestamp_hour: current hour in local time (0–24 float)

        Returns:
            (current_state, alert_level_or_none)
        """
        new_state = self.classify(csi_amplitude)

        alert = None

        if new_state == ActivityState.ACTIVE:
            self._inactivity_start = None
        elif new_state == ActivityState.INACTIVITY:
            if self._inactivity_start is None:
                self._inactivity_start = 0.0

            # Check if it's daytime (inactivity during sleep hours is normal)
            is_daytime = self.daytime_start <= timestamp_hour < self.daytime_end

            if is_daytime and new_state == ActivityState.INACTIVITY:
                # Inactivity timeout exceeded
                alert = "WARNING"

        self._state = new_state
        return new_state, alert

    def is_prolonged_inactivity(self, inactivity_duration_seconds: float) -> bool:
        """Check if inactivity has exceeded the timeout threshold."""
        return inactivity_duration_seconds >= self.inactivity_timeout

    def is_daytime(self, timestamp_hour: float) -> bool:
        return self.daytime_start <= timestamp_hour < self.daytime_end

    def reset(self) -> None:
        self._inactivity_start = None
        self._state = ActivityState.ACTIVE


class PostFallInactivityChecker:
    """Checks for inactivity following a fall event.

    If no recovery movement within 30 seconds → EMERGENCY alert.
    """

    def __init__(self, recovery_timeout_seconds: float = 30.0) -> None:
        self.recovery_timeout = recovery_timeout_seconds
        self._fall_time: Optional[float] = None
        self._triggered: bool = False

    def on_fall_detected(self, timestamp: float) -> None:
        self._fall_time = timestamp
        self._triggered = False

    def check(self, csi_amplitude: np.ndarray, current_time: float) -> Optional[str]:
        """Check if post-fall inactivity alert should fire.

        Returns:
            "EMERGENCY" if no recovery within timeout, None otherwise.
        """
        if self._fall_time is None or self._triggered:
            return None

        elapsed = current_time - self._fall_time
        if elapsed < self.recovery_timeout:
            return None

        # Check if person is still inactive
        per_frame_amp = np.mean(np.abs(csi_amplitude), axis=1)
        variance = np.var(per_frame_amp)

        if variance < 0.15:  # same threshold as inactivity
            self._triggered = True
            return "EMERGENCY"

        # Person has recovered — reset
        self._fall_time = None
        return None

    def reset(self) -> None:
        self._fall_time = None
        self._triggered = False
