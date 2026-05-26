"""
Multi-Person Occupancy Detection

Detects when multiple people are present in a zone by analyzing
CSI amplitude variance patterns across subcarriers.

Single-person movement produces localized variance (a few dominant
subcarrier clusters). Multi-person scenarios produce diffuse,
high-variance patterns across many subcarriers simultaneously.

When multi-person detected:
  - Fall/vitals/activity alerts are suppressed
  - Dashboard shows "Multiple occupants - monitoring suspended"
  - Auto-resumes when variance returns to single-person patterns

Detection uses two metrics:
  1. Subcarrier spread ratio — ratio of active subcarriers to total
  2. Variance entropy — how uniformly variance is distributed
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OccupancyState:
    UNKNOWN = "unknown"
    SINGLE = "single"
    MULTI = "multi"


@dataclass
class OccupancyResult:
    state: str
    spread_ratio: float
    variance_entropy: float
    confidence: float
    timestamp: float


class MultiPersonDetector:
    """Detects multi-person occupancy from CSI variance patterns.

    Single person: variance concentrated on 10-40% of subcarriers
    Multi-person:  variance spread across 50%+ of subcarriers,
                    higher Shannon entropy of variance distribution
    """

    def __init__(
        self,
        window_frames: int = 250,       # 5 seconds at 50Hz
        spread_threshold: float = 0.45,  # subcarrier activation ratio
        entropy_threshold: float = 2.5,  # Shannon entropy floor for multi
        confirmation_frames: int = 50,    # consecutive frames to confirm state change
        cooldown_seconds: float = 30.0,   # min time before reverting to single
    ) -> None:
        self.window_frames = window_frames
        self.spread_threshold = spread_threshold
        self.entropy_threshold = entropy_threshold
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds

        self._state = OccupancyState.UNKNOWN
        self._state_start: float = time.time()
        self._consecutive_same: int = 0
        self._last_predicted: str = OccupancyState.UNKNOWN

        # Rolling window for temporal smoothing
        self._spread_history: deque[float] = deque(maxlen=10)
        self._entropy_history: deque[float] = deque(maxlen=10)

    def analyze(self, csi_amplitude: np.ndarray) -> OccupancyResult:
        """Analyze a window of CSI amplitude data for occupancy classification.

        Args:
            csi_amplitude: shape (N_time, n_subcarriers)

        Returns:
            OccupancyResult with current classification and metrics
        """
        if csi_amplitude.shape[0] < 2:
            return OccupancyResult(
                state=self._state,
                spread_ratio=0.0,
                variance_entropy=0.0,
                confidence=0.0,
                timestamp=time.time(),
            )

        n_subcarriers = csi_amplitude.shape[1]

        # Per-subcarrier variance over time
        sc_variance = np.var(np.abs(csi_amplitude), axis=0)  # (n_subcarriers,)

        # Metric 1: Subcarrier spread ratio
        # Active subcarriers = those with variance above 10% of max
        variance_threshold = max(np.max(sc_variance) * 0.10, 1e-6)
        active_subcarriers = np.sum(sc_variance > variance_threshold)
        spread_ratio = active_subcarriers / n_subcarriers

        # Metric 2: Shannon entropy of normalized variance distribution
        total_variance = np.sum(sc_variance)
        if total_variance > 1e-10:
            normed = sc_variance / total_variance
            # Add small epsilon to avoid log(0)
            entropy = -np.sum(normed * np.log(normed + 1e-10))
        else:
            entropy = 0.0

        # Smooth metrics over recent frames
        self._spread_history.append(spread_ratio)
        self._entropy_history.append(entropy)
        smooth_spread = np.mean(self._spread_history)
        smooth_entropy = np.mean(self._entropy_history)

        # Classification
        if smooth_spread >= self.spread_threshold and smooth_entropy >= self.entropy_threshold:
            predicted = OccupancyState.MULTI
        else:
            predicted = OccupancyState.SINGLE

        # Temporal confirmation (debounce)
        if predicted == self._last_predicted:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 1
            self._last_predicted = predicted

        now = time.time()

        # Apply cooldown when transitioning from multi -> single
        if self._state == OccupancyState.MULTI and predicted == OccupancyState.SINGLE:
            if (now - self._state_start) < self.cooldown_seconds:
                predicted = OccupancyState.MULTI  # hold multi state during cooldown

        # Confirm state change
        if self._consecutive_same >= self.confirmation_frames:
            if predicted != self._state:
                logger.info(
                    f"Occupancy changed: {self._state} -> {predicted} "
                    f"(spread={smooth_spread:.3f}, entropy={smooth_entropy:.2f})"
                )
                self._state = predicted
                self._state_start = now
                self._consecutive_same = self.confirmation_frames

        # Confidence: how far past thresholds
        if self._state == OccupancyState.MULTI:
            confidence = min(1.0, (smooth_spread - self.spread_threshold) / 0.3 +
                                   (smooth_entropy - self.entropy_threshold) / 2.0)
        else:
            confidence = min(1.0, (self.spread_threshold - smooth_spread) / 0.3 +
                                   (self.entropy_threshold - smooth_entropy) / 2.0)
        confidence = max(0.0, confidence)

        return OccupancyResult(
            state=self._state,
            spread_ratio=round(smooth_spread, 4),
            variance_entropy=round(smooth_entropy, 2),
            confidence=round(confidence, 3),
            timestamp=now,
        )

    def should_suppress_alerts(self) -> bool:
        """True when alerts should be suppressed (multi-person detected)."""
        return self._state == OccupancyState.MULTI

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        self._state = OccupancyState.UNKNOWN
        self._state_start = time.time()
        self._consecutive_same = 0
        self._spread_history.clear()
        self._entropy_history.clear()
