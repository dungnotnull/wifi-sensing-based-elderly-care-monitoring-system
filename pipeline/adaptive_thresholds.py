"""Adaptive thresholds that self-tune from deployment data.

Replaces static thresholds in configs/thresholds.yaml with adaptive
values computed from rolling statistics of actual CSI data.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveThreshold:
    """A single threshold that adapts based on observed data distribution."""

    def __init__(
        self,
        initial_value: float,
        adaptation_rate: float = 0.01,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        window_size: int = 1000,
    ) -> None:
        self._value = initial_value
        self._initial = initial_value
        self._adaptation_rate = adaptation_rate
        self._min = min_value
        self._max = max_value
        self._samples = deque(maxlen=window_size)
        self._lock = threading.Lock()

    @property
    def value(self) -> float:
        return self._value

    def observe(self, sample: float) -> None:
        """Feed a new observation for adaptation."""
        with self._lock:
            self._samples.append(sample)

    def adapt(self) -> float:
        """Recompute threshold from accumulated samples.

        Returns the new threshold value.
        """
        with self._lock:
            if len(self._samples) < 100:
                return self._value

            samples = np.array(self._samples)
            proposed = np.percentile(samples, 75)

            # Smooth transition: don't jump too fast
            delta = proposed - self._value
            new_value = self._value + self._adaptation_rate * delta

            if self._min is not None:
                new_value = max(new_value, self._min)
            if self._max is not None:
                new_value = min(new_value, self._max)

            self._value = new_value
            return self._value

    def reset(self) -> None:
        self._value = self._initial
        self._samples.clear()


class ActivityThresholdManager:
    """Manages adaptive thresholds for activity detection per zone.

    Learns baseline noise level during the first hours of deployment,
    then adjusts ACTIVE/STILL/INACTIVITY thresholds accordingly.
    """

    def __init__(
        self,
        initial_active: float = 0.5,
        initial_still: float = 0.15,
        inactivity_timeout_seconds: float = 7200.0,
        adaptation_period_hours: float = 48.0,
    ) -> None:
        self._inactivity_timeout = inactivity_timeout_seconds
        self._adaptation_period = adaptation_period_hours * 3600

        self._threshold_active = AdaptiveThreshold(
            initial_active, adaptation_rate=0.02, min_value=0.1, max_value=2.0,
        )
        self._threshold_still = AdaptiveThreshold(
            initial_still, adaptation_rate=0.02, min_value=0.02, max_value=0.5,
        )
        self._start_time: Optional[float] = None
        self._last_adaptation: float = 0

    @property
    def threshold_active(self) -> float:
        return self._threshold_active.value

    @property
    def threshold_still(self) -> float:
        return self._threshold_still.value

    @property
    def inactivity_timeout(self) -> float:
        return self._inactivity_timeout

    def observe_variance(self, variance: float) -> None:
        """Feed CSI amplitude variance observation."""
        if self._start_time is None:
            self._start_time = time.time()
        self._threshold_active.observe(variance)
        self._threshold_still.observe(variance * 0.3)

    def maybe_adapt(self) -> bool:
        """Check if adaptation period has elapsed and recompute thresholds.

        Returns True if thresholds were adapted.
        """
        if self._start_time is None:
            return False
        now = time.time()
        elapsed = now - self._start_time

        # Only adapt after minimum observation period (1 hour)
        if elapsed < 3600:
            return False
        # Adapt at most every 5 minutes
        if now - self._last_adaptation < 300:
            return False

        self._last_adaptation = now
        old_active = self._threshold_active.value
        old_still = self._threshold_still.value

        self._threshold_active.adapt()
        self._threshold_still.adapt()

        # After adaptation period, also tune inactivity timeout
        if elapsed >= self._adaptation_period and self._inactivity_timeout == 7200.0:
            self._inactivity_timeout = 5400.0  # Tighten to 90 min after learning
            logger.info(f"Inactivity timeout adapted: 7200s -> {self._inactivity_timeout}s")

        changed = (
            abs(self._threshold_active.value - old_active) > 0.001
            or abs(self._threshold_still.value - old_still) > 0.001
        )
        if changed:
            logger.info(
                f"Thresholds adapted: active={self._threshold_active.value:.4f}, "
                f"still={self._threshold_still.value:.4f}"
            )
        return changed

    @property
    def adaptation_progress(self) -> float:
        """Returns 0.0-1.0 indicating how far through the adaptation period."""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(elapsed / self._adaptation_period, 1.0)


class FallConfidenceTuner:
    """Per-zone fall confidence threshold tuning based on false positive rate.

    If a zone generates too many false positives, raises the confidence threshold.
    If too few detections (possible missed falls), lowers it.
    """

    def __init__(
        self,
        initial_threshold: float = 0.85,
        min_threshold: float = 0.7,
        max_threshold: float = 0.95,
        target_fp_per_day: float = 2.0,
        adaptation_rate: float = 0.02,
    ) -> None:
        self._threshold = initial_threshold
        self._min = min_threshold
        self._max = max_threshold
        self._target_fp = target_fp_per_day
        self._rate = adaptation_rate
        self._detections: deque = deque(maxlen=1000)
        self._confirmed_falls: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float:
        return self._threshold

    def record_detection(self, confidence: float, confirmed: bool) -> None:
        """Record a fall detection event.

        Args:
            confidence: Model confidence score.
            confirmed: Whether a human confirmed this was a real fall.
        """
        with self._lock:
            self._detections.append({
                "timestamp": time.time(), "confidence": confidence, "confirmed": confirmed,
            })
            if confirmed:
                self._confirmed_falls.append(time.time())

    def adapt(self) -> float:
        """Adjust threshold based on recent false positive rate.

        Returns the new threshold value.
        """
        with self._lock:
            if len(self._detections) < 10:
                return self._threshold

            now = time.time()
            day_ago = now - 86400
            recent = [d for d in self._detections if d["timestamp"] > day_ago]

            if not recent:
                return self._threshold

            false_positives = sum(1 for d in recent if not d["confirmed"])
            fp_rate = false_positives / max(len(recent), 1) * len(recent)

            if fp_rate > self._target_fp * 1.5:
                new_threshold = min(self._threshold + self._rate, self._max)
            elif fp_rate < self._target_fp * 0.5:
                new_threshold = max(self._threshold - self._rate * 0.5, self._min)
            else:
                return self._threshold

            old = self._threshold
            self._threshold = new_threshold
            if abs(new_threshold - old) > 0.001:
                logger.info(f"Fall confidence threshold adapted: {old:.3f} -> {new_threshold:.3f}")
            return self._threshold
