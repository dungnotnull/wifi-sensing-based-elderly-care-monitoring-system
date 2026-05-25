"""Gradual rollout (shadow mode) for safe first deployment.

In shadow mode, the system runs and detects events but does NOT send alerts.
All detections are logged with optional ground-truth labels. After a tuning
period (typically 48-72 hours), live alerting can be enabled with confidence.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowEvent:
    """A detected event during shadow mode."""
    timestamp: float
    zone_id: str
    event_type: str
    model_name: str
    confidence: float
    data: dict
    ground_truth: Optional[str] = None
    reviewer: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "zone_id": self.zone_id,
            "event_type": self.event_type,
            "model_name": self.model_name,
            "confidence": self.confidence,
            "data": self.data,
            "ground_truth": self.ground_truth,
            "reviewer": self.reviewer,
        }


@dataclass
class ShadowStats:
    """Accumulated statistics from shadow mode."""
    total_events: int = 0
    true_positives: int = 0
    false_positives: int = 0
    unlabeled: int = 0
    by_type: dict = field(default_factory=dict)
    by_zone: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        labeled = self.true_positives + self.false_positives
        return self.true_positives / labeled if labeled > 0 else 0.0

    @property
    def fp_rate_per_day(self) -> float:
        return self.false_positives  # Caller should divide by days elapsed


class ShadowMode:
    """Manages shadow mode operation for gradual rollout."""

    def __init__(
        self,
        enabled: bool = False,
        log_path: str = "data/shadow_mode/events.jsonl",
        auto_enable_live_after_hours: float = 72.0,
    ) -> None:
        self._enabled = enabled
        self._log_path = Path(log_path)
        self._auto_enable_after = auto_enable_live_after_hours * 3600
        self._start_time: Optional[float] = None
        self._events: list[ShadowEvent] = []
        self._suppress_count: int = 0

    @property
    def is_shadow(self) -> bool:
        """True if shadow mode is active (alerts suppressed)."""
        return self._enabled

    @property
    def is_live(self) -> bool:
        """True if live mode is active (alerts sent)."""
        if self._start_time is None:
            return not self._enabled
        elapsed = time.time() - self._start_time
        if self._enabled and elapsed >= self._auto_enable_after:
            logger.info(
                f"Shadow mode auto-expired after {elapsed / 3600:.1f} hours. "
                f"Switching to live mode."
            )
            self._enabled = False
            return True
        return not self._enabled

    def start(self) -> None:
        """Begin shadow mode session."""
        if self._enabled:
            self._start_time = time.time()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Shadow mode STARTED — all alerts will be suppressed and logged")
            logger.info(f"Events will be saved to {self._log_path}")

    def record_event(
        self,
        zone_id: str,
        event_type: str,
        model_name: str,
        confidence: float,
        data: dict,
    ) -> Optional[ShadowEvent]:
        """Record a detection event. Returns the event if in shadow mode, None otherwise."""
        if not self._enabled:
            return None

        event = ShadowEvent(
            timestamp=time.time(),
            zone_id=zone_id,
            event_type=event_type,
            model_name=model_name,
            confidence=confidence,
            data=data,
        )
        self._events.append(event)
        self._suppress_count += 1

        self._append_to_log(event)

        if self._suppress_count % 10 == 0:
            logger.info(f"Shadow mode: {self._suppress_count} alerts suppressed so far")

        return event

    def label_event(
        self,
        event_index: int,
        ground_truth: str,
        reviewer: str = "manual",
    ) -> None:
        """Label a recorded event with ground truth."""
        if event_index < 0 or event_index >= len(self._events):
            return
        self._events[event_index].ground_truth = ground_truth
        self._events[event_index].reviewer = reviewer

    def compute_stats(self) -> ShadowStats:
        """Compute statistics from labeled events."""
        stats = ShadowStats()
        for event in self._events:
            stats.total_events += 1
            stats.by_zone.setdefault(event.zone_id, 0)
            stats.by_zone[event.zone_id] += 1
            stats.by_type.setdefault(event.event_type, 0)
            stats.by_type[event.event_type] += 1

            if event.ground_truth is None:
                stats.unlabeled += 1
            elif event.ground_truth == "true_positive":
                stats.true_positives += 1
            elif event.ground_truth == "false_positive":
                stats.false_positives += 1

        return stats

    def generate_report(self) -> dict:
        """Generate a shadow mode report for review."""
        stats = self.compute_stats()
        elapsed_hours = 0.0
        if self._start_time:
            elapsed_hours = (time.time() - self._start_time) / 3600

        return {
            "mode": "shadow" if self.is_shadow else "live",
            "elapsed_hours": round(elapsed_hours, 1),
            "alerts_suppressed": self._suppress_count,
            "total_events": stats.total_events,
            "labeled": stats.total_events - stats.unlabeled,
            "unlabeled": stats.unlabeled,
            "true_positives": stats.true_positives,
            "false_positives": stats.false_positives,
            "precision": round(stats.precision, 3),
            "fp_rate_per_day": round(stats.false_positives / max(elapsed_hours / 24, 0.01), 2),
            "by_zone": stats.by_zone,
            "by_type": stats.by_type,
            "recommendation": self._recommendation(stats, elapsed_hours),
        }

    def _recommendation(self, stats: ShadowStats, elapsed_hours: float) -> str:
        """Generate a recommendation based on shadow mode results."""
        if elapsed_hours < 24:
            return "Insufficient data. Continue shadow mode for at least 48 hours."

        if stats.unlabeled > stats.total_events * 0.5:
            return f"{stats.unlabeled} events still unlabeled. Label events before going live."

        if stats.precision >= 0.9:
            return "Precision is high (>90%). Safe to switch to live mode."
        elif stats.precision >= 0.7:
            return "Precision is moderate (70-90%). Consider adjusting confidence threshold before going live."
        else:
            return "Precision is low (<70%). Do NOT go live. Retrain or retune model first."

    def switch_to_live(self) -> None:
        """Manually switch from shadow to live mode."""
        if self._enabled:
            self._enabled = False
            report = self.generate_report()
            logger.info(f"Switched to LIVE mode. Report: {json.dumps(report, indent=2)}")

    def _append_to_log(self, event: ShadowEvent) -> None:
        """Append an event to the JSONL log file."""
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except IOError as e:
            logger.warning(f"Failed to write shadow event to log: {e}")


# Module-level singleton (reads config at import time)
try:
    import yaml
    _cfg: dict = {}
    _cfg_path = Path("configs/thresholds.yaml")
    if _cfg_path.exists():
        with open(_cfg_path) as _f:
            _cfg = yaml.safe_load(_f) or {}
    _shadow_cfg = _cfg.get("shadow_mode", {})
    shadow_mode = ShadowMode(
        enabled=_shadow_cfg.get("enabled", False),
        log_path=_shadow_cfg.get("event_log", "data/shadow_mode/events.jsonl"),
        auto_enable_live_after_hours=_shadow_cfg.get("auto_enable_live_after_hours", 72.0),
    )
except Exception:
    shadow_mode = ShadowMode(enabled=False)
