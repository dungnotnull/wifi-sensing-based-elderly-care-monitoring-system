"""Local telemetry for monitoring system performance and accuracy.

Opt-in, local-only telemetry that tracks inference latency, false positive/negative
rates, system resource usage, and operational metrics. All data stays on-device.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_METRICS_HISTORY = 86400  # 1 day at 1 per second
TELEMETRY_DIR = "data/telemetry"


@dataclass
class InferenceMetric:
    """Single inference measurement."""
    timestamp: float
    model_name: str
    zone_id: str
    latency_ms: float
    confidence: Optional[float] = None
    prediction: Optional[str] = None


@dataclass
class SystemMetric:
    """System resource snapshot."""
    timestamp: float
    cpu_pct: float
    memory_pct: float
    disk_pct: float
    active_workers: int
    total_workers: int
    queue_depth: int = 0


@dataclass
class FeedbackMetric:
    """Caregiver feedback on an alert."""
    timestamp: float
    zone_id: str
    event_type: str
    was_correct: bool
    notes: Optional[str] = None


class TelemetryCollector:
    """Thread-safe collector for all telemetry data."""

    def __init__(self, storage_dir: str = TELEMETRY_DIR) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._inference_metrics: deque[InferenceMetric] = deque(maxlen=MAX_METRICS_HISTORY)
        self._system_metrics: deque[SystemMetric] = deque(maxlen=MAX_METRICS_HISTORY)
        self._feedback: list[FeedbackMetric] = []
        self._start_time = time.time()
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background flush thread."""
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop background flush and write final state."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self._flush_to_disk()

    def record_inference(
        self,
        model_name: str,
        zone_id: str,
        latency_ms: float,
        confidence: Optional[float] = None,
        prediction: Optional[str] = None,
    ) -> None:
        metric = InferenceMetric(
            timestamp=time.time(),
            model_name=model_name,
            zone_id=zone_id,
            latency_ms=latency_ms,
            confidence=confidence,
            prediction=prediction,
        )
        with self._lock:
            self._inference_metrics.append(metric)

    def record_system(
        self,
        cpu_pct: float,
        memory_pct: float,
        disk_pct: float,
        active_workers: int,
        total_workers: int,
        queue_depth: int = 0,
    ) -> None:
        metric = SystemMetric(
            timestamp=time.time(),
            cpu_pct=cpu_pct,
            memory_pct=memory_pct,
            disk_pct=disk_pct,
            active_workers=active_workers,
            total_workers=total_workers,
            queue_depth=queue_depth,
        )
        with self._lock:
            self._system_metrics.append(metric)

    def record_feedback(
        self,
        zone_id: str,
        event_type: str,
        was_correct: bool,
        notes: Optional[str] = None,
    ) -> None:
        feedback = FeedbackMetric(
            timestamp=time.time(),
            zone_id=zone_id,
            event_type=event_type,
            was_correct=was_correct,
            notes=notes,
        )
        with self._lock:
            self._feedback.append(feedback)

    def get_inference_stats(self, model_name: Optional[str] = None, last_n: int = 1000) -> dict:
        """Compute inference latency statistics."""
        with self._lock:
            metrics = list(self._inference_metrics)[-last_n:]
            if model_name:
                metrics = [m for m in metrics if m.model_name == model_name]

        if not metrics:
            return {"count": 0}

        latencies = sorted([m.latency_ms for m in metrics])
        n = len(latencies)
        return {
            "count": n,
            "avg_ms": round(sum(latencies) / n, 2),
            "p50_ms": round(latencies[n // 2], 2),
            "p95_ms": round(latencies[int(n * 0.95)], 2),
            "p99_ms": round(latencies[min(int(n * 0.99), n - 1)], 2),
            "max_ms": round(latencies[-1], 2),
            "min_ms": round(latencies[0], 2),
        }

    def get_feedback_stats(self) -> dict:
        """Compute feedback accuracy statistics."""
        with self._lock:
            feedbacks = list(self._feedback)

        if not feedbacks:
            return {"count": 0}

        correct = sum(1 for f in feedbacks if f.was_correct)
        by_type: dict[str, dict[str, int]] = {}
        for f in feedbacks:
            entry = by_type.setdefault(f.event_type, {"correct": 0, "incorrect": 0})
            if f.was_correct:
                entry["correct"] += 1
            else:
                entry["incorrect"] += 1

        return {
            "count": len(feedbacks),
            "accuracy": round(correct / len(feedbacks), 3),
            "correct": correct,
            "incorrect": len(feedbacks) - correct,
            "by_event_type": by_type,
        }

    def get_system_trends(self, last_n: int = 3600) -> dict:
        """Get system resource usage trends."""
        with self._lock:
            metrics = list(self._system_metrics)[-last_n:]

        if not metrics:
            return {"count": 0}

        return {
            "count": len(metrics),
            "avg_cpu_pct": round(sum(m.cpu_pct for m in metrics) / len(metrics), 1),
            "avg_memory_pct": round(sum(m.memory_pct for m in metrics) / len(metrics), 1),
            "max_cpu_pct": round(max(m.cpu_pct for m in metrics), 1),
            "max_memory_pct": round(max(m.memory_pct for m in metrics), 1),
            "avg_disk_pct": round(sum(m.disk_pct for m in metrics) / len(metrics), 1),
            "uptime_hours": round((time.time() - self._start_time) / 3600, 1),
        }

    def get_dashboard_summary(self) -> dict:
        """Compact summary for the telemetry dashboard widget."""
        return {
            "inference": self.get_inference_stats(last_n=500),
            "system": self.get_system_trends(last_n=3600),
            "feedback": self.get_feedback_stats(),
            "uptime_hours": round((time.time() - self._start_time) / 3600, 1),
        }

    def _flush_loop(self) -> None:
        """Background loop that flushes metrics to disk every 5 minutes."""
        while not self._stop_event.wait(300):
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """Write current metrics to JSON files."""
        with self._lock:
            inference = [
                {"t": m.timestamp, "model": m.model_name, "zone": m.zone_id,
                 "latency_ms": m.latency_ms, "confidence": m.confidence}
                for m in self._inference_metrics
            ]
            system = [
                {"t": m.timestamp, "cpu": m.cpu_pct, "mem": m.memory_pct,
                 "disk": m.disk_pct, "workers": f"{m.active_workers}/{m.total_workers}"}
                for m in self._system_metrics
            ]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        try:
            with open(self._dir / f"inference_{timestamp}.json", "w") as f:
                json.dump(inference[-5000:], f)
            with open(self._dir / f"system_{timestamp}.json", "w") as f:
                json.dump(system[-5000:], f)
        except IOError as e:
            logger.warning(f"Failed to flush telemetry: {e}")


# Module-level singleton
telemetry = TelemetryCollector()
