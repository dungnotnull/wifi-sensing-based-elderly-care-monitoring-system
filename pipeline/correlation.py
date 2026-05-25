"""Correlation ID tracking for end-to-end pipeline tracing.

Each CSI packet carries a correlation_id through ingestion -> preprocessing
-> inference -> alerting, enabling latency tracing and false positive replay.
"""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MAX_TRACE_HISTORY = 10000


@dataclass
class TracePoint:
    """A single point in a packet's journey through the pipeline."""
    correlation_id: str
    stage: str
    timestamp: float
    zone_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PacketTrace:
    """Full trace of a single packet through the pipeline."""
    correlation_id: str
    zone_id: str
    created_at: float
    points: list[TracePoint] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> Optional[float]:
        if len(self.points) < 2:
            return None
        return (self.points[-1].timestamp - self.created_at) * 1000

    @property
    def stage_latencies_ms(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for i in range(1, len(self.points)):
            prev = self.points[i - 1]
            curr = self.points[i]
            stage_name = curr.stage
            latency = (curr.timestamp - prev.timestamp) * 1000
            result[stage_name] = latency
        return result


class CorrelationTracker:
    """Thread-safe tracker for packet correlation IDs and traces."""

    def __init__(self, max_history: int = MAX_TRACE_HISTORY) -> None:
        self._lock = threading.Lock()
        self._traces: dict[str, PacketTrace] = {}
        self._history: deque[PacketTrace] = deque(maxlen=max_history)
        self._stage_order: list[str] = ["ingestion", "preprocessing", "inference", "alerting"]

    def new_id(self) -> str:
        """Generate a new correlation ID."""
        return uuid.uuid4().hex[:16]

    def start_trace(self, zone_id: str, correlation_id: Optional[str] = None) -> str:
        """Start a new trace for a packet. Returns correlation_id."""
        cid = correlation_id or self.new_id()
        trace = PacketTrace(
            correlation_id=cid,
            zone_id=zone_id,
            created_at=time.time(),
        )
        with self._lock:
            self._traces[cid] = trace
        return cid

    def record(self, correlation_id: str, stage: str, metadata: Optional[dict] = None) -> None:
        """Record a trace point for a packet at a given pipeline stage."""
        point = TracePoint(
            correlation_id=correlation_id,
            stage=stage,
            timestamp=time.time(),
            zone_id="",
            metadata=metadata or {},
        )
        with self._lock:
            trace = self._traces.get(correlation_id)
            if trace:
                point.zone_id = trace.zone_id
                trace.points.append(point)
            else:
                logger.debug(f"Trace point for unknown correlation_id: {correlation_id}")

    def finish_trace(self, correlation_id: str) -> Optional[PacketTrace]:
        """Complete a trace and move it to history."""
        with self._lock:
            trace = self._traces.pop(correlation_id, None)
            if trace:
                self._history.append(trace)
            return trace

    def get_trace(self, correlation_id: str) -> Optional[PacketTrace]:
        with self._lock:
            trace = self._traces.get(correlation_id)
            if trace:
                return trace
            for h in reversed(self._history):
                if h.correlation_id == correlation_id:
                    return h
        return None

    def get_recent_traces(self, n: int = 100) -> list[dict]:
        """Get recent completed traces with latency info."""
        with self._lock:
            traces = list(self._history)[-n:]
        results: list[dict] = []
        for t in traces:
            results.append({
                "correlation_id": t.correlation_id,
                "zone_id": t.zone_id,
                "total_latency_ms": t.total_latency_ms,
                "stage_latencies_ms": t.stage_latencies_ms,
                "stages": [p.stage for p in t.points],
                "created_at": t.created_at,
            })
        return results

    def get_latency_stats(self, n: int = 1000) -> dict:
        """Compute latency statistics from recent traces."""
        with self._lock:
            traces = list(self._history)[-n:]

        if not traces:
            return {"count": 0}

        latencies = [t.total_latency_ms for t in traces if t.total_latency_ms is not None]
        if not latencies:
            return {"count": len(traces), "completed": 0}

        latencies.sort()
        count = len(latencies)
        stage_latencies: dict[str, list[float]] = {}
        for t in traces:
            for stage, lat in t.stage_latencies_ms.items():
                stage_latencies.setdefault(stage, []).append(lat)

        stats: dict[str, Any] = {
            "count": len(traces),
            "completed": count,
            "p50_ms": latencies[count // 2],
            "p95_ms": latencies[int(count * 0.95)],
            "p99_ms": latencies[min(int(count * 0.99), count - 1)],
            "avg_ms": sum(latencies) / count,
            "max_ms": latencies[-1],
        }
        for stage, lats in stage_latencies.items():
            if lats:
                stats[f"{stage}_avg_ms"] = sum(lats) / len(lats)
        return stats


# Module-level singleton
tracker = CorrelationTracker()
