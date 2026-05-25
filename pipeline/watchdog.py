"""
Worker Watchdog and System Health Monitor.

Background daemon thread that monitors inference worker health via
heartbeat timestamps in a shared dict. Provides auto-restart logic
for unhealthy workers and system-level resource monitoring.
"""

import logging
import multiprocessing as mp
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import psutil

if TYPE_CHECKING:
    from pipeline.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

# Heartbeat interval must match what workers send (seconds)
HEARTBEAT_INTERVAL = 5.0
# How often the watchdog checks heartbeats
CHECK_INTERVAL = 10.0
# Number of consecutive misses before flagging a worker
MAX_MISSED_HEARTBEATS = 3
# Maximum restart attempts within the cooldown window
MAX_RESTARTS = 3
# Time window for restart counting (seconds)
RESTART_WINDOW = 600.0  # 10 minutes


@dataclass
class WorkerHealth:
    """Health status for a single worker."""

    last_heartbeat: Optional[float] = None
    status: str = "unknown"  # healthy | unhealthy | unknown
    restart_count: int = 0
    restart_timestamps: list[float] = field(default_factory=list)


class WorkerWatchdog:
    """Monitors worker processes via heartbeat timestamps.

    Each inference worker writes its current timestamp to a shared dict
    every 5 seconds. The watchdog checks every 10 seconds; if a worker
    misses 3 consecutive checks (30s), it is flagged as unhealthy.
    """

    def __init__(
        self,
        heartbeat_dict: dict[str, float],
        stop_event: threading.Event,
        engine: "InferenceEngine",
    ) -> None:
        self._heartbeat_dict = heartbeat_dict
        self._stop_event = stop_event
        self._engine = engine
        self._worker_health: dict[str, WorkerHealth] = {}
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the watchdog background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        # Initialize health entries for all registered workers
        for name in self._heartbeat_dict:
            self._worker_health[name] = WorkerHealth()
        self._thread = threading.Thread(target=self._run, name="worker-watchdog", daemon=True)
        self._thread.start()
        logger.info("WorkerWatchdog started (check_interval=%ss, max_missed=%d)", CHECK_INTERVAL, MAX_MISSED_HEARTBEATS)

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("WorkerWatchdog stopped")

    def _run(self) -> None:
        """Main watchdog loop."""
        while not self._stop_event.wait(CHECK_INTERVAL):
            try:
                self._check_workers()
            except Exception:
                logger.exception("Error in watchdog check loop")

    def _check_workers(self) -> None:
        """Check all worker heartbeats and flag unhealthy ones."""
        now = time.time()
        for name, last_hb in list(self._heartbeat_dict.items()):
            with self._lock:
                health = self._worker_health.get(name)
                if health is None:
                    health = WorkerHealth()
                    self._worker_health[name] = health

                health.last_heartbeat = last_hb

                if last_hb == 0.0:
                    # Worker has not sent first heartbeat yet
                    health.status = "unknown"
                    continue

                elapsed = now - last_hb
                missed_intervals = int(elapsed / HEARTBEAT_INTERVAL)

                if missed_intervals >= MAX_MISSED_HEARTBEATS:
                    if health.status != "unhealthy":
                        logger.critical(
                            "Worker '%s' missed %d heartbeats (%.1fs since last). Flagging as unhealthy.",
                            name, missed_intervals, elapsed,
                        )
                    health.status = "unhealthy"
                    self._attempt_restart(name)
                else:
                    health.status = "healthy"

    def _attempt_restart(self, worker_name: str) -> None:
        """Try to restart an unhealthy worker. Gives up after MAX_RESTARTS in RESTART_WINDOW."""
        with self._lock:
            health = self._worker_health.get(worker_name)
            if health is None:
                return

            # Prune restart timestamps outside the window
            cutoff = time.time() - RESTART_WINDOW
            health.restart_timestamps = [t for t in health.restart_timestamps if t > cutoff]

            if len(health.restart_timestamps) >= MAX_RESTARTS:
                logger.error(
                    "Worker '%s' exceeded max restarts (%d in %ds). Giving up.",
                    worker_name, MAX_RESTARTS, int(RESTART_WINDOW),
                )
                return

            health.restart_timestamps.append(time.time())
            health.restart_count += 1

        logger.warning("Attempting restart #%d for worker '%s'", health.restart_count, worker_name)
        try:
            self.restart_worker(worker_name)
        except Exception:
            logger.exception("Failed to restart worker '%s'", worker_name)

    def restart_worker(self, worker_name: str) -> None:
        """Terminate and recreate a worker process by name.

        The InferenceEngine is responsible for the actual process
        lifecycle; this method delegates to engine.restart_worker().
        """
        self._engine.restart_worker(worker_name)

    def get_unhealthy_workers(self) -> list[str]:
        """Return names of workers currently flagged as unhealthy."""
        with self._lock:
            return [name for name, h in self._worker_health.items() if h.status == "unhealthy"]

    def get_all_worker_status(self) -> dict[str, dict[str, Any]]:
        """Return health info for every tracked worker."""
        with self._lock:
            return {
                name: {
                    "last_heartbeat": h.last_heartbeat,
                    "status": h.status,
                    "restart_count": h.restart_count,
                }
                for name, h in self._worker_health.items()
            }

    def register_worker(self, name: str) -> None:
        """Register a new worker for monitoring."""
        with self._lock:
            if name not in self._worker_health:
                self._worker_health[name] = WorkerHealth()


def get_system_health(
    worker_status: Optional[dict[str, dict[str, Any]]] = None,
    mqtt_connected: bool = False,
    start_time: float = 0.0,
) -> dict[str, Any]:
    """Collect system-level resource metrics and worker health.

    Returns a dict with disk, memory, CPU usage, MQTT connection status,
    worker status, and uptime.
    """
    disk_usage = shutil.disk_usage(os.path.sep)
    disk_pct = (disk_usage.used / disk_usage.total) * 100.0

    mem = psutil.virtual_memory()
    mem_pct = mem.percent

    cpu_pct = psutil.cpu_percent(interval=0.5)

    uptime = time.time() - start_time if start_time > 0 else 0.0

    alerts: list[str] = []
    if disk_pct > 90.0:
        alerts.append(f"Disk usage critical: {disk_pct:.1f}%")
    if mem_pct > 85.0:
        alerts.append(f"Memory usage critical: {mem_pct:.1f}%")
    if worker_status:
        unhealthy = [n for n, s in worker_status.items() if s.get("status") == "unhealthy"]
        if unhealthy:
            alerts.append(f"Unhealthy workers: {', '.join(unhealthy)}")

    return {
        "disk_usage_pct": round(disk_pct, 1),
        "memory_usage_pct": round(mem_pct, 1),
        "cpu_usage_pct": round(cpu_pct, 1),
        "mqtt_connected": mqtt_connected,
        "worker_status": worker_status or {},
        "uptime_seconds": round(uptime, 1),
        "alerts": alerts,
    }
