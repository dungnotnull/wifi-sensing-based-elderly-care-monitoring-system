"""
InfluxDB time-series writer for ElderCare inference data.

Buffers points in memory and flushes to InfluxDB in batches for
efficiency. Degrades gracefully: if the influxdb-client library is
not installed or the InfluxDB server is unreachable, all write
calls become silent no-ops.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Batch defaults
_BATCH_SIZE = 100
_FLUSH_INTERVAL = 10.0  # seconds


class InfluxWriter:
    """Writes time-series data to InfluxDB for historical analysis.

    Thread-safe. If InfluxDB is unavailable at init time or during
    writes, the writer silently skips them so the rest of the pipeline
    is unaffected.
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        bucket: str = "eldercare",
        org: str = "eldercare",
        token: str = "",
    ) -> None:
        self._url = url
        self._bucket = bucket
        self._org = org
        self._token = token
        self._lock = threading.Lock()
        self._buffer: list[str] = []
        self._available = False
        self._client: Any = None
        self._write_api: Any = None
        self._flush_timer: Optional[threading.Timer] = None

        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            self._client = InfluxDBClient(url=url, token=token, org=org)
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            # Verify connectivity with a lightweight health check
            self._client.health()
            self._available = True
            self._start_flush_timer()
            logger.info("InfluxDB writer initialized: %s", url)
        except ImportError:
            logger.info("influxdb-client not installed; InfluxDB writes disabled")
        except Exception:
            logger.warning("InfluxDB unavailable at %s; writes disabled", url, exc_info=True)

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def write_vitals(
        self,
        zone_id: str,
        respiration_bpm: Optional[float],
        heart_rate_bpm: Optional[float],
        confidence: Optional[float],
    ) -> None:
        """Write a vitals data point."""
        ts = self._now_nano()
        fields: list[str] = []
        if respiration_bpm is not None:
            fields.append(f"respiration_bpm={respiration_bpm}")
        if heart_rate_bpm is not None:
            fields.append(f"heart_rate_bpm={heart_rate_bpm}")
        if confidence is not None:
            fields.append(f"confidence={confidence}")
        if not fields:
            return
        line = f"vitals,zone={zone_id} {','.join(fields)} {ts}"
        self._enqueue(line)

    def write_activity(self, zone_id: str, state: str) -> None:
        """Write an activity state change."""
        ts = self._now_nano()
        line = f'activity,zone={zone_id} state="{state}" {ts}'
        self._enqueue(line)

    def write_sleep_epoch(
        self,
        zone_id: str,
        stage: str,
        score: Optional[float],
    ) -> None:
        """Write a sleep epoch."""
        ts = self._now_nano()
        fields: list[str] = [f'stage="{stage}"']
        if score is not None:
            fields.append(f"score={score}")
        line = f"sleep,zone={zone_id} {','.join(fields)} {ts}"
        self._enqueue(line)

    def write_system_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """Write a generic system metric."""
        ts = self._now_nano()
        tag_str = ""
        if tags:
            tag_str = "," + ",".join(f"{k}={v}" for k, v in tags.items())
        line = f"system{tag_str} {metric_name}={value} {ts}"
        self._enqueue(line)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining points and close the client."""
        self._cancel_flush_timer()
        self._flush()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Error closing InfluxDB client", exc_info=True)
        self._available = False
        logger.info("InfluxDB writer closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, line: str) -> None:
        """Add a line-protocol point to the buffer, flushing when full."""
        if not self._available:
            return
        flush_needed = False
        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= _BATCH_SIZE:
                flush_needed = True
        if flush_needed:
            self._flush()

    def _flush(self) -> None:
        """Send all buffered points to InfluxDB."""
        if not self._available or self._write_api is None:
            return
        with self._lock:
            batch = self._buffer
            self._buffer = []
        if not batch:
            return
        try:
            self._write_api.write(
                bucket=self._bucket,
                org=self._org,
                record="\n".join(batch),
            )
            logger.debug("Flushed %d points to InfluxDB", len(batch))
        except Exception:
            logger.warning("Failed to write %d points to InfluxDB", len(batch), exc_info=True)
            # Re-enqueue on failure so data is not lost
            with self._lock:
                self._buffer = batch + self._buffer

    def _start_flush_timer(self) -> None:
        """Start periodic flush timer."""
        self._flush_timer = threading.Timer(_FLUSH_INTERVAL, self._flush_and_reschedule)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _flush_and_reschedule(self) -> None:
        """Flush then schedule the next flush."""
        self._flush()
        if self._available:
            self._start_flush_timer()

    def _cancel_flush_timer(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    @staticmethod
    def _now_nano() -> int:
        """Return current UTC time as nanosecond epoch."""
        return int(datetime.now(timezone.utc).timestamp() * 1e9)
