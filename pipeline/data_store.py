"""
Shared data store for ElderCare inference results.

Provides thread-safe in-memory storage (fast reads) backed by SQLite
persistence (survives restarts). On startup, state is restored from
the last known SQLite snapshot. All writes go to both memory and a
write-behind buffer that flushes to SQLite every 60 seconds.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pipeline.csi_quality import QualityReport
from pipeline.persistence import SQLitePersistence

logger = logging.getLogger(__name__)

# Lazily loaded InfluxWriter -- None until first successful import
_influx_writer: Optional["InfluxWriter"] = None


def _get_influx_writer() -> Optional["InfluxWriter"]:
    """Return the shared InfluxWriter singleton, or None if unavailable."""
    global _influx_writer
    if _influx_writer is not None:
        return _influx_writer
    try:
        from pipeline.influx_writer import InfluxWriter
        _influx_writer = InfluxWriter()
        return _influx_writer
    except Exception:
        logger.debug("InfluxDB writer not available", exc_info=True)
        return None

MAX_HISTORY = 3600  # 1 hour at 1 per second
MAX_ALERTS = 500
MAX_SLEEP_RECORDS = 90  # 90 nights


@dataclass
class ZoneStatus:
    zone_id: str
    name: str
    last_seen: Optional[float] = None
    activity_state: str = "unknown"
    respiration_bpm: Optional[float] = None
    respiration_confidence: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    heart_rate_confidence: Optional[float] = None
    fall_detected: bool = False
    fall_confidence: float = 0.0
    sleep_stage: str = "unknown"
    sleep_score: Optional[float] = None
    online: bool = False


@dataclass
class VitalsRecord:
    timestamp: float
    respiration_bpm: Optional[float]
    respiration_confidence: Optional[float]
    heart_rate_bpm: Optional[float]
    heart_rate_confidence: Optional[float]


@dataclass
class SleepRecord:
    date: str
    sleep_score: float
    awake_pct: float
    light_pct: float
    deep_pct: float
    sleep_efficiency: float


class InferenceDataStore:
    """Thread-safe store for all inference results.

    Keeps fast in-memory data structures for reads. Writes are also
    buffered to SQLite via write-behind persistence. On init, previous
    state is restored from SQLite if the database exists.
    """

    def __init__(self, db_path: Optional[str] = "data/eldercare.db") -> None:
        self._lock = threading.Lock()
        self._zone_status: dict[str, ZoneStatus] = {}
        self._vitals_history: dict[str, deque[VitalsRecord]] = {}
        self._sleep_records: dict[str, list[SleepRecord]] = {}
        self._alerts: list[dict] = []
        self._csi_quality: dict[str, QualityReport] = {}

        self._persistence: Optional[SQLitePersistence] = None
        if db_path is not None:
            try:
                self._persistence = SQLitePersistence(db_path=db_path)
                self._restore_from_db()
                self._persistence.start_flush_timer()
            except Exception:
                logger.exception("Failed to initialize SQLite persistence, running in-memory only")
                self._persistence = None

    def _restore_from_db(self) -> None:
        """Load last known state from SQLite into in-memory structures."""
        if self._persistence is None:
            return
        try:
            state = self._persistence.restore()
        except Exception:
            logger.exception("Error restoring state from SQLite")
            return

        # Restore zone statuses
        for zs_data in state.get("zone_statuses", []):
            zone_id = zs_data["zone_id"]
            self._zone_status[zone_id] = ZoneStatus(
                zone_id=zone_id,
                name=zs_data.get("name", zone_id),
                activity_state=zs_data.get("activity_state", "unknown"),
                respiration_bpm=zs_data.get("respiration_bpm"),
                respiration_confidence=zs_data.get("respiration_confidence"),
                heart_rate_bpm=zs_data.get("heart_rate_bpm"),
                heart_rate_confidence=zs_data.get("heart_rate_confidence"),
                fall_detected=zs_data.get("fall_detected", False),
                fall_confidence=zs_data.get("fall_confidence", 0.0),
                sleep_stage=zs_data.get("sleep_stage", "unknown"),
                sleep_score=zs_data.get("sleep_score"),
                online=zs_data.get("online", False),
                last_seen=zs_data.get("last_seen"),
            )

        # Restore vitals history
        for zone_id, records in state.get("vitals_history", {}).items():
            dq: deque[VitalsRecord] = deque(maxlen=MAX_HISTORY)
            for r in records[-MAX_HISTORY:]:
                dq.append(VitalsRecord(
                    timestamp=r["timestamp"],
                    respiration_bpm=r.get("respiration_bpm"),
                    respiration_confidence=r.get("respiration_confidence"),
                    heart_rate_bpm=r.get("heart_rate_bpm"),
                    heart_rate_confidence=r.get("heart_rate_confidence"),
                ))
            self._vitals_history[zone_id] = dq

        # Restore alerts
        self._alerts = state.get("alerts", [])[-MAX_ALERTS:]

        # Restore sleep records
        for zone_id, records in state.get("sleep_records", {}).items():
            sr_list: list[SleepRecord] = []
            for r in records[-MAX_SLEEP_RECORDS:]:
                sr_list.append(SleepRecord(
                    date=r["date"],
                    sleep_score=r["sleep_score"],
                    awake_pct=r["awake_pct"],
                    light_pct=r["light_pct"],
                    deep_pct=r["deep_pct"],
                    sleep_efficiency=r["sleep_efficiency"],
                ))
            self._sleep_records[zone_id] = sr_list

    def flush(self) -> None:
        """Force-flush all buffered writes to SQLite. Call on graceful shutdown."""
        if self._persistence is not None:
            self._persistence.flush()

    def close(self) -> None:
        """Flush and close persistence. Call on process exit."""
        if self._persistence is not None:
            self._persistence.close()
        global _influx_writer
        if _influx_writer is not None:
            _influx_writer.close()
            _influx_writer = None

    def ensure_zone(self, zone_id: str, name: str) -> None:
        with self._lock:
            if zone_id not in self._zone_status:
                self._zone_status[zone_id] = ZoneStatus(zone_id=zone_id, name=name)
            if zone_id not in self._vitals_history:
                self._vitals_history[zone_id] = deque(maxlen=MAX_HISTORY)
            if zone_id not in self._sleep_records:
                self._sleep_records[zone_id] = []

    def update_activity(self, zone_id: str, state: str, alert: Optional[str] = None) -> None:
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.activity_state = state
                zs.last_seen = time.time()
                zs.online = True
        if self._persistence is not None:
            self._persistence.buffer_zone_status(zone_id, {
                "activity_state": state, "online": 1,
                "last_seen": time.time(), "updated_at": time.time(),
            })
        influx = _get_influx_writer()
        if influx is not None:
            influx.write_activity(zone_id, state)

    def update_fall(self, zone_id: str, detected: bool, confidence: float) -> None:
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.fall_detected = detected
                zs.fall_confidence = confidence
                zs.last_seen = time.time()
        if self._persistence is not None:
            self._persistence.buffer_zone_status(zone_id, {
                "fall_detected": int(detected), "fall_confidence": confidence,
                "last_seen": time.time(), "updated_at": time.time(),
            })

    def update_vitals(
        self, zone_id: str,
        respiration_bpm: Optional[float], respiration_confidence: Optional[float],
        heart_rate_bpm: Optional[float], heart_rate_confidence: Optional[float],
    ) -> None:
        now = time.time()
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.respiration_bpm = respiration_bpm
                zs.respiration_confidence = respiration_confidence
                zs.heart_rate_bpm = heart_rate_bpm
                zs.heart_rate_confidence = heart_rate_confidence

            record = VitalsRecord(
                timestamp=now,
                respiration_bpm=respiration_bpm,
                respiration_confidence=respiration_confidence,
                heart_rate_bpm=heart_rate_bpm,
                heart_rate_confidence=heart_rate_confidence,
            )
            self._vitals_history.setdefault(zone_id, deque(maxlen=MAX_HISTORY)).append(record)

        if self._persistence is not None:
            self._persistence.buffer_zone_status(zone_id, {
                "respiration_bpm": respiration_bpm,
                "respiration_confidence": respiration_confidence,
                "heart_rate_bpm": heart_rate_bpm,
                "heart_rate_confidence": heart_rate_confidence,
                "updated_at": now,
            })
            self._persistence.buffer_vitals(zone_id, {
                "timestamp": now,
                "respiration_bpm": respiration_bpm,
                "respiration_confidence": respiration_confidence,
                "heart_rate_bpm": heart_rate_bpm,
                "heart_rate_confidence": heart_rate_confidence,
            })
        influx = _get_influx_writer()
        if influx is not None:
            influx.write_vitals(
                zone_id,
                respiration_bpm=respiration_bpm,
                heart_rate_bpm=heart_rate_bpm,
                confidence=respiration_confidence,
            )

    def update_sleep(self, zone_id: str, stage: str, score: Optional[float] = None) -> None:
        now = time.time()
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.sleep_stage = stage
                zs.sleep_score = score
                zs.last_seen = now
        if self._persistence is not None:
            self._persistence.buffer_zone_status(zone_id, {
                "sleep_stage": stage, "sleep_score": score,
                "last_seen": now, "updated_at": now,
            })
        influx = _get_influx_writer()
        if influx is not None:
            influx.write_sleep_epoch(zone_id, stage, score)

    def add_sleep_record(self, zone_id: str, record: SleepRecord) -> None:
        with self._lock:
            records = self._sleep_records.setdefault(zone_id, [])
            records.append(record)
            if len(records) > MAX_SLEEP_RECORDS:
                records.pop(0)
        if self._persistence is not None:
            self._persistence.buffer_sleep_record(zone_id, {
                "date": record.date,
                "sleep_score": record.sleep_score,
                "awake_pct": record.awake_pct,
                "light_pct": record.light_pct,
                "deep_pct": record.deep_pct,
                "sleep_efficiency": record.sleep_efficiency,
            })

    def add_alert(self, alert: dict) -> None:
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > MAX_ALERTS:
                self._alerts = self._alerts[-MAX_ALERTS:]
        if self._persistence is not None:
            self._persistence.buffer_alert(alert)

    def get_zone_status(self, zone_id: str) -> Optional[ZoneStatus]:
        with self._lock:
            return self._zone_status.get(zone_id)

    def get_all_zone_statuses(self) -> list[ZoneStatus]:
        with self._lock:
            return list(self._zone_status.values())

    def get_vitals_history(self, zone_id: str, n: int = 300) -> list[dict]:
        with self._lock:
            records = list(self._vitals_history.get(zone_id, []))[-n:]
            return [{"timestamp": r.timestamp, "respiration_bpm": r.respiration_bpm, "respiration_confidence": r.respiration_confidence, "heart_rate_bpm": r.heart_rate_bpm, "heart_rate_confidence": r.heart_rate_confidence} for r in records]

    def get_sleep_records(self, zone_id: str, n: int = 30) -> list[dict]:
        with self._lock:
            records = self._sleep_records.get(zone_id, [])[-n:]
            return [{"date": r.date, "sleep_score": r.sleep_score, "awake_pct": r.awake_pct, "light_pct": r.light_pct, "deep_pct": r.deep_pct, "sleep_efficiency": r.sleep_efficiency} for r in records]

    def get_alerts(self, n: int = 50) -> list[dict]:
        with self._lock:
            return list(self._alerts[-n:])

    def update_csi_quality(self, zone_id: str, report: QualityReport) -> None:
        """Store the latest CSI quality report for a zone."""
        with self._lock:
            self._csi_quality[zone_id] = report

    def get_csi_quality(self, zone_id: str) -> Optional[QualityReport]:
        """Return the latest CSI quality report for a zone, or None."""
        with self._lock:
            return self._csi_quality.get(zone_id)


# Global singleton
store = InferenceDataStore()
