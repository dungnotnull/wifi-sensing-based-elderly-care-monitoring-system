"""
Shared in-memory data store for ElderCare inference results.

Provides thread-safe storage of zone status, vital signs history,
alerts, and sleep scores — consumed by the dashboard API and alert manager.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

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
    """Thread-safe store for all inference results."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._zone_status: dict[str, ZoneStatus] = {}
        self._vitals_history: dict[str, deque[VitalsRecord]] = {}
        self._sleep_records: dict[str, list[SleepRecord]] = {}
        self._alerts: list[dict] = []

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

    def update_fall(self, zone_id: str, detected: bool, confidence: float) -> None:
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.fall_detected = detected
                zs.fall_confidence = confidence
                zs.last_seen = time.time()

    def update_vitals(
        self, zone_id: str,
        respiration_bpm: Optional[float], respiration_confidence: Optional[float],
        heart_rate_bpm: Optional[float], heart_rate_confidence: Optional[float],
    ) -> None:
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.respiration_bpm = respiration_bpm
                zs.respiration_confidence = respiration_confidence
                zs.heart_rate_bpm = heart_rate_bpm
                zs.heart_rate_confidence = heart_rate_confidence

            record = VitalsRecord(
                timestamp=time.time(),
                respiration_bpm=respiration_bpm,
                respiration_confidence=respiration_confidence,
                heart_rate_bpm=heart_rate_bpm,
                heart_rate_confidence=heart_rate_confidence,
            )
            self._vitals_history.setdefault(zone_id, deque(maxlen=MAX_HISTORY)).append(record)

    def update_sleep(self, zone_id: str, stage: str, score: Optional[float] = None) -> None:
        with self._lock:
            zs = self._zone_status.get(zone_id)
            if zs:
                zs.sleep_stage = stage
                zs.sleep_score = score
                zs.last_seen = time.time()

    def add_sleep_record(self, zone_id: str, record: SleepRecord) -> None:
        with self._lock:
            records = self._sleep_records.setdefault(zone_id, [])
            records.append(record)
            if len(records) > MAX_SLEEP_RECORDS:
                records.pop(0)

    def add_alert(self, alert: dict) -> None:
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > MAX_ALERTS:
                self._alerts = self._alerts[-MAX_ALERTS:]

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


# Global singleton
store = InferenceDataStore()
