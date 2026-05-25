"""
SQLite write-behind persistence for ElderCare inference state.

Accumulates writes in a thread-safe buffer and flushes to SQLite
periodically (every 60 seconds) or on explicit flush/shutdown.
On startup, restore() recovers the last known state from disk.
"""

import logging
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS zone_status (
    zone_id TEXT PRIMARY KEY,
    name TEXT,
    activity_state TEXT DEFAULT 'unknown',
    respiration_bpm REAL,
    respiration_confidence REAL,
    heart_rate_bpm REAL,
    heart_rate_confidence REAL,
    fall_detected INTEGER DEFAULT 0,
    fall_confidence REAL DEFAULT 0.0,
    sleep_stage TEXT DEFAULT 'unknown',
    sleep_score REAL,
    online INTEGER DEFAULT 0,
    last_seen REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS vitals_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    respiration_bpm REAL,
    respiration_confidence REAL,
    heart_rate_bpm REAL,
    heart_rate_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_vitals_zone_time ON vitals_history(zone_id, timestamp);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id TEXT NOT NULL,
    zone_name TEXT,
    level TEXT NOT NULL,
    event_type TEXT,
    timestamp REAL NOT NULL,
    description TEXT,
    dispatched INTEGER DEFAULT 0,
    acknowledged INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sleep_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id TEXT NOT NULL,
    date TEXT NOT NULL,
    sleep_score REAL,
    awake_pct REAL,
    light_pct REAL,
    deep_pct REAL,
    sleep_efficiency REAL
);
"""

_FLUSH_INTERVAL = 60.0


class SQLitePersistence:
    """Write-behind SQLite persistence layer.

    Writes are buffered and flushed to disk every 60 seconds or on
    explicit flush(). All public methods are thread-safe.
    """

    def __init__(self, db_path: str = "data/eldercare.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._buffer: list[tuple[str, dict]] = []
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._init_db()

    # -- Connection management --

    def _init_db(self) -> None:
        """Create DB file, enable WAL mode, create tables."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("SQLite persistence initialized at %s (WAL mode)", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "DB connection not initialized"
        return self._conn

    # -- Background flush --

    def start_flush_timer(self) -> None:
        """Start the periodic background flush thread."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, name="sqlite-persistence-flush", daemon=True,
        )
        self._flush_thread.start()
        logger.info("SQLite persistence flush timer started (interval=%ss)", _FLUSH_INTERVAL)

    def stop_flush_timer(self) -> None:
        """Stop the background flush thread."""
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(_FLUSH_INTERVAL):
            try:
                self.flush()
            except Exception:
                logger.exception("Error during periodic SQLite flush")

    # -- Buffer operations --

    def buffer_zone_status(self, zone_id: str, fields: dict) -> None:
        """Buffer a zone_status upsert."""
        with self._lock:
            self._buffer.append(("zone_status", {"zone_id": zone_id, **fields}))

    def buffer_vitals(self, zone_id: str, record: dict) -> None:
        """Buffer a vitals_history insert."""
        with self._lock:
            self._buffer.append(("vitals", {"zone_id": zone_id, **record}))

    def buffer_alert(self, alert: dict) -> None:
        """Buffer an alerts insert."""
        with self._lock:
            self._buffer.append(("alert", dict(alert)))

    def buffer_sleep_record(self, zone_id: str, record: dict) -> None:
        """Buffer a sleep_records insert."""
        with self._lock:
            self._buffer.append(("sleep_record", {"zone_id": zone_id, **record}))

    # -- Flush --

    def flush(self) -> None:
        """Flush all buffered writes to SQLite."""
        with self._lock:
            if not self._buffer:
                return
            pending = self._buffer[:]
            self._buffer.clear()

        conn = self._get_conn()
        try:
            self._write_batch(conn, pending)
            conn.commit()
            logger.debug("Flushed %d buffered writes to SQLite", len(pending))
        except Exception:
            # Re-queue failed writes at the front so they are retried next flush
            with self._lock:
                self._buffer = pending + self._buffer
            logger.exception("Failed to flush %d writes to SQLite", len(pending))

    @staticmethod
    def _write_batch(conn: sqlite3.Connection, items: list[tuple[str, dict]]) -> None:
        for kind, data in items:
            if kind == "zone_status":
                conn.execute(
                    """
                    INSERT INTO zone_status (
                        zone_id, name, activity_state, respiration_bpm,
                        respiration_confidence, heart_rate_bpm, heart_rate_confidence,
                        fall_detected, fall_confidence, sleep_stage, sleep_score,
                        online, last_seen, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(zone_id) DO UPDATE SET
                        name=COALESCE(excluded.name, zone_status.name),
                        activity_state=COALESCE(excluded.activity_state, zone_status.activity_state),
                        respiration_bpm=COALESCE(excluded.respiration_bpm, zone_status.respiration_bpm),
                        respiration_confidence=COALESCE(excluded.respiration_confidence, zone_status.respiration_confidence),
                        heart_rate_bpm=COALESCE(excluded.heart_rate_bpm, zone_status.heart_rate_bpm),
                        heart_rate_confidence=COALESCE(excluded.heart_rate_confidence, zone_status.heart_rate_confidence),
                        fall_detected=COALESCE(excluded.fall_detected, zone_status.fall_detected),
                        fall_confidence=COALESCE(excluded.fall_confidence, zone_status.fall_confidence),
                        sleep_stage=COALESCE(excluded.sleep_stage, zone_status.sleep_stage),
                        sleep_score=COALESCE(excluded.sleep_score, zone_status.sleep_score),
                        online=COALESCE(excluded.online, zone_status.online),
                        last_seen=COALESCE(excluded.last_seen, zone_status.last_seen),
                        updated_at=excluded.updated_at
                    """,
                    (
                        data.get("zone_id"), data.get("name"),
                        data.get("activity_state"), data.get("respiration_bpm"),
                        data.get("respiration_confidence"), data.get("heart_rate_bpm"),
                        data.get("heart_rate_confidence"),
                        data.get("fall_detected"), data.get("fall_confidence"),
                        data.get("sleep_stage"), data.get("sleep_score"),
                        data.get("online"), data.get("last_seen"),
                        data.get("updated_at", time.time()),
                    ),
                )
            elif kind == "vitals":
                conn.execute(
                    """
                    INSERT INTO vitals_history (zone_id, timestamp, respiration_bpm,
                        respiration_confidence, heart_rate_bpm, heart_rate_confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data.get("zone_id"), data.get("timestamp"),
                        data.get("respiration_bpm"), data.get("respiration_confidence"),
                        data.get("heart_rate_bpm"), data.get("heart_rate_confidence"),
                    ),
                )
            elif kind == "alert":
                conn.execute(
                    """
                    INSERT INTO alerts (zone_id, zone_name, level, event_type,
                        timestamp, description, dispatched, acknowledged)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data.get("zone_id"), data.get("zone_name"),
                        data.get("level"), data.get("event_type"),
                        data.get("timestamp"), data.get("description"),
                        data.get("dispatched", 0), data.get("acknowledged", 0),
                    ),
                )
            elif kind == "sleep_record":
                conn.execute(
                    """
                    INSERT INTO sleep_records (zone_id, date, sleep_score,
                        awake_pct, light_pct, deep_pct, sleep_efficiency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data.get("zone_id"), data.get("date"),
                        data.get("sleep_score"), data.get("awake_pct"),
                        data.get("light_pct"), data.get("deep_pct"),
                        data.get("sleep_efficiency"),
                    ),
                )

    # -- Restore --

    def restore(self) -> dict:
        """Load last known state from SQLite.

        Returns a dict with keys:
          zone_statuses: list[dict]
          vitals_history: dict[str, list[dict]]
          alerts: list[dict]
          sleep_records: dict[str, list[dict]]
        """
        conn = self._get_conn()
        zone_statuses = []
        try:
            rows = conn.execute(
                "SELECT zone_id, name, activity_state, respiration_bpm, "
                "respiration_confidence, heart_rate_bpm, heart_rate_confidence, "
                "fall_detected, fall_confidence, sleep_stage, sleep_score, "
                "online, last_seen, updated_at "
                "FROM zone_status ORDER BY updated_at DESC"
            ).fetchall()
            for r in rows:
                zone_statuses.append({
                    "zone_id": r[0], "name": r[1], "activity_state": r[2],
                    "respiration_bpm": r[3], "respiration_confidence": r[4],
                    "heart_rate_bpm": r[5], "heart_rate_confidence": r[6],
                    "fall_detected": bool(r[7]), "fall_confidence": r[8],
                    "sleep_stage": r[9], "sleep_score": r[10],
                    "online": bool(r[11]), "last_seen": r[12],
                })
        except Exception:
            logger.exception("Error restoring zone_status from SQLite")

        vitals_history: dict[str, list[dict]] = {}
        try:
            # Restore last 3600 vitals points per zone (matching MAX_HISTORY)
            rows = conn.execute(
                "SELECT zone_id, timestamp, respiration_bpm, respiration_confidence, "
                "heart_rate_bpm, heart_rate_confidence "
                "FROM vitals_history ORDER BY timestamp ASC"
            ).fetchall()
            for r in rows:
                vitals_history.setdefault(r[0], []).append({
                    "timestamp": r[1], "respiration_bpm": r[2],
                    "respiration_confidence": r[3],
                    "heart_rate_bpm": r[4], "heart_rate_confidence": r[5],
                })
        except Exception:
            logger.exception("Error restoring vitals_history from SQLite")

        alerts: list[dict] = []
        try:
            rows = conn.execute(
                "SELECT zone_id, zone_name, level, event_type, timestamp, "
                "description, dispatched, acknowledged "
                "FROM alerts ORDER BY timestamp DESC LIMIT 500"
            ).fetchall()
            for r in rows:
                alerts.append({
                    "zone_id": r[0], "zone_name": r[1], "level": r[2],
                    "event_type": r[3], "timestamp": r[4], "description": r[5],
                    "dispatched": bool(r[6]), "acknowledged": bool(r[7]),
                })
            alerts.reverse()  # chronological order (oldest first)
        except Exception:
            logger.exception("Error restoring alerts from SQLite")

        sleep_records: dict[str, list[dict]] = {}
        try:
            rows = conn.execute(
                "SELECT zone_id, date, sleep_score, awake_pct, light_pct, "
                "deep_pct, sleep_efficiency "
                "FROM sleep_records ORDER BY date ASC"
            ).fetchall()
            for r in rows:
                sleep_records.setdefault(r[0], []).append({
                    "date": r[1], "sleep_score": r[2], "awake_pct": r[3],
                    "light_pct": r[4], "deep_pct": r[5], "sleep_efficiency": r[6],
                })
        except Exception:
            logger.exception("Error restoring sleep_records from SQLite")

        logger.info(
            "Restored from SQLite: %d zones, %d vitals points, %d alerts, %d sleep records",
            len(zone_statuses),
            sum(len(v) for v in vitals_history.values()),
            len(alerts),
            sum(len(v) for v in sleep_records.values()),
        )
        return {
            "zone_statuses": zone_statuses,
            "vitals_history": vitals_history,
            "alerts": alerts,
            "sleep_records": sleep_records,
        }

    # -- Lifecycle --

    def close(self) -> None:
        """Flush remaining writes and close the database connection."""
        self.stop_flush_timer()
        try:
            self.flush()
        except Exception:
            logger.exception("Error during final flush on close")
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("SQLite persistence closed")
