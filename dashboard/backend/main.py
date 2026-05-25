"""
ElderCare Dashboard — FastAPI Backend

Provides REST API for the React frontend:
  - Current zone status
  - Vital sign history
  - Alert log
  - Sleep scores
  - Daily reports

All data stays local — no cloud dependency.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ElderCare Dashboard API",
    version="0.1.0",
    description="Privacy-preserving WiFi-based elderly monitoring system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local network only in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class ZoneStatus(BaseModel):
    zone_id: str
    name: str
    last_seen: Optional[float] = None
    activity_state: str = "unknown"
    respiration_bpm: Optional[float] = None
    fall_detected: bool = False
    online: bool = False


class AlertEntry(BaseModel):
    id: int
    zone_id: str
    zone_name: str
    level: str
    event_type: str
    timestamp: float
    description: str
    acknowledged: bool = False


class VitalSignPoint(BaseModel):
    timestamp: float
    respiration_bpm: float
    heart_rate_bpm: Optional[float] = None


class SleepScoreEntry(BaseModel):
    date: str
    score: float
    deep_pct: float
    light_pct: float
    awake_pct: float
    sleep_efficiency: float


class SystemHealth(BaseModel):
    uptime_seconds: float
    active_zones: int
    packets_per_second: float
    inference_latency_ms: float
    disk_usage_pct: float


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """
    <html>
        <head><title>ElderCare Dashboard</title></head>
        <body>
            <h1>🏥 ElderCare Dashboard API</h1>
            <p>Version 0.1.0 — Running</p>
            <ul>
                <li><a href="/api/zones">/api/zones</a> — Zone status</li>
                <li><a href="/api/vitals">/api/vitals</a> — Vital signs history</li>
                <li><a href="/api/alerts">/api/alerts</a> — Alert log</li>
                <li><a href="/api/sleep">/api/sleep</a> — Sleep scores</li>
                <li><a href="/api/health">/api/health</a> — System health</li>
            </ul>
        </body>
    </html>
    """


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/zones", response_model=list[ZoneStatus])
async def get_zones(active_only: bool = Query(False)) -> list[dict]:
    """Get current status for all zones."""
    # Placeholder — Phase 3+ will read from live inference state
    return [
        {
            "zone_id": "zone_bedroom",
            "name": "Phòng ngủ",
            "last_seen": datetime.now().timestamp(),
            "activity_state": "still",
            "respiration_bpm": 15.0,
            "fall_detected": False,
            "online": True,
        },
        {
            "zone_id": "zone_living",
            "name": "Phòng khách",
            "last_seen": datetime.now().timestamp(),
            "activity_state": "active",
            "respiration_bpm": None,
            "fall_detected": False,
            "online": True,
        },
    ]


@app.get("/api/zones/{zone_id}", response_model=ZoneStatus)
async def get_zone(zone_id: str) -> dict:
    """Get status for a specific zone."""
    # Placeholder
    return {
        "zone_id": zone_id,
        "name": f"Zone {zone_id}",
        "last_seen": datetime.now().timestamp(),
        "activity_state": "unknown",
        "respiration_bpm": None,
        "fall_detected": False,
        "online": True,
    }


@app.get("/api/vitals", response_model=list[VitalSignPoint])
async def get_vitals(
    zone_id: str = Query(...),
    hours: int = Query(1, ge=1, le=24),
) -> list[dict]:
    """Get vital signs history for a zone."""
    # Placeholder — Phase 3+ reads from InfluxDB
    return []


@app.get("/api/alerts", response_model=list[AlertEntry])
async def get_alerts(
    zone_id: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    unacknowledged_only: bool = False,
) -> list[dict]:
    """Get alert history with optional filtering."""
    # Placeholder — Phase 3+ reads from InfluxDB / log file
    return []


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int) -> dict:
    """Acknowledge an alert (mark as seen by caregiver)."""
    # Placeholder
    return {"alert_id": alert_id, "acknowledged": True}


@app.get("/api/sleep", response_model=list[SleepScoreEntry])
async def get_sleep_scores(
    zone_id: str = Query(...),
    days: int = Query(30, ge=1, le=90),
) -> list[dict]:
    """Get sleep score history."""
    # Placeholder — Phase 4
    return []


@app.get("/api/health/system", response_model=SystemHealth)
async def get_system_health() -> dict:
    """Get full system health metrics."""
    # Placeholder
    import time
    return {
        "uptime_seconds": 0.0,
        "active_zones": 0,
        "packets_per_second": 0.0,
        "inference_latency_ms": 0.0,
        "disk_usage_pct": 0.0,
    }
