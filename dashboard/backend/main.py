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

from fastapi import FastAPI, Query
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """<html><head><title>ElderCare Dashboard</title></head>
    <body><h1>ElderCare Dashboard API</h1><p>Version 0.1.0</p>
    <ul><li><a href="/api/zones">/api/zones</a></li>
    <li><a href="/api/vitals">/api/vitals</a></li>
    <li><a href="/api/alerts">/api/alerts</a></li>
    <li><a href="/api/sleep">/api/sleep</a></li>
    <li><a href="/api/health">/api/health</a></li></ul></body></html>"""


@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "healthy", "version": "0.1.0", "timestamp": datetime.now().isoformat()}


@app.get("/api/zones", response_model=list[ZoneStatus])
async def get_zones() -> list[dict]:
    return [{"zone_id": "zone_bedroom", "name": "Phòng ngủ", "activity_state": "still", "respiration_bpm": 15.0, "fall_detected": False, "online": True}]


@app.get("/api/vitals")
async def get_vitals(zone_id: str = Query(...), hours: int = Query(1)) -> list[dict]:
    return []


@app.get("/api/alerts", response_model=list[AlertEntry])
async def get_alerts(limit: int = Query(20)) -> list[dict]:
    return []


@app.get("/api/sleep")
async def get_sleep_scores(zone_id: str = Query(...), days: int = Query(30)) -> list[dict]:
    return []
