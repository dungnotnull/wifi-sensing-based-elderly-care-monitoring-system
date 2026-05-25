"""
ElderCare Dashboard -- FastAPI Backend

Serves REST API for the React frontend. All data comes from the
shared InferenceDataStore (populated by the inference engine in real time).

/api/zones - zone status (activity, vitals, fall, online)
/api/vitals - respiration/heart rate history per zone
/api/alerts - alert log
/api/sleep - sleep scores per zone
/api/health - service health
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from pipeline.data_store import store

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


class ZoneStatusOut(BaseModel):
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


class AlertEntryOut(BaseModel):
    id: int
    zone_id: str
    zone_name: str
    level: str
    event_type: str
    timestamp: float
    description: str
    dispatched: bool = False
    acknowledged: bool = False


class VitalsPoint(BaseModel):
    timestamp: float
    respiration_bpm: Optional[float] = None
    respiration_confidence: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    heart_rate_confidence: Optional[float] = None


class SleepRecordOut(BaseModel):
    date: str
    sleep_score: float
    awake_pct: float
    light_pct: float
    deep_pct: float
    sleep_efficiency: float


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """<html><head><title>ElderCare Dashboard</title></head>
    <body><h1>ElderCare Dashboard API</h1><p>Version 0.1.0</p>
    <ul><li><a href="/api/zones">/api/zones</a></li>
    <li><a href="/api/vitals">/api/vitals</a></li>
    <li><a href="/api/alerts">/api/alerts</a></li>
    <li><a href="/api/sleep">/api/sleep</a></li>
    <li><a href="/api/health">/api/health</a></li>
    <li><a href="/api/events">/api/events (SSE)</a></li>
    <li><a href="/api/telemetry">/api/telemetry</a></li>
    <li><a href="/api/shadow-mode/report">/api/shadow-mode/report</a></li></ul></body></html>"""


@app.get("/api/health")
async def health_check() -> dict:
    import shutil
    import psutil

    zone_statuses = store.get_all_zone_statuses()
    disk = shutil.disk_usage("/")
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)

    system_health: dict = {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "zones_online": sum(1 for z in zone_statuses if z.online),
        "zones_total": len(zone_statuses),
        "disk_usage_pct": round(disk.used / disk.total * 100, 1),
        "memory_usage_pct": round(mem.percent, 1),
        "cpu_usage_pct": round(cpu, 1),
        "uptime_seconds": 0,
    }

    alerts: list[str] = []
    if system_health["disk_usage_pct"] > 90:
        alerts.append("disk_usage_high")
    if system_health["memory_usage_pct"] > 85:
        alerts.append("memory_usage_high")
    if alerts:
        system_health["status"] = "degraded"
        system_health["alerts"] = alerts

    try:
        from pipeline.watchdog import get_system_health
        system_health.update(get_system_health(
            worker_status=None, mqtt_connected=False, start_time=0.0,
        ))
    except ImportError:
        pass

    return system_health


@app.get("/api/zones", response_model=list[ZoneStatusOut])
async def get_zones() -> list[dict]:
    results = store.get_all_zone_statuses()
    if not results:
        return [{
            "zone_id": "zone_default", "name": "Hệ thống",
            "activity_state": "starting", "online": True,
        }]
    return [
        {
            "zone_id": z.zone_id, "name": z.name,
            "last_seen": z.last_seen, "activity_state": z.activity_state,
            "respiration_bpm": z.respiration_bpm,
            "respiration_confidence": z.respiration_confidence,
            "heart_rate_bpm": z.heart_rate_bpm,
            "heart_rate_confidence": z.heart_rate_confidence,
            "fall_detected": z.fall_detected, "fall_confidence": z.fall_confidence,
            "sleep_stage": z.sleep_stage, "sleep_score": z.sleep_score,
            "online": z.online,
        }
        for z in results
    ]


@app.get("/api/vitals", response_model=list[VitalsPoint])
async def get_vitals(zone_id: str = Query(...), hours: int = Query(1)) -> list[dict]:
    n_points = hours * 720  # 5s update interval = 720 points per hour
    return store.get_vitals_history(zone_id, n=n_points)


@app.get("/api/alerts", response_model=list[AlertEntryOut])
async def get_alerts(limit: int = Query(50)) -> list[dict]:
    alerts = store.get_alerts(n=limit)
    for i, a in enumerate(alerts):
        a.setdefault("id", i)
        a.setdefault("dispatched", True)
        a.setdefault("acknowledged", False)
    return alerts


@app.get("/api/sleep", response_model=list[SleepRecordOut])
async def get_sleep_scores(zone_id: str = Query(...), days: int = Query(30)) -> list[dict]:
    return store.get_sleep_records(zone_id, n=days)


@app.get("/api/daily-summary")
async def get_daily_summary(dummy: bool = Query(False)) -> dict:
    from alerts.daily_summary import generate_daily_summary
    text = generate_daily_summary(dummy=dummy)
    return {"summary": text, "dummy": dummy}


@app.get("/api/telemetry")
async def get_telemetry() -> dict:
    from pipeline.telemetry import telemetry
    return telemetry.get_dashboard_summary()


@app.get("/api/correlation/traces")
async def get_traces(limit: int = Query(100)) -> list[dict]:
    from pipeline.correlation import tracker
    return tracker.get_recent_traces(n=limit)


@app.get("/api/correlation/stats")
async def get_correlation_stats() -> dict:
    from pipeline.correlation import tracker
    return tracker.get_latency_stats()


@app.get("/api/shadow-mode/report")
async def get_shadow_report() -> dict:
    from pipeline.shadow_mode import shadow_mode
    return shadow_mode.generate_report()


@app.post("/api/shadow-mode/go-live")
async def shadow_go_live() -> dict:
    from pipeline.shadow_mode import shadow_mode
    shadow_mode.switch_to_live()
    return {"status": "live", "message": "Alerts are now being sent."}


@app.get("/api/csi-quality/{zone_id}")
async def get_csi_quality(zone_id: str) -> dict:
    quality = store.get_csi_quality(zone_id)
    if quality is None:
        return {"zone_id": zone_id, "status": "no_data"}
    return quality


@app.get("/api/adaptive-thresholds")
async def get_adaptive_thresholds() -> dict:
    return {
        "message": "Thresholds are managed by AdaptiveThresholdManager in the inference engine.",
        "config_path": "configs/thresholds.yaml",
    }


@app.get("/api/events")
async def sse_events() -> StreamingResponse:
    """Server-Sent Events endpoint for real-time dashboard updates.

    Pushes zone status, fall alerts, vitals, and activity changes
    as they occur. Falls back to 5-second polling of data store.
    """
    async def event_generator():
        last_alert_count = 0
        while True:
            data: dict = {"zones": [], "alerts": []}

            zone_statuses = store.get_all_zone_statuses()
            for z in zone_statuses:
                data["zones"].append({
                    "zone_id": z.zone_id, "name": z.name,
                    "activity_state": z.activity_state, "online": z.online,
                    "respiration_bpm": z.respiration_bpm,
                    "heart_rate_bpm": z.heart_rate_bpm,
                    "fall_detected": z.fall_detected,
                    "fall_confidence": z.fall_confidence,
                    "sleep_stage": z.sleep_stage, "sleep_score": z.sleep_score,
                    "last_seen": z.last_seen,
                })

            alerts = store.get_alerts(n=5)
            if len(alerts) != last_alert_count:
                data["new_alerts"] = alerts[-(len(alerts) - last_alert_count):]
            last_alert_count = len(alerts)
            data["alerts"] = alerts

            yield f"data: {json.dumps(data, default=str)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
