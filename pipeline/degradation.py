"""
System Degradation Manager

Defines explicit fallback paths for every critical component so the
system never crashes — it degrades gracefully.

Degradation Levels:
  FULL      - All systems operational
  DEGRADED  - One or more components running on fallback
  MINIMAL   - Only basic detection active
  OFFLINE   - System cannot function

Component fallback chains:
  1. Vital Signs: Rust wifi_densepose → Python scipy FFT → "unavailable"
  2. Fall Detection: PyTorch model → rule-based fallback → "unavailable"
  3. Sleep Monitor: SleepLSTM → heuristic classifier → "unavailable"
  4. Activity Detection: always rule-based (no fallback needed)
  5. Alert Dispatch: Telegram → Webhook → local log only
  6. MQTT Transport: connected → file replay → "offline"
  7. Persistence: SQLite → in-memory only → "no persistence"
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    FULL = "full"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    OFFLINE = "offline"


class ComponentMode(Enum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


@dataclass
class ComponentStatus:
    name: str
    mode: ComponentMode = ComponentMode.PRIMARY
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: float = 0.0
    degraded_since: float = 0.0

    def mark_error(self, error: str) -> None:
        self.error_count += 1
        self.last_error = error
        self.last_error_time = time.time()

    def degrade(self) -> Optional[str]:
        """Attempt fallback. Returns the new mode description or None if exhausted."""
        if self.mode == ComponentMode.PRIMARY:
            self.mode = ComponentMode.FALLBACK
            self.degraded_since = time.time()
            logger.warning(f"[{self.name}] Degraded: PRIMARY → FALLBACK")
            return f"{self.name}: primary failed, using fallback"
        elif self.mode == ComponentMode.FALLBACK:
            self.mode = ComponentMode.UNAVAILABLE
            self.degraded_since = time.time()
            logger.error(f"[{self.name}] Degraded: FALLBACK → UNAVAILABLE")
            return f"{self.name}: fallback failed, component unavailable"
        else:
            return None

    @property
    def is_healthy(self) -> bool:
        return self.mode == ComponentMode.PRIMARY


class DegradationManager:
    """Tracks per-component health and overall system degradation level."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentStatus] = {
            "vital_signs": ComponentStatus(name="vital_signs"),
            "fall_detection": ComponentStatus(name="fall_detection"),
            "sleep_monitor": ComponentStatus(name="sleep_monitor"),
            "activity_detection": ComponentStatus(name="activity_detection"),
            "alert_dispatch": ComponentStatus(name="alert_dispatch"),
            "mqtt_transport": ComponentStatus(name="mqtt_transport"),
            "persistence": ComponentStatus(name="persistence"),
        }
        self._start_time: float = time.time()

    def mark_error(self, component: str, error: str) -> None:
        comp = self._components.get(component)
        if comp is None:
            return
        comp.mark_error(error)

    def degrade(self, component: str) -> Optional[str]:
        comp = self._components.get(component)
        if comp is None:
            return None
        return comp.degrade()

    @property
    def level(self) -> DegradationLevel:
        """Compute overall system degradation level."""
        unavailable = sum(1 for c in self._components.values() if c.mode == ComponentMode.UNAVAILABLE)
        in_fallback = sum(1 for c in self._components.values() if c.mode == ComponentMode.FALLBACK)

        # MQTT offline means the entire system is OFFLINE
        if self._components["mqtt_transport"].mode == ComponentMode.UNAVAILABLE:
            return DegradationLevel.OFFLINE

        # Fall detection unavailable = system is minimal
        if self._components["fall_detection"].mode == ComponentMode.UNAVAILABLE:
            return DegradationLevel.MINIMAL

        # Multiple unavailable or fallback components
        if unavailable >= 2 or in_fallback >= 3:
            return DegradationLevel.MINIMAL

        # Any fallback active = degraded
        if in_fallback > 0 or unavailable > 0:
            return DegradationLevel.DEGRADED

        return DegradationLevel.FULL

    def get_health_report(self) -> dict:
        """Health report suitable for dashboard / API."""
        components = {}
        for name, comp in self._components.items():
            components[name] = {
                "mode": comp.mode.value,
                "error_count": comp.error_count,
                "last_error": comp.last_error,
                "last_error_time": comp.last_error_time,
                "degraded_since": comp.degraded_since if comp.degraded_since else None,
            }
        return {
            "level": self.level.value,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "components": components,
        }

    def get_component_mode(self, component: str) -> ComponentMode:
        comp = self._components.get(component)
        return comp.mode if comp else ComponentMode.UNAVAILABLE

    def reset_component(self, component: str) -> None:
        comp = self._components.get(component)
        if comp:
            comp.mode = ComponentMode.PRIMARY
            comp.error_count = 0
            comp.last_error = None
            comp.degraded_since = 0.0
            logger.info(f"[{component}] Reset to PRIMARY")

    def reset_all(self) -> None:
        for comp in self._components.values():
            comp.mode = ComponentMode.PRIMARY
            comp.error_count = 0
            comp.last_error = None
            comp.degraded_since = 0.0
        logger.info("All components reset to PRIMARY")


degradation = DegradationManager()
