"""
Alert Manager

Subscribes to inference events, applies threshold logic, cooldown
management, and dispatches alerts to Telegram and local log/InfluxDB.

Alert levels:
  - INFO    : Normal daily summary / status update
  - WARNING : Abnormal condition (inactivity, irregular breathing)
  - EMERGENCY: Fall detected / no recovery / life-threatening
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    EMERGENCY = "EMERGENCY"


@dataclass
class AlertMessage:
    zone_id: str
    zone_name: str
    level: AlertLevel
    event_type: str
    timestamp: float
    description: str
    data: dict[str, Any] = field(default_factory=dict)

    def format_vn(self) -> str:
        """Format alert message in Vietnamese."""
        level_labels = {
            AlertLevel.INFO: "ℹ️ THÔNG BÁO",
            AlertLevel.WARNING: "⚠️ CẢNH BÁO",
            AlertLevel.EMERGENCY: "🚨 KHẨN CẤP",
        }

        dt = datetime.fromtimestamp(self.timestamp)
        return (
            f"{level_labels[self.level]}\n"
            f"📍 Khu vực: {self.zone_name}\n"
            f"⏰ Thời gian: {dt.strftime('%H:%M:%S %d/%m/%Y')}\n"
            f"📋 Sự kiện: {self.description}"
        )


class AlertManager:
    """Manages alert generation, cooldown, and dispatch."""

    def __init__(self, config_path: str = "configs/alerts.yaml") -> None:
        self.config = self._load_config(config_path)
        self._last_alert_time: dict[str, float] = {}  # keyed by (level, zone)
        self._telegram_bot: Optional[Any] = None
        self._active_alerts: list[AlertMessage] = []

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = f.read()
                # Resolve env vars in template ${VAR}
                raw = os.path.expandvars(raw)
                return yaml.safe_load(raw)
        logger.warning(f"Alert config not found: {path}. Using defaults.")
        return {
            "alert_levels": {
                "INFO": {"cooldown_seconds": 3600},
                "WARNING": {"cooldown_seconds": 300},
                "EMERGENCY": {"cooldown_seconds": 60},
            },
        }

    def send_alert(self, alert: AlertMessage) -> bool:
        """Send alert with cooldown check. Returns True if alert was dispatched."""
        cooldown_key = f"{alert.level.value}_{alert.zone_id}"
        now = time.time()

        # Cooldown check
        cooldown_config = self.config.get("alert_levels", {}).get(alert.level.value, {})
        cooldown_seconds = cooldown_config.get("cooldown_seconds", 300)

        if cooldown_key in self._last_alert_time:
            elapsed = now - self._last_alert_time[cooldown_key]
            if elapsed < cooldown_seconds:
                logger.debug(
                    f"Alert suppressed by cooldown: {cooldown_key} (elapsed={elapsed:.0f}s < {cooldown_seconds}s)"
                )
                return False

        self._last_alert_time[cooldown_key] = now

        # Log the alert
        logger.info(f"[{alert.level.value}] {alert.zone_name}: {alert.description}")

        # Dispatch to Telegram
        self._dispatch_telegram(alert)

        # Store in active alerts
        self._active_alerts.append(alert)
        if len(self._active_alerts) > 1000:
            self._active_alerts = self._active_alerts[-500:]

        # Write to log file
        self._log_to_file(alert)

        # Write to InfluxDB (if configured)
        self._write_to_influxdb(alert)

        return True

    def _dispatch_telegram(self, alert: AlertMessage) -> None:
        """Send alert via Telegram bot."""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_ids_str = os.getenv("TELEGRAM_CHAT_ID_PRIMARY", "")

        if not bot_token or not chat_ids_str:
            logger.debug("Telegram not configured — skipping dispatch")
            return

        try:
            # Lazy import to keep dependency optional
            from telegram import Bot

            if self._telegram_bot is None:
                self._telegram_bot = Bot(token=bot_token)

            message = alert.format_vn()
            for chat_id in chat_ids_str.split(","):
                chat_id = chat_id.strip()
                if chat_id:
                    self._telegram_bot.send_message(chat_id=chat_id, text=message)
                    logger.info(f"Telegram alert sent to chat_id={chat_id}")
        except Exception:
            logger.exception("Failed to send Telegram alert")

    def _log_to_file(self, alert: AlertMessage) -> None:
        """Persist alert to local log."""
        try:
            log_dir = "data"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "alerts.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.fromtimestamp(alert.timestamp).isoformat()} | "
                    f"{alert.level.value:9s} | {alert.zone_id:20s} | "
                    f"{alert.event_type:30s} | {alert.description}\n"
                )
        except Exception:
            logger.exception("Failed to write alert to log file")

    def _write_to_influxdb(self, alert: AlertMessage) -> None:
        """Write alert event to InfluxDB (placeholder — Phase 3+)."""
        # Will be implemented in Phase 3 when InfluxDB is integrated
        pass

    def get_active_alerts(self, limit: int = 20) -> list[AlertMessage]:
        return self._active_alerts[-limit:]

    def send_info(self, zone_id: str, zone_name: str, description: str, data: Optional[dict] = None) -> bool:
        return self.send_alert(
            AlertMessage(
                zone_id=zone_id,
                zone_name=zone_name,
                level=AlertLevel.INFO,
                event_type="info",
                timestamp=time.time(),
                description=description,
                data=data or {},
            )
        )

    def send_warning(self, zone_id: str, zone_name: str, description: str, data: Optional[dict] = None) -> bool:
        return self.send_alert(
            AlertMessage(
                zone_id=zone_id,
                zone_name=zone_name,
                level=AlertLevel.WARNING,
                event_type="warning",
                timestamp=time.time(),
                description=description,
                data=data or {},
            )
        )

    def send_emergency(self, zone_id: str, zone_name: str, description: str, data: Optional[dict] = None) -> bool:
        return self.send_alert(
            AlertMessage(
                zone_id=zone_id,
                zone_name=zone_name,
                level=AlertLevel.EMERGENCY,
                event_type="emergency",
                timestamp=time.time(),
                description=description,
                data=data or {},
            )
        )

    def generate_daily_summary(self) -> str:
        """Generate a daily summary report (placeholder — Phase 3+)."""
        return "📊 Báo cáo hàng ngày ElderCare\n(Chức năng đang phát triển)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mgr = AlertManager()
    mgr.send_info(
        zone_id="zone_test",
        zone_name="Phòng Test",
        description="Hệ thống Alert Manager đã khởi động",
    )
    print("Test alert dispatched. Check logs.")
