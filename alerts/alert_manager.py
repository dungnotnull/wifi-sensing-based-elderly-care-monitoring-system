"""
Alert Manager

Subscribes to inference events, applies threshold logic, cooldown
management, and dispatches alerts to Telegram, Webhook, and local log.

Alert levels:
  - INFO    : Normal daily summary / status update
  - WARNING : Abnormal condition (inactivity, irregular breathing)
  - EMERGENCY: Fall detected / no recovery / life-threatening
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib import request
from urllib.error import URLError

import yaml
from dotenv import load_dotenv

from alerts.i18n import locale

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
        level_key = self.level.value.lower()
        level_label = locale.t(f"alerts.level_{level_key}")
        level_icon = locale.t(f"alerts.icon_{level_key}")

        dt = datetime.fromtimestamp(self.timestamp)
        timestamp_str = dt.strftime("%H:%M:%S %d/%m/%Y")

        return locale.t(
            "alerts.alert_format",
            level_icon=level_icon,
            level_label=level_label,
            zone_name=self.zone_name,
            timestamp=timestamp_str,
            description=self.description,
        )

    def to_webhook_payload(self) -> dict:
        """JSON payload for webhook delivery."""
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "level": self.level.value,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "description": self.description,
            "data": self.data,
        }


class AlertManager:
    """Manages alert generation, cooldown, and dispatch to multiple channels."""

    def __init__(self, config_path: str = "configs/alerts.yaml") -> None:
        self.config = self._load_config(config_path)
        self._last_alert_time: dict[str, float] = {}
        self._telegram_bot: Optional[Any] = None
        self._active_alerts: list[AlertMessage] = []
        self._webhook_session_lock = threading.Lock()
        self._webhook_failures: int = 0
        self._webhook_last_failure: float = 0.0

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
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

        cooldown_config = self.config.get("alert_levels", {}).get(alert.level.value, {})
        cooldown_seconds = cooldown_config.get("cooldown_seconds", 300)

        if cooldown_key in self._last_alert_time:
            elapsed = now - self._last_alert_time[cooldown_key]
            if elapsed < cooldown_seconds:
                logger.debug(
                    f"Alert suppressed by cooldown: {cooldown_key} "
                    f"(elapsed={elapsed:.0f}s < {cooldown_seconds}s)"
                )
                return False

        self._last_alert_time[cooldown_key] = now

        logger.info(f"[{alert.level.value}] {alert.zone_name}: {alert.description}")

        self._dispatch_telegram(alert)
        self._dispatch_webhook(alert)

        self._active_alerts.append(alert)
        if len(self._active_alerts) > 1000:
            self._active_alerts = self._active_alerts[-500:]

        self._log_to_file(alert)
        self._write_to_influxdb(alert)

        return True

    def _dispatch_telegram(self, alert: AlertMessage) -> None:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_ids_str = os.getenv("TELEGRAM_CHAT_ID_PRIMARY", "")

        if not bot_token or not chat_ids_str:
            logger.debug("Telegram not configured — skipping dispatch")
            return

        try:
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

    def _dispatch_webhook(self, alert: AlertMessage) -> None:
        """Send alert to configured webhook endpoint (supports Zalo, Slack, custom integrations).

        Configure in configs/alerts.yaml or via env vars:
          WEBHOOK_URL          - full endpoint URL
          WEBHOOK_HEADERS      - JSON string of extra headers
          WEBHOOK_RETRY_COUNT  - max retry attempts (default 3)
          WEBHOOK_TIMEOUT_SEC  - request timeout in seconds (default 10)
        """
        webhook_url = os.getenv("WEBHOOK_URL") or self.config.get("webhook", {}).get("url")
        if not webhook_url:
            return

        retry_count = int(os.getenv("WEBHOOK_RETRY_COUNT", "3"))
        timeout_sec = int(os.getenv("WEBHOOK_TIMEOUT_SEC", "10"))

        # Back off if we've had recent failures
        if self._webhook_failures >= 5 and (time.time() - self._webhook_last_failure) < 300:
            logger.debug("Webhook circuit breaker open — skipping dispatch")
            return

        payload = json.dumps(alert.to_webhook_payload()).encode("utf-8")

        extra_headers_raw = os.getenv("WEBHOOK_HEADERS") or self.config.get("webhook", {}).get("headers", "{}")
        try:
            extra_headers = json.loads(extra_headers_raw) if isinstance(extra_headers_raw, str) else extra_headers_raw
        except json.JSONDecodeError:
            extra_headers = {}

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ElderCare-AlertManager/0.2",
            **extra_headers,
        }

        for attempt in range(retry_count):
            try:
                req = request.Request(webhook_url, data=payload, headers=headers, method="POST")
                resp = request.urlopen(req, timeout=timeout_sec)
                if 200 <= resp.status < 300:
                    logger.info(f"Webhook alert sent to {webhook_url} (status={resp.status})")
                    self._webhook_failures = 0
                    return
                else:
                    logger.warning(f"Webhook returned status {resp.status} (attempt {attempt+1}/{retry_count})")
            except URLError as e:
                logger.warning(f"Webhook request failed (attempt {attempt+1}/{retry_count}): {e}")
            except Exception:
                logger.exception(f"Webhook unexpected error (attempt {attempt+1}/{retry_count})")

            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)

        self._webhook_failures += 1
        self._webhook_last_failure = time.time()

    def _log_to_file(self, alert: AlertMessage) -> None:
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
        pass

    def get_active_alerts(self, limit: int = 20) -> list[AlertMessage]:
        return self._active_alerts[-limit:]

    def send_info(self, zone_id: str, zone_name: str, description: str, data: Optional[dict] = None) -> bool:
        return self.send_alert(
            AlertMessage(
                zone_id=zone_id, zone_name=zone_name, level=AlertLevel.INFO,
                event_type="info", timestamp=time.time(), description=description,
                data=data or {},
            )
        )

    def send_warning(self, zone_id: str, zone_name: str, description: str, data: Optional[dict] = None) -> bool:
        return self.send_alert(
            AlertMessage(
                zone_id=zone_id, zone_name=zone_name, level=AlertLevel.WARNING,
                event_type="warning", timestamp=time.time(), description=description,
                data=data or {},
            )
        )

    def send_emergency(self, zone_id: str, zone_name: str, description: str, data: Optional[dict] = None) -> bool:
        return self.send_alert(
            AlertMessage(
                zone_id=zone_id, zone_name=zone_name, level=AlertLevel.EMERGENCY,
                event_type="emergency", timestamp=time.time(), description=description,
                data=data or {},
            )
        )

    def generate_daily_summary(self, dummy: bool = False) -> str:
        from alerts.daily_summary import generate_daily_summary
        return generate_daily_summary(dummy=dummy)

    def send_daily_summary(self, dummy: bool = False) -> bool:
        summary = self.generate_daily_summary(dummy=dummy)
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_ids_str = os.getenv("TELEGRAM_CHAT_ID_PRIMARY", "")
        if not bot_token or not chat_ids_str:
            logger.info("Telegram not configured -- summary logged only")
            logger.info("\n%s", summary)
            return False
        try:
            from telegram import Bot
            if self._telegram_bot is None:
                self._telegram_bot = Bot(token=bot_token)
            for chat_id in chat_ids_str.split(","):
                if chat_id.strip():
                    self._telegram_bot.send_message(chat_id=chat_id.strip(), text=summary)
                    logger.info("Daily summary sent to chat_id=%s", chat_id.strip())
            return True
        except Exception:
            logger.exception("Failed to send daily summary via Telegram")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mgr = AlertManager()
    mgr.send_info(
        zone_id="zone_test",
        zone_name="Phong Test",
        description=locale.t("alerts.system_started"),
    )
    print("Test alert dispatched. Check logs.")
