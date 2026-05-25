"""Daily summary report generator for ElderCare.

Produces localized morning reports summarizing overnight
activity, sleep quality, vitals, and any alerts from the past 24 hours.
Works with both real data from InferenceDataStore and dummy fallback data.
"""

import logging
import random
from datetime import datetime
from typing import Optional

from alerts.i18n import locale
from pipeline.data_store import store as data_store

logger = logging.getLogger(__name__)


class DailySummaryGenerator:
    """Generates localized daily summary reports from the data store."""

    def __init__(self) -> None:
        self._last_report_date: Optional[str] = None

    def generate(self, dummy: bool = False) -> str:
        """Generate morning summary in the active locale.

        Args:
            dummy: If True, generates plausible dummy data when the data store
                   is empty (for testing/verification).

        Returns:
            Formatted text ready for Telegram or API display.
        """
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        days = locale.t("daily_summary.days")
        day_of_week = days[now.weekday()] if isinstance(days, list) else now.strftime("%A")
        self._last_report_date = date_str

        separator = locale.t("daily_summary.separator")

        lines: list[str] = []
        lines.append(locale.t("daily_summary.header"))
        lines.append(f"{day_of_week}, {date_str}")
        lines.append(separator)

        zone_statuses = data_store.get_all_zone_statuses()
        alerts = data_store.get_alerts(n=100)

        if not zone_statuses and dummy:
            zone_statuses = self._generate_dummy_zones()
            lines.append(locale.t("daily_summary.dummy_notice"))
            lines.append("")

        # Filter alerts from last 24h
        cutoff = now.timestamp() - 86400
        recent_alerts = [a for a in alerts if a.get("timestamp", 0) >= cutoff]

        # --- Section 1: night summary ---
        lines.append("")
        lines.append(locale.t("daily_summary.section_night"))

        any_sleep = False
        for zs in zone_statuses:
            sleep_records = data_store.get_sleep_records(zs.zone_id, n=1)
            if sleep_records:
                sr = sleep_records[-1]
                sleep_label = self._sleep_quality_label(sr["sleep_score"])
                score_fmt = f"{sr['sleep_score']:.0f}"
                deep_fmt = f"{sr['deep_pct']:.0f}"
                light_fmt = f"{sr['light_pct']:.0f}"
                awake_fmt = f"{sr['awake_pct']:.0f}"
                lines.append(f"  {self._zone_icon(zs.zone_id)} {zs.name}:")
                lines.append(f"     {locale.t('daily_summary.sleep_score_label', score=score_fmt, quality=sleep_label)}")
                lines.append(f"     {locale.t('daily_summary.sleep_breakdown', deep=deep_fmt, light=light_fmt, awake=awake_fmt)}")
                any_sleep = True
            elif dummy:
                any_sleep = True
                score = max(20, min(98, round(random.gauss(72, 12))))
                deep = max(5, min(60, round(random.gauss(35, 10))))
                light = max(20, min(70, round(random.gauss(45, 10))))
                awake = max(0, 100 - deep - light)
                sleep_label = self._sleep_quality_label(score)
                lines.append(f"  {self._zone_icon(zs.zone_id)} {zs.name}:")
                lines.append(f"     {locale.t('daily_summary.sleep_score_label', score=str(score), quality=sleep_label)}")
                lines.append(f"     {locale.t('daily_summary.sleep_breakdown', deep=str(deep), light=str(light), awake=str(awake))}")

        if not zone_statuses and dummy:
            for zone_id, name in [("zone_bedroom", "Phong Ngu"), ("zone_living", "Phong Khach")]:
                score = max(20, min(98, round(random.gauss(72, 12))))
                sleep_label = self._sleep_quality_label(score)
                lines.append(f"  {self._zone_icon(zone_id)} {name}:")
                lines.append(f"     {locale.t('daily_summary.sleep_score_label', score=str(score), quality=sleep_label)}")

        if not any_sleep and not dummy:
            lines.append(f"  {locale.t('daily_summary.no_sleep_data')}")

        # --- Section 2: current status ---
        lines.append("")
        lines.append(locale.t("daily_summary.section_current"))

        na = locale.t("daily_summary.value_not_available")

        for zs in zone_statuses:
            activity_label = locale.t(f"zone_states.{zs.activity_state}")
            resp_val = f"{zs.respiration_bpm:.0f}" if zs.respiration_bpm else na
            hr_val = f"{zs.heart_rate_bpm:.0f}" if zs.heart_rate_bpm else na
            resp_str = locale.t("daily_summary.respiration_label", value=resp_val)
            hr_str = locale.t("daily_summary.heart_rate_label", value=hr_val)
            lines.append(f"  {self._zone_icon(zs.zone_id)} {zs.name}: {activity_label}")
            lines.append(f"     {resp_str} | {hr_str}")
            if zs.fall_detected:
                lines.append(f"     {locale.t('alerts.fall_detected_format', confidence=zs.fall_confidence)}")

        if not zone_statuses and dummy:
            for zone_id, name in [("zone_bedroom", "Phong Ngu"), ("zone_living", "Phong Khach")]:
                lines.append(f"  {self._zone_icon(zone_id)} {name}: {locale.t('zone_states.still')}")
                resp = round(random.uniform(12, 18), 1)
                hr = round(random.uniform(60, 80), 1)
                resp_str = locale.t("daily_summary.respiration_label", value=str(resp))
                hr_str = locale.t("daily_summary.heart_rate_label", value=str(hr))
                lines.append(f"     {resp_str} | {hr_str}")

        # --- Section 3: alert summary ---
        lines.append("")
        lines.append(locale.t("daily_summary.section_alerts"))

        if recent_alerts:
            emergency_count = sum(1 for a in recent_alerts if a.get("level") == "EMERGENCY")
            warning_count = sum(1 for a in recent_alerts if a.get("level") == "WARNING")
            info_count = sum(1 for a in recent_alerts if a.get("level") == "INFO")

            lines.append(f"  {locale.t('daily_summary.alert_counts', emergency=emergency_count, warning=warning_count, info=info_count)}")

            if emergency_count > 0:
                lines.append(f"  {locale.t('daily_summary.emergency_events')}")
                for a in recent_alerts:
                    if a.get("level") == "EMERGENCY":
                        ts = datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M")
                        lines.append(f"     [{ts}] {a['zone_name']}: {a.get('description', '')}")
        elif dummy:
            lines.append(f"  {locale.t('daily_summary.dummy_alert_counts')}")
            lines.append(f"  {locale.t('daily_summary.dummy_warning_detail')}")
        else:
            lines.append(f"  {locale.t('daily_summary.no_alerts')}")

        # --- Section 4: daily advice ---
        lines.append("")
        lines.append(locale.t("daily_summary.section_advice"))
        lines.append(self._daily_advice(zone_statuses, recent_alerts, dummy))

        # --- Footer ---
        lines.append("")
        lines.append(separator)
        lines.append(locale.t("daily_summary.footer_line"))
        lines.append(locale.t("daily_summary.footer_timestamp", time=now.strftime("%H:%M")))

        return "\n".join(lines)

    def _sleep_quality_label(self, score: float) -> str:
        if score >= 80:
            return locale.t("sleep_quality.excellent")
        elif score >= 60:
            return locale.t("sleep_quality.good")
        elif score >= 40:
            return locale.t("sleep_quality.fair")
        else:
            return locale.t("sleep_quality.poor")

    def _zone_icon(self, zone_id: str) -> str:
        icon = locale.t(f"daily_summary.zone_icons.{zone_id}")
        # If key not found, locale.t() returns the key itself
        if icon == f"daily_summary.zone_icons.{zone_id}":
            return locale.t("daily_summary.zone_icons.default")
        return icon

    def _daily_advice(
        self,
        zone_statuses: list,
        recent_alerts: list,
        dummy: bool,
    ) -> str:
        advices: list[str] = []

        has_sleep_data = any(
            data_store.get_sleep_records(zs.zone_id, n=1)
            for zs in zone_statuses
        )

        if has_sleep_data or dummy:
            advices.append(locale.t("daily_summary.advice_sleep_schedule"))

        if any(a.get("level") == "WARNING" for a in recent_alerts) or dummy:
            advices.append(locale.t("daily_summary.advice_check_zone"))

        if any(a.get("level") == "EMERGENCY" for a in recent_alerts):
            advices.append(locale.t("daily_summary.advice_fall_doctor"))

        if not advices:
            advices.append(locale.t("daily_summary.advice_all_normal"))

        advices.append(locale.t("daily_summary.advice_hydration"))
        return "\n".join(advices)

    def _generate_dummy_zones(self) -> list:
        """Generate plausible dummy zone statuses for testing."""
        from pipeline.data_store import ZoneStatus

        return [
            ZoneStatus(
                zone_id="zone_bedroom", name="Phong Ngu",
                activity_state="still", respiration_bpm=15.2, respiration_confidence=0.82,
                heart_rate_bpm=68, heart_rate_confidence=0.45,
                online=True, last_seen=0, sleep_stage="light",
            ),
            ZoneStatus(
                zone_id="zone_living", name="Phong Khach",
                activity_state="active", respiration_bpm=18.1, respiration_confidence=0.75,
                heart_rate_bpm=None, online=True, last_seen=0,
            ),
        ]


def generate_daily_summary(dummy: bool = False) -> str:
    """Convenience function for generating a daily summary report."""
    generator = DailySummaryGenerator()
    return generator.generate(dummy=dummy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_daily_summary(dummy=True))
