"""Daily summary report generator for ElderCare.

Produces Vietnamese-language morning reports summarizing overnight
activity, sleep quality, vitals, and any alerts from the past 24 hours.
Works with both real data from InferenceDataStore and dummy fallback data.
"""

import logging
import random
from datetime import datetime
from typing import Optional

from pipeline.data_store import store as data_store

logger = logging.getLogger(__name__)

STAGE_LABELS_VN: dict[str, str] = {
    "awake": "thức",
    "light": "ngủ nông",
    "deep": "ngủ sâu",
    "unknown": "không rõ",
}

ACTIVITY_LABELS_VN: dict[str, str] = {
    "active": "hoạt động",
    "still": "nghỉ ngơi",
    "inactivity": "không hoạt động",
    "unknown": "không rõ",
}


class DailySummaryGenerator:
    """Generates Vietnamese daily summary reports from the data store."""

    def __init__(self) -> None:
        self._last_report_date: Optional[str] = None

    def generate(self, dummy: bool = False) -> str:
        """Generate Vietnamese morning summary.

        Args:
            dummy: If True, generates plausible dummy data when the data store
                   is empty (for testing/verification).

        Returns:
            Formatted Vietnamese text ready for Telegram or API display.
        """
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        day_of_week = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][now.weekday()]
        self._last_report_date = date_str

        lines: list[str] = []
        lines.append(f"📊 BÁO CÁO HÀNG NGÀY ELDERCARE")
        lines.append(f"{day_of_week}, {date_str}")
        lines.append("─" * 28)

        zone_statuses = data_store.get_all_zone_statuses()
        alerts = data_store.get_alerts(n=100)

        if not zone_statuses and dummy:
            zone_statuses = self._generate_dummy_zones()
            lines.append("⚠️ Dữ liệu mẫu — hệ thống chưa có dữ liệu thực tế")
            lines.append("")

        # Filter alerts from last 24h
        cutoff = now.timestamp() - 86400
        recent_alerts = [a for a in alerts if a.get("timestamp", 0) >= cutoff]

        # --- Section 1: night summary ---
        lines.append("")
        lines.append("🌙 ĐÊM QUA")

        any_sleep = False
        for zs in zone_statuses:
            sleep_records = data_store.get_sleep_records(zs.zone_id, n=1)
            if sleep_records:
                sr = sleep_records[-1]
                sleep_label = self._sleep_quality_label(sr["sleep_score"])
                lines.append(f"  {self._zone_icon(zs.zone_id)} {zs.name}:")
                lines.append(f"     Điểm giấc ngủ: {sr['sleep_score']:.0f}/100 — {sleep_label}")
                lines.append(f"     Ngủ sâu: {sr['deep_pct']:.0f}% | Ngủ nông: {sr['light_pct']:.0f}% | Thức: {sr['awake_pct']:.0f}%")
                any_sleep = True
            elif dummy:
                any_sleep = True
                score = round(random.gauss(72, 12))
                score = max(20, min(98, score))
                deep = round(random.gauss(35, 10))
                deep = max(5, min(60, deep))
                light = round(random.gauss(45, 10))
                light = max(20, min(70, light))
                awake = 100 - deep - light
                if awake < 0:
                    awake = 0
                sleep_label = self._sleep_quality_label(score)
                lines.append(f"  {self._zone_icon(zs.zone_id)} {zs.name}:")
                lines.append(f"     Điểm giấc ngủ: {score}/100 — {sleep_label}")
                lines.append(f"     Ngủ sâu: {deep}% | Ngủ nông: {light}% | Thức: {awake}%")

        if not zone_statuses and dummy:
            for zone_id, name in [("zone_bedroom", "Phòng ngủ"), ("zone_living", "Phòng khách")]:
                score = round(random.gauss(72, 12))
                score = max(20, min(98, score))
                sleep_label = self._sleep_quality_label(score)
                lines.append(f"  🛏️ {name}:")
                lines.append(f"     Điểm giấc ngủ: {score}/100 — {sleep_label}")

        if not any_sleep and not dummy:
            lines.append("  Chưa có dữ liệu giấc ngủ đêm qua.")

        # --- Section 2: current status ---
        lines.append("")
        lines.append("📋 TRẠNG THÁI HIỆN TẠI")

        for zs in zone_statuses:
            activity_vn = ACTIVITY_LABELS_VN.get(zs.activity_state, zs.activity_state)
            resp_str = f"{zs.respiration_bpm:.0f} nhịp/phút" if zs.respiration_bpm else "chưa có"
            hr_str = f"{zs.heart_rate_bpm:.0f} BPM" if zs.heart_rate_bpm else "chưa có"
            lines.append(f"  {self._zone_icon(zs.zone_id)} {zs.name}: {activity_vn}")
            lines.append(f"     Hô hấp: {resp_str} | Nhịp tim: {hr_str}")
            if zs.fall_detected:
                lines.append(f"     ⚠️ PHÁT HIỆN TÉ NGÃ! (độ tin cậy: {zs.fall_confidence:.0%})")

        if not zone_statuses and dummy:
            for zone_id, name in [("zone_bedroom", "Phòng ngủ"), ("zone_living", "Phòng khách")]:
                lines.append(f"  🛏️ {name}: nghỉ ngơi")
                resp = round(random.uniform(12, 18), 1)
                hr = round(random.uniform(60, 80), 1)
                lines.append(f"     Hô hấp: {resp} nhịp/phút | Nhịp tim: {hr} BPM")

        # --- Section 3: alert summary ---
        lines.append("")
        lines.append("🔔 CẢNH BÁO 24 GIỜ QUA")

        if recent_alerts:
            emergency_count = sum(1 for a in recent_alerts if a.get("level") == "EMERGENCY")
            warning_count = sum(1 for a in recent_alerts if a.get("level") == "WARNING")
            info_count = sum(1 for a in recent_alerts if a.get("level") == "INFO")

            lines.append(f"  🚨 Khẩn cấp: {emergency_count} | ⚠️ Cảnh báo: {warning_count} | ℹ️ Thông báo: {info_count}")

            if emergency_count > 0:
                lines.append("  ❗ Các sự kiện khẩn cấp:")
                for a in recent_alerts:
                    if a.get("level") == "EMERGENCY":
                        ts = datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M")
                        lines.append(f"     [{ts}] {a['zone_name']}: {a.get('description', '')}")
        elif dummy:
            lines.append("  🚨 Khẩn cấp: 0 | ⚠️ Cảnh báo: 1 | ℹ️ Thông báo: 2")
            lines.append("  ⚠️ Cảnh báo: Phát hiện không hoạt động kéo dài tại Phòng khách (21:15)")
        else:
            lines.append("  Không có cảnh báo nào trong 24 giờ qua. ✅")

        # --- Section 4: daily advice ---
        lines.append("")
        lines.append("💡 KHUYẾN NGHỊ")
        lines.append(self._daily_advice(zone_statuses, recent_alerts, dummy))

        # --- Footer ---
        lines.append("")
        lines.append("─" * 28)
        lines.append("ElderCare • Hệ thống giám sát người cao tuổi")
        lines.append(f"Báo cáo tự động lúc {now.strftime('%H:%M')}")

        return "\n".join(lines)

    def _sleep_quality_label(self, score: float) -> str:
        if score >= 80:
            return "rất tốt ✅"
        elif score >= 60:
            return "tốt ✅"
        elif score >= 40:
            return "trung bình ⚠️"
        else:
            return "kém ❌"

    def _zone_icon(self, zone_id: str) -> str:
        icons = {
            "zone_bedroom": "🛏️",
            "zone_living": "🛋️",
            "zone_hallway": "🚪",
        }
        return icons.get(zone_id, "📍")

    def _daily_advice(
        self,
        zone_statuses: list,
        recent_alerts: list,
        dummy: bool,
    ) -> str:
        advices = []

        has_sleep_data = any(
            data_store.get_sleep_records(zs.zone_id, n=1)
            for zs in zone_statuses
        )

        if has_sleep_data or dummy:
            advices.append("• Duy trì thời gian đi ngủ và thức dậy đều đặn để cải thiện chất lượng giấc ngủ.")

        if any(a.get("level") == "WARNING" for a in recent_alerts) or dummy:
            advices.append("• Kiểm tra khu vực thường xuyên nếu có cảnh báo không hoạt động kéo dài.")

        if any(a.get("level") == "EMERGENCY" for a in recent_alerts):
            advices.append("• Liên hệ ngay với bác sĩ nếu có sự kiện té ngã — dù không thấy chấn thương bên ngoài.")

        if not advices:
            advices.append("• Mọi chỉ số trong giới hạn bình thường. Tiếp tục theo dõi.")

        advices.append("• Đảm bảo uống đủ nước và vận động nhẹ nhàng trong ngày.")
        return "\n".join(advices)

    def _generate_dummy_zones(self) -> list:
        """Generate plausible dummy zone statuses for testing."""
        from pipeline.data_store import ZoneStatus

        return [
            ZoneStatus(
                zone_id="zone_bedroom", name="Phòng ngủ",
                activity_state="still", respiration_bpm=15.2, respiration_confidence=0.82,
                heart_rate_bpm=68, heart_rate_confidence=0.45,
                online=True, last_seen=0, sleep_stage="light",
            ),
            ZoneStatus(
                zone_id="zone_living", name="Phòng khách",
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
