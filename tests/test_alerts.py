"""Tests for alert manager."""
import pytest
from alerts.alert_manager import AlertManager, AlertLevel


class TestAlertManager:
    def test_create_alert_manager(self) -> None:
        mgr = AlertManager(config_path="configs/alerts.yaml")
        assert mgr is not None

    def test_send_info_alert(self) -> None:
        mgr = AlertManager(config_path="configs/alerts.yaml")
        result = mgr.send_info(
            zone_id="test_zone",
            zone_name="Test Zone",
            description="Test info alert",
        )
        assert result is True

    def test_cooldown_blocks_repeated_alert(self) -> None:
        mgr = AlertManager(config_path="configs/alerts.yaml")
        # First alert should send
        result1 = mgr.send_warning(
            zone_id="test_zone",
            zone_name="Test Zone",
            description="First warning",
        )
        assert result1 is True

        # Second alert should be blocked by cooldown
        result2 = mgr.send_warning(
            zone_id="test_zone",
            zone_name="Test Zone",
            description="Second warning — should be blocked",
        )
        assert result2 is False

    def test_format_alert_vietnamese(self) -> None:
        from alerts.alert_manager import AlertMessage
        alert = AlertMessage(
            zone_id="zone_bedroom",
            zone_name="Phòng ngủ",
            level=AlertLevel.EMERGENCY,
            event_type="fall_detected",
            timestamp=1716600000.0,
            description="Phát hiện té ngã!",
        )
        formatted = alert.format_vn()
        assert "KHẨN CẤP" in formatted
        assert "Phòng ngủ" in formatted
        assert "té ngã" in formatted
