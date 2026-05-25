"""Tests for daily summary generation."""
import pytest
from alerts.daily_summary import DailySummaryGenerator, generate_daily_summary


class TestDailySummary:
    def test_generates_with_dummy_data(self) -> None:
        """Dummy mode should produce a valid Vietnamese report."""
        text = generate_daily_summary(dummy=True)
        assert "ELDERCARE" in text.upper()
        assert "ĐÊM QUA" in text
        assert "TRẠNG THÁI HIỆN TẠI" in text
        assert "CẢNH BÁO" in text
        assert "KHUYẾN NGHỊ" in text
        assert 200 < len(text) < 3000

    def test_generates_non_dummy_without_crash(self) -> None:
        """Non-dummy mode should not crash even with empty data store."""
        text = generate_daily_summary(dummy=False)
        assert "ELDERCARE" in text.upper()
        assert len(text) > 0

    def test_generator_class(self) -> None:
        gen = DailySummaryGenerator()
        text = gen.generate(dummy=True)
        assert "ELDERCARE" in text.upper()
        assert len(text) > 100

    def test_sleep_quality_labels(self) -> None:
        gen = DailySummaryGenerator()
        assert "rất tốt" in gen._sleep_quality_label(90)
        assert "tốt" in gen._sleep_quality_label(65)
        assert "trung bình" in gen._sleep_quality_label(50)
        assert "kém" in gen._sleep_quality_label(30)

    def test_zone_icons(self) -> None:
        gen = DailySummaryGenerator()
        assert gen._zone_icon("zone_bedroom") == "🛏️"
        assert gen._zone_icon("zone_living") == "🛋️"
        assert gen._zone_icon("unknown_zone") == "📍"
