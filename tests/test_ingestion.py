"""Tests for CSI ingestion layer."""
import pytest
from ingestion.receiver import CSIRingBuffer, ZoneConfig


class TestCSIRingBuffer:
    def test_push_and_get_recent(self) -> None:
        buf = CSIRingBuffer(zone_id="test", max_frames=5)
        for i in range(10):
            buf.push({"seq": i, "csi_amplitude": [1.0] * 52, "csi_phase": [0.0] * 52})
        assert len(buf) == 5

        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[0]["seq"] == 7
        assert recent[-1]["seq"] == 9

    def test_empty_buffer(self) -> None:
        buf = CSIRingBuffer(zone_id="test")
        assert len(buf) == 0
        assert buf.get_recent(5) == []
