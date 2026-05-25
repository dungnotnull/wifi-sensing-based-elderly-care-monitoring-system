"""Tests for CSI simulator."""
import pytest
from firmware.esp32_csi.csi_simulator import CSISimulator, CSISimulatorConfig


class TestCSISimulator:
    def test_generates_valid_packet(self) -> None:
        sim = CSISimulator()
        packet = sim.generate_packet()
        assert packet["zone_id"] == "zone_test"
        assert len(packet["csi_amplitude"]) == 52
        assert len(packet["csi_phase"]) == 52
        assert isinstance(packet["rssi"], float)
        assert packet["sequence_number"] > 0

    def test_stream_generates_packets(self) -> None:
        sim = CSISimulator(config=CSISimulatorConfig(sample_rate_hz=100.0))
        packets = list(sim.stream(duration_seconds=0.05))
        assert len(packets) >= 4

    def test_different_states_produce_different_signals(self) -> None:
        sim = CSISimulator()
        sim.set_state("idle")
        idle_packet = sim.generate_packet()
        idle_amp = idle_packet["csi_amplitude"]

        sim.set_state("falling")
        sim2 = CSISimulator()
        sim2.set_state("falling")
        fall_packet = sim2.generate_packet()

        # Falling should change the amplitude pattern
        assert idle_amp != fall_packet["csi_amplitude"]
