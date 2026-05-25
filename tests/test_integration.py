"""End-to-end integration tests for the ElderCare inference pipeline.

Feeds simulated CSI data through the full chain:
  CSI Simulator -> feed_packet -> InferenceEngine workers -> output_queue
"""
import time

import numpy as np
import pytest

from firmware.esp32_csi.csi_simulator import CSISimulator, CSISimulatorConfig
from pipeline.inference_engine import InferenceEngine, InferenceResult


class TestEndToEndPipeline:
    """Test the full pipeline with simulated multi-zone CSI data."""

    def _create_engine_with_default_config(self) -> InferenceEngine:
        """Create engine loading real configs."""
        return InferenceEngine(config_path="configs/zones.yaml")

    def _feed_simulated_packets(
        self,
        engine: InferenceEngine,
        zone_id: str,
        state: str,
        n_packets: int = 200,
    ) -> list[InferenceResult]:
        """Feed simulated packets and collect results."""
        config = CSISimulatorConfig(zone_id=zone_id)
        sim = CSISimulator(config)
        sim.set_state(state)

        for _ in range(n_packets):
            pkt = sim.generate_packet()
            engine.feed_packet(zone_id, pkt)

        # Wait for workers to process
        time.sleep(1.0)

        results = engine.get_results()
        return [r for r in results if r.zone_id == zone_id]

    def test_engine_starts_and_stops(self) -> None:
        engine = self._create_engine_with_default_config()
        assert len(engine.workers) == 12  # 4 workers x 3 zones

        engine.start()
        assert all(w.is_alive() for w in engine.workers)

        engine.stop()
        assert all(not w.is_alive() for w in engine.workers)

    def test_fall_detection_produces_results(self) -> None:
        engine = self._create_engine_with_default_config()
        engine.start()

        try:
            results = self._feed_simulated_packets(
                engine, "zone_bedroom", "idle", n_packets=200,
            )
            fall_results = [r for r in results if r.model_name == "fall_detection"]
            assert len(fall_results) > 0, "FallDetectionWorker produced no results"

            for r in fall_results:
                assert "fall_detected" in r.data
                assert "fall_confidence" in r.data
                assert "confirmation_pending" in r.data
        finally:
            engine.stop()

    def test_activity_detection_produces_results(self) -> None:
        engine = self._create_engine_with_default_config()
        engine.start()

        try:
            results = self._feed_simulated_packets(
                engine, "zone_living", "moving", n_packets=200,
            )
            activity_results = [r for r in results if r.model_name == "activity"]
            assert len(activity_results) > 0, "ActivityWorker produced no results"

            for r in activity_results:
                assert "state" in r.data
                assert r.data["state"] in ("active", "still", "inactivity")
        finally:
            engine.stop()

    def test_multi_zone_isolation(self) -> None:
        """Packets for one zone do not appear in another zone's results."""
        engine = self._create_engine_with_default_config()
        engine.start()

        try:
            for zone_id in ["zone_bedroom", "zone_living", "zone_hallway"]:
                config = CSISimulatorConfig(zone_id=zone_id)
                sim = CSISimulator(config)
                sim.set_state("idle")
                for _ in range(200):
                    pkt = sim.generate_packet()
                    engine.feed_packet(zone_id, pkt)

            time.sleep(1.0)
            results = engine.get_results()

            zone_ids = {r.zone_id for r in results}
            assert "zone_bedroom" in zone_ids
            assert "zone_living" in zone_ids
            assert "zone_hallway" in zone_ids
        finally:
            engine.stop()
