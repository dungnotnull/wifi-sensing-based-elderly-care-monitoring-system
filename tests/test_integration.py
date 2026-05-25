"""End-to-end integration tests for the ElderCare inference pipeline.

Feeds simulated CSI data through the full chain and verifies results
appear in the InferenceDataStore.
"""
import time

import numpy as np
import pytest

from firmware.esp32_csi.csi_simulator import CSISimulator, CSISimulatorConfig
from pipeline.inference_engine import InferenceEngine
from pipeline.data_store import store as data_store


class TestEndToEndPipeline:
    """Test the full pipeline with simulated multi-zone CSI data."""

    def _create_engine_with_default_config(self) -> InferenceEngine:
        return InferenceEngine(config_path="configs/zones.yaml")

    def _feed_simulated_packets(
        self,
        engine: InferenceEngine,
        zone_id: str,
        state: str,
        n_packets: int = 200,
    ) -> None:
        config = CSISimulatorConfig(zone_id=zone_id)
        sim = CSISimulator(config)
        sim.set_state(state)
        for _ in range(n_packets):
            pkt = sim.generate_packet()
            engine.feed_packet(zone_id, pkt)
        time.sleep(8.0)  # wait for workers + model lazy init + result propagation

    def test_engine_starts_and_stops(self) -> None:
        engine = self._create_engine_with_default_config()
        assert len(engine.workers) == 12
        engine.start()
        assert all(w.is_alive() for w in engine.workers)
        engine.stop()
        assert all(not w.is_alive() for w in engine.workers)

    def test_fall_detection_produces_results(self) -> None:
        engine = self._create_engine_with_default_config()
        engine.start()
        try:
            self._feed_simulated_packets(engine, "zone_bedroom", "idle", n_packets=200)
            zs = data_store.get_zone_status("zone_bedroom")
            assert zs is not None, "Zone status not populated"
            assert zs.activity_state != "unknown", f"Activity state not updated: {zs.activity_state}"
        finally:
            engine.stop()

    def test_activity_detection_produces_results(self) -> None:
        engine = self._create_engine_with_default_config()
        engine.start()
        try:
            self._feed_simulated_packets(engine, "zone_living", "moving", n_packets=200)
            zs = data_store.get_zone_status("zone_living")
            assert zs is not None
            assert zs.activity_state in ("active", "still", "inactivity")
        finally:
            engine.stop()

    def test_multi_zone_isolation(self) -> None:
        engine = self._create_engine_with_default_config()
        engine.start()
        try:
            for zone_id in ["zone_bedroom", "zone_living", "zone_hallway"]:
                config = CSISimulatorConfig(zone_id=zone_id)
                sim = CSISimulator(config)
                sim.set_state("idle")
                for _ in range(200):
                    engine.feed_packet(zone_id, sim.generate_packet())

            time.sleep(8.0)

            zone_ids = {z.zone_id for z in data_store.get_all_zone_statuses()}
            assert "zone_bedroom" in zone_ids
            assert "zone_living" in zone_ids
            assert "zone_hallway" in zone_ids
        finally:
            engine.stop()
