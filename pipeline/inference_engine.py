"""
Real-Time Inference Engine

Multiprocessing pipeline that consumes preprocessed CSI data
and runs all four models (fall detection, vital signs, sleep, activity)
in parallel processes. Results are published to an internal event bus.
"""

import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Unified inference output from any model."""
    zone_id: str
    model_name: str
    timestamp: float
    data: dict[str, Any]


@dataclass
class FallConfirmationEvent:
    """Event posted when a fall is confirmed by TwoStageConfirmer."""
    zone_id: str
    timestamp: float
    confidence: float


class InferenceWorker(mp.Process):
    """Base class for model-specific inference workers."""

    def __init__(
        self,
        name: str,
        zone_id: str,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        stop_event: mp.Event,
        config: dict,
    ) -> None:
        super().__init__(name=name)
        self.model_name = name
        self.zone_id = zone_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.config = config

    def run(self) -> None:
        logger.info(f"[{self.model_name}] Worker started for zone={self.zone_id}")
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent handles SIGINT

        while not self.stop_event.is_set():
            try:
                packet = self.input_queue.get(timeout=0.5)
                result = self.process(packet)
                if result is not None:
                    self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception:
                logger.exception(f"[{self.model_name}] Error in inference loop")

        logger.info(f"[{self.model_name}] Worker stopped")

    def process(self, packet: dict) -> Optional[InferenceResult]:
        """Override in subclass to implement model-specific inference."""
        raise NotImplementedError


class FallDetectionWorker(InferenceWorker):
    """Fall detection inference worker."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_size = self.config.get("window_size", 100)
        self._buffer: list[dict] = []

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._buffer.append(packet)
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_size:
            return None

        # Extract amplitude matrix
        amp = np.array([p["csi_amplitude"] for p in self._buffer])

        # Placeholder: actual model inference goes here (Phase 1+)
        # For Phase 0, return a stub result
        fall_confidence = 0.01  # dummy value

        return InferenceResult(
            zone_id=self.zone_id,
            model_name="fall_detection",
            timestamp=packet["timestamp"],
            data={
                "fall_detected": False,
                "fall_confidence": fall_confidence,
            },
        )


class VitalSignsWorker(InferenceWorker):
    """Vital signs estimation worker."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_seconds = self.config.get("fft_window_seconds", 30.0)
        self._sample_rate = self.config.get("sample_rate", 50.0)
        self._window_frames = int(self._window_seconds * self._sample_rate)
        self._buffer: list[dict] = []
        self._update_interval = self.config.get("update_interval_seconds", 5.0)
        self._last_report: float = 0.0

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._buffer.append(packet)
        if len(self._buffer) > self._window_frames:
            self._buffer.pop(0)

        # Report at update interval
        if time.time() - self._last_report < self._update_interval:
            return None

        if len(self._buffer) < self._window_frames // 2:
            return None

        self._last_report = time.time()

        # Placeholder: actual FFT-based estimation goes here (Phase 1+)
        return InferenceResult(
            zone_id=self.zone_id,
            model_name="vital_signs",
            timestamp=packet["timestamp"],
            data={
                "respiration_bpm": 15.0,
                "respiration_confidence": 0.8,
                "heart_rate_bpm": None,        # experimental
                "heart_rate_confidence": None,  # experimental
            },
        )


class SleepWorker(InferenceWorker):
    """Sleep monitoring worker."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epoch_minutes = self.config.get("epoch_duration_minutes", 1)
        self._epoch_frames = int(self._epoch_minutes * 60 * self.config.get("sample_rate", 50.0))
        self._buffer: list[dict] = []
        self._features: list[np.ndarray] = []

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._buffer.append(packet)
        if len(self._buffer) < self._epoch_frames:
            return None

        # Extract epoch features (placeholder)
        epoch_data = self._buffer[-self._epoch_frames:]
        # In Phase 4, extract real features via SleepFeatureExtractor
        self._buffer = []  # reset for next epoch

        return InferenceResult(
            zone_id=self.zone_id,
            model_name="sleep",
            timestamp=packet["timestamp"],
            data={
                "sleep_stage": "awake",
                "sleep_score": None,  # computed nightly
            },
        )


class ActivityWorker(InferenceWorker):
    """Activity / inactivity detection worker."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_frames = int(self.config.get("window_seconds", 30.0) * self.config.get("sample_rate", 50.0))
        self._buffer: list[dict] = []
        self._state: str = "active"

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._buffer.append(packet)
        if len(self._buffer) > self._window_frames:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_frames // 2:
            return None

        amp = np.array([p["csi_amplitude"] for p in self._buffer])
        variance = float(np.var(np.mean(np.abs(amp), axis=1)))

        # Rule-based classification
        if variance > 0.5:
            new_state = "active"
        elif variance > 0.15:
            new_state = "still"
        else:
            new_state = "inactivity"

        self._state = new_state

        return InferenceResult(
            zone_id=self.zone_id,
            model_name="activity",
            timestamp=packet["timestamp"],
            data={
                "state": new_state,
                "variance": variance,
            },
        )


class InferenceEngine:
    """Orchestrates all inference workers.

    Consumes CSI from the ingestion layer (via shared queue),
    runs preprocessing, fans out to inference workers, and
    collects results onto the output event bus.
    """

    def __init__(self, config_path: str = "configs/zones.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config()

        self.zones = self.config.get("zones", [])
        self.input_queue = mp.Queue(maxsize=500)
        self.output_queue = mp.Queue(maxsize=500)

        self.stop_event = mp.Event()
        self.workers: list[InferenceWorker] = []
        self._create_workers()

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        logger.warning(f"Config not found: {self.config_path}. Using defaults.")
        return {
            "zones": [
                {"zone_id": "zone_default", "name": "Default", "active": True},
            ]
        }

    def _create_workers(self) -> None:
        for zone in self.zones:
            if not zone.get("active", True):
                continue

            zid = zone["zone_id"]
            worker_classes = [
                FallDetectionWorker,
                VitalSignsWorker,
                SleepWorker,
                ActivityWorker,
            ]

            for wcls in worker_classes:
                worker_name = f"{wcls.__name__.replace('Worker', '')}_{zid}"
                worker_config = {
                    "sample_rate": 50.0,
                    "window_size": 100,
                    "fft_window_seconds": 30.0,
                    "update_interval_seconds": 5.0,
                    "epoch_duration_minutes": 1,
                    "window_seconds": 30.0,
                }
                worker = wcls(
                    name=worker_name,
                    zone_id=zid,
                    input_queue=self.input_queue,
                    output_queue=self.output_queue,
                    stop_event=self.stop_event,
                    config=worker_config,
                )
                self.workers.append(worker)

    def start(self) -> None:
        logger.info(f"Starting inference engine with {len(self.workers)} workers across {len(self.zones)} zones")
        for w in self.workers:
            w.start()

    def stop(self) -> None:
        logger.info("Stopping inference engine...")
        self.stop_event.set()
        for w in self.workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()
        logger.info("Inference engine stopped")

    def feed_packet(self, zone_id: str, packet: dict) -> None:
        """Feed a preprocessed CSI packet to all workers for the given zone."""
        try:
            self.input_queue.put({"zone_id": zone_id, "packet": packet}, timeout=1.0)
        except queue.Full:
            logger.warning("Input queue full, dropping packet")

    def get_results(self, timeout: float = 0.1) -> list[InferenceResult]:
        """Non-blocking fetch of inference results."""
        results = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results


def main() -> None:
    """Entry point for standalone inference engine (without Docker)."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ElderCare Inference Engine")
    parser.add_argument("--config", default="configs/zones.yaml", help="Path to zones config")
    args = parser.parse_args()

    engine = InferenceEngine(config_path=args.config)
    engine.start()

    logger.info(f"Inference engine running. {len(engine.workers)} workers active.")
    logger.info("Feed CSI packets via MQTT ingestion layer. Press Ctrl+C to stop.")

    try:
        while True:
            results = engine.get_results(timeout=0.5)
            for r in results:
                logger.info(f"[{r.zone_id}] {r.model_name}: {r.data}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
