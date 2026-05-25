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
        signal.signal(signal.SIGINT, signal.SIG_IGN)

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
    """Fall detection inference worker with TwoStageConfirmer."""

    def __init__(self, *args, fall_event_queue: Optional[mp.Queue] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_size = self.config.get("window_size", 100)
        self._buffer: list[dict] = []
        self._detector = None
        self._confirmer = None
        self._fall_event_queue = fall_event_queue

    def _ensure_model(self) -> None:
        """Lazy-initialize model on first use (avoids blocking process spawn)."""
        if self._detector is not None:
            return
        from models.fall_detection.model import FallDetector, TwoStageConfirmer

        self._detector = FallDetector(
            n_subcarriers=self.config.get("n_subcarriers", 52),
            sequence_length=self._window_size,
        )
        self._detector.eval()

        self._confirmer = TwoStageConfirmer(
            confidence_threshold=self.config.get("confidence_threshold", 0.85),
            confirmation_window_seconds=self.config.get("confirmation_window_seconds", 3.0),
            inactivity_threshold=self.config.get("inactivity_threshold", 0.15),
            sample_rate=self.config.get("sample_rate", 50.0),
        )

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._ensure_model()
        self._buffer.append(packet)
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_size:
            return None

        amp = np.array([p["csi_amplitude"] for p in self._buffer])

        import torch
        csi_tensor = torch.tensor(amp, dtype=torch.float32)
        pred_class, confidence = self._detector.predict(csi_tensor)

        csi_for_confirmer = torch.tensor(amp, dtype=torch.float32)
        confirmation = self._confirmer.check(csi_for_confirmer, float(confidence))

        fall_detected = False
        if confirmation is True:
            fall_detected = True
            if self._fall_event_queue is not None:
                self._fall_event_queue.put(FallConfirmationEvent(
                    zone_id=self.zone_id,
                    timestamp=packet["timestamp"],
                    confidence=float(confidence),
                ))
            self._confirmer.reset()

        return InferenceResult(
            zone_id=self.zone_id,
            model_name="fall_detection",
            timestamp=packet["timestamp"],
            data={
                "fall_detected": fall_detected,
                "fall_confidence": float(confidence),
                "confirmation_pending": confirmation is None,
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
    """Activity / inactivity detection worker with day/night awareness and post-fall monitoring."""

    def __init__(self, *args, fall_event_queue: Optional[mp.Queue] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_frames = int(
            self.config.get("window_seconds", 30.0) * self.config.get("sample_rate", 50.0)
        )
        self._buffer: list[dict] = []
        self._detector = None
        self._post_fall_checker = None
        self._fall_event_queue = fall_event_queue
        self._inactivity_start: Optional[float] = None

    def _ensure_model(self) -> None:
        """Lazy-initialize model on first use (avoids blocking process spawn)."""
        if self._detector is not None:
            return
        from models.activity.detector import ActivityDetector, PostFallInactivityChecker

        self._detector = ActivityDetector(
            threshold_active=self.config.get("threshold_active", 0.5),
            threshold_still=self.config.get("threshold_still", 0.15),
            window_seconds=self.config.get("window_seconds", 30.0),
            sample_rate=self.config.get("sample_rate", 50.0),
            inactivity_timeout_seconds=self.config.get("inactivity_timeout_seconds", 7200.0),
            daytime_start_hour=self.config.get("daytime_start_hour", 6),
            daytime_end_hour=self.config.get("daytime_end_hour", 22),
        )

        self._post_fall_checker = PostFallInactivityChecker(
            recovery_timeout_seconds=self.config.get("recovery_timeout_seconds", 30.0),
        )

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._ensure_model()
        self._buffer.append(packet)
        if len(self._buffer) > self._window_frames:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_frames // 10:
            return None

        # Check for fall events from FallDetectionWorker
        if self._fall_event_queue is not None:
            while True:
                try:
                    fall_event = self._fall_event_queue.get_nowait()
                    self._post_fall_checker.on_fall_detected(fall_event.timestamp)
                except Exception:
                    break

        amp = np.array([p["csi_amplitude"] for p in self._buffer])

        # Derive timestamp hour for day/night check
        from datetime import datetime
        ts = packet.get("timestamp", 0.0)
        hour = datetime.fromtimestamp(ts).hour + datetime.fromtimestamp(ts).minute / 60.0

        state, alert = self._detector.update(amp, hour)

        # Track inactivity duration
        if state == "inactivity":
            if self._inactivity_start is None:
                self._inactivity_start = ts
            inactivity_duration = ts - self._inactivity_start
            if self._detector.is_prolonged_inactivity(inactivity_duration):
                alert = "WARNING"
        else:
            self._inactivity_start = None

        # Post-fall emergency check
        post_fall_alert = self._post_fall_checker.check(amp, ts)

        data = {
            "state": state,
            "alert": alert,
        }
        if post_fall_alert is not None:
            data["post_fall_alert"] = post_fall_alert

        return InferenceResult(
            zone_id=self.zone_id,
            model_name="activity",
            timestamp=packet["timestamp"],
            data=data,
        )


class InferenceEngine:
    """Orchestrates all inference workers.

    Consumes CSI from the ingestion layer, fans out to per-zone
    inference workers, and collects results onto the output event bus.
    """

    def __init__(self, config_path: str = "configs/zones.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config()

        self.zones = self.config.get("zones", [])
        self.output_queue = mp.Queue(maxsize=2000)

        self.stop_event = mp.Event()
        self.workers: list[InferenceWorker] = []
        self._zone_queues: dict[str, list[mp.Queue]] = {}
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

            # Shared fall event queue for this zone (fall worker -> activity worker)
            fall_event_queue = mp.Queue(maxsize=50)

            worker_configs = [
                (FallDetectionWorker, {
                    "sample_rate": 50.0,
                    "window_size": 100,
                    "n_subcarriers": 52,
                    "confidence_threshold": 0.85,
                    "confirmation_window_seconds": 3.0,
                    "inactivity_threshold": 0.15,
                }),
                (VitalSignsWorker, {
                    "sample_rate": 50.0,
                    "fft_window_seconds": 30.0,
                    "update_interval_seconds": 5.0,
                }),
                (SleepWorker, {
                    "sample_rate": 50.0,
                    "epoch_duration_minutes": 1,
                }),
                (ActivityWorker, {
                    "sample_rate": 50.0,
                    "window_seconds": 30.0,
                    "threshold_active": 0.5,
                    "threshold_still": 0.15,
                    "inactivity_timeout_seconds": 7200.0,
                    "daytime_start_hour": 6,
                    "daytime_end_hour": 22,
                    "recovery_timeout_seconds": 30.0,
                }),
            ]

            for wcls, worker_config in worker_configs:
                worker_name = f"{wcls.__name__.replace('Worker', '')}_{zid}"

                # Each worker gets its own input queue to guarantee
                # every worker receives every packet for its zone.
                worker_queue = mp.Queue(maxsize=2000)

                kwargs: dict[str, Any] = {
                    "name": worker_name,
                    "zone_id": zid,
                    "input_queue": worker_queue,
                    "output_queue": self.output_queue,
                    "stop_event": self.stop_event,
                    "config": worker_config,
                }

                # Only fall and activity workers get the shared fall event queue
                if wcls in (FallDetectionWorker, ActivityWorker):
                    kwargs["fall_event_queue"] = fall_event_queue

                worker = wcls(**kwargs)
                self.workers.append(worker)

                # Track worker queues for feed_packet routing
                self._zone_queues.setdefault(zid, []).append(worker_queue)

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
        """Feed a preprocessed CSI packet to all workers for the given zone.

        Puts one copy on each worker's dedicated input queue so that every
        worker (fall, vital, sleep, activity) receives its own copy.
        """
        worker_queues = self._zone_queues.get(zone_id)
        if not worker_queues:
            logger.warning(f"No queue for zone {zone_id}, dropping packet")
            return
        for wq in worker_queues:
            try:
                wq.put(packet, timeout=1.0)
            except queue.Full:
                logger.warning(f"Queue full for zone {zone_id}, dropping packet")

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
