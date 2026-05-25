"""
Real-Time Inference Engine

Multiprocessing pipeline that consumes preprocessed CSI data
and runs all four models (fall detection, vital signs, sleep, activity)
in parallel processes. Results flow to data store and alert manager.

Architecture:
  per-zone input queues -> workers -> output queue -> orchestrator (main thread)
    -> data store (dashboard) + alert manager (Telegram)
"""

import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import yaml

from pipeline.data_store import store as data_store

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    zone_id: str
    model_name: str
    timestamp: float
    data: dict[str, Any]


@dataclass
class FallConfirmationEvent:
    zone_id: str
    timestamp: float
    confidence: float


class InferenceWorker(mp.Process):
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
        raise NotImplementedError


class FallDetectionWorker(InferenceWorker):
    def __init__(self, *args, fall_event_queue: Optional[mp.Queue] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_size = self.config.get("window_size", 100)
        self._buffer: list[dict] = []
        self._detector = None
        self._confirmer = None
        self._fall_event_queue = fall_event_queue

    def _ensure_model(self) -> None:
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
        csi_tensor = torch.tensor(amp, dtype=torch.float32)
        pred_class, confidence = self._detector.predict(csi_tensor)
        confirmation = self._confirmer.check(csi_tensor, float(confidence))
        fall_detected = False
        if confirmation is True:
            fall_detected = True
            if self._fall_event_queue is not None:
                self._fall_event_queue.put(FallConfirmationEvent(
                    zone_id=self.zone_id, timestamp=packet["timestamp"], confidence=float(confidence),
                ))
            self._confirmer.reset()
        return InferenceResult(
            zone_id=self.zone_id, model_name="fall_detection", timestamp=packet["timestamp"],
            data={"fall_detected": fall_detected, "fall_confidence": float(confidence),
                  "confirmation_pending": confirmation is None},
        )


class VitalSignsWorker(InferenceWorker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sample_rate = self.config.get("sample_rate", 50.0)
        self._n_subcarriers = self.config.get("n_subcarriers", 52)
        self._update_interval = self.config.get("update_interval_seconds", 5.0)
        self._last_report: float = 0.0
        self._adapter = None

    def _ensure_adapter(self) -> None:
        if self._adapter is not None:
            return
        from models.vital_signs.estimator import VitalsAdapter
        self._adapter = VitalsAdapter(
            n_subcarriers=self._n_subcarriers, sample_rate=self._sample_rate,
            respiration_window_secs=self.config.get("respiration_window_seconds", 30.0),
            heart_rate_window_secs=self.config.get("heart_rate_window_seconds", 15.0),
        )

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._ensure_adapter()
        amplitude = packet.get("csi_amplitude")
        if amplitude is not None and len(amplitude) > 0:
            self._adapter.feed_frame(np.array(amplitude, dtype=np.float32))
        if time.time() - self._last_report < self._update_interval:
            return None
        self._last_report = time.time()
        return InferenceResult(
            zone_id=self.zone_id, model_name="vital_signs", timestamp=packet.get("timestamp", time.time()),
            data={"respiration_bpm": self._adapter.respiration_bpm,
                  "respiration_confidence": self._adapter.respiration_confidence,
                  "heart_rate_bpm": self._adapter.heart_rate_bpm,
                  "heart_rate_confidence": self._adapter.heart_rate_confidence},
        )


class SleepWorker(InferenceWorker):
    """Sleep monitoring with SleepLSTM + SleepScorer + SleepFeatureExtractor."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sample_rate = self.config.get("sample_rate", 50.0)
        self._epoch_frames = int(60 * self._sample_rate)  # 1 minute
        self._buffer: list[dict] = []
        self._feature_extractor = None
        self._model = None
        self._scorer = None
        self._night_epochs: list[dict] = []  # accumulate per night for scoring

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from models.sleep.model import SleepLSTM, SleepScorer, SleepFeatureExtractor
        self._feature_extractor = SleepFeatureExtractor(sample_rate=self._sample_rate)
        self._model = SleepLSTM(n_features=4, hidden_dim=64, n_layers=2, n_classes=3, dropout=0.3)
        self._model.eval()
        try:
            ckpt = torch.load("models/sleep/checkpoints/sleep_lstm_best.pth", map_location="cpu")
            self._model.load_state_dict(ckpt)
            logger.info(f"[{self.model_name}] Loaded SleepLSTM checkpoint")
        except FileNotFoundError:
            logger.warning(f"[{self.model_name}] No checkpoint found, using untrained model")
        self._scorer = SleepScorer()

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._ensure_model()
        self._buffer.append(packet)
        if len(self._buffer) < self._epoch_frames:
            return None

        # Extract features from one epoch of CSI
        epoch_data = self._buffer[-self._epoch_frames:]
        amp_matrix = np.array([p["csi_amplitude"] for p in epoch_data])
        resp_bpm = packet.get("respiration_bpm")
        features = self._feature_extractor.extract(amp_matrix, respiration_rate=resp_bpm)

        # Predict sleep stage
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            probs = self._model(feat_tensor)
            stage_idx = torch.argmax(probs).item()

        stages = {0: "awake", 1: "light", 2: "deep"}
        stage = stages.get(stage_idx, "awake")

        self._night_epochs.append({"stage_idx": stage_idx, "probs": probs[0, 0].tolist()})
        if len(self._night_epochs) > 480:
            self._night_epochs.pop(0)

        # Compute sleep score if we have enough epochs (>= 4 hours = 240 epochs)
        sleep_score = None
        if len(self._night_epochs) >= 240:
            probs_list = [e["probs"] for e in self._night_epochs]
            stage_probs = torch.tensor(probs_list, dtype=torch.float32)
            hours = len(self._night_epochs) / 60.0
            sleep_score, _ = self._scorer.compute_score(
                stage_probs=stage_probs, session_duration_hours=hours,
            )

        self._buffer = []
        return InferenceResult(
            zone_id=self.zone_id, model_name="sleep", timestamp=packet["timestamp"],
            data={"sleep_stage": stage, "sleep_score": sleep_score,
                  "night_epochs": len(self._night_epochs)},
        )


class ActivityWorker(InferenceWorker):
    def __init__(self, *args, fall_event_queue: Optional[mp.Queue] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_frames = int(self.config.get("window_seconds", 30.0) * self.config.get("sample_rate", 50.0))
        self._buffer: list[dict] = []
        self._detector = None
        self._post_fall_checker = None
        self._fall_event_queue = fall_event_queue
        self._inactivity_start: Optional[float] = None

    def _ensure_model(self) -> None:
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
        if self._fall_event_queue is not None:
            while True:
                try:
                    fall_event = self._fall_event_queue.get_nowait()
                    self._post_fall_checker.on_fall_detected(fall_event.timestamp)
                except Exception:
                    break
        amp = np.array([p["csi_amplitude"] for p in self._buffer])
        from datetime import datetime
        ts = packet.get("timestamp", time.time())
        hour = datetime.fromtimestamp(ts).hour + datetime.fromtimestamp(ts).minute / 60.0
        state, alert = self._detector.update(amp, hour)
        if state == "inactivity":
            if self._inactivity_start is None:
                self._inactivity_start = ts
            if self._detector.is_prolonged_inactivity(ts - self._inactivity_start):
                alert = "WARNING"
        else:
            self._inactivity_start = None
        post_fall_alert = self._post_fall_checker.check(amp, ts)
        data = {"state": state, "alert": alert}
        if post_fall_alert is not None:
            data["post_fall_alert"] = post_fall_alert
        return InferenceResult(
            zone_id=self.zone_id, model_name="activity", timestamp=ts, data=data,
        )


class _ResultOrchestrator(threading.Thread):
    """Reads results from output queue, updates data store, alert manager, and handles daily summaries."""

    def __init__(self, engine: "InferenceEngine", output_queue: mp.Queue, stop_event: mp.Event) -> None:
        super().__init__(name="result-orchestrator", daemon=True)
        self._engine = engine
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._alert_mgr = None
        self._last_summary_date: str = ""

    def _ensure_alert_mgr(self) -> None:
        if self._alert_mgr is not None:
            return
        from alerts.alert_manager import AlertManager, AlertLevel, AlertMessage
        self.AlertLevel = AlertLevel
        self.AlertMessage = AlertMessage
        self._alert_mgr = AlertManager()

    def run(self) -> None:
        logger.info("[Orchestrator] Started")
        while not self._stop_event.is_set():
            try:
                result = self._output_queue.get(timeout=0.3)
                self._handle_result(result)
            except queue.Empty:
                pass
            except Exception:
                logger.exception("[Orchestrator] Error handling result")

            self._check_daily_summary()

        logger.info("[Orchestrator] Stopped")

    def _handle_result(self, r: InferenceResult) -> None:
        zid = r.zone_id
        zone_cfg = self._engine._zone_configs.get(zid, {})
        zone_name = zone_cfg.get("name", zid)

        if r.model_name == "fall_detection":
            fd = r.data
            data_store.update_fall(zid, fd["fall_detected"], fd["fall_confidence"])
            if fd["fall_detected"]:
                self._send_alert(zid, zone_name, "EMERGENCY", "Phát hiện té ngã!",
                                 {"confidence": fd["fall_confidence"]})

        elif r.model_name == "vital_signs":
            data_store.update_vitals(
                zid, r.data["respiration_bpm"], r.data["respiration_confidence"],
                r.data["heart_rate_bpm"], r.data["heart_rate_confidence"],
            )

        elif r.model_name == "sleep":
            data_store.update_sleep(zid, r.data["sleep_stage"], r.data["sleep_score"])

        elif r.model_name == "activity":
            data_store.update_activity(zid, r.data["state"], r.data.get("alert"))
            if r.data.get("alert") == "WARNING":
                self._send_alert(zid, zone_name, "WARNING",
                                 f"Phát hiện không hoạt động kéo dài ({zid})")
            if r.data.get("post_fall_alert") == "EMERGENCY":
                self._send_alert(zid, zone_name, "EMERGENCY",
                                 f"KHẨN CẤP: Không cử động sau té ngã tại {zone_name}!")

    def _check_daily_summary(self) -> None:
        from datetime import datetime

        try:
            config = self._alert_mgr.config if self._alert_mgr else {}
        except Exception:
            config = {}
        ds_config = config.get("daily_summary", {})
        if not ds_config.get("enabled", True):
            return

        summary_time = ds_config.get("time_local", "07:00")
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")

        if self._last_summary_date == today:
            return
        if current_time < summary_time:
            return

        self._last_summary_date = today
        self._ensure_alert_mgr()
        try:
            self._alert_mgr.send_daily_summary(dummy=False)
            logger.info("[Orchestrator] Daily summary sent for %s", today)
        except Exception:
            logger.exception("[Orchestrator] Failed to send daily summary")

    def _send_alert(self, zone_id: str, zone_name: str, level: str, desc: str, data: dict = None) -> None:
        self._ensure_alert_mgr()
        lvl = getattr(self.AlertLevel, level, self.AlertLevel.INFO)
        msg = self.AlertMessage(
            zone_id=zone_id, zone_name=zone_name, level=lvl,
            event_type=level.lower(), timestamp=time.time(),
            description=desc, data=data or {},
        )
        dispatched = self._alert_mgr.send_alert(msg)
        data_store.add_alert({
            "zone_id": zone_id, "zone_name": zone_name, "level": level,
            "event_type": "fall" if "té ngã" in desc else "inactivity" if "không hoạt động" in desc else "alert",
            "timestamp": time.time(), "description": desc, "dispatched": dispatched,
            "id": len(data_store._alerts),
        })


class InferenceEngine:
    """Orchestrates all inference workers + result consumer."""

    def __init__(self, config_path: str = "configs/zones.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config()
        self.zones = self.config.get("zones", [])
        self._zone_configs: dict[str, dict] = {z["zone_id"]: z for z in self.zones}
        self.output_queue = mp.Queue(maxsize=2000)
        self.stop_event = mp.Event()
        self.workers: list[InferenceWorker] = []
        self._zone_queues: dict[str, list[mp.Queue]] = {}
        self._orchestrator: Optional[_ResultOrchestrator] = None
        self._create_workers()

        # Ensure data store has zones registered
        for zone in self.zones:
            if zone.get("active", True):
                data_store.ensure_zone(zone["zone_id"], zone.get("name", zone["zone_id"]))

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        return {"zones": [{"zone_id": "zone_default", "name": "Default", "active": True}]}

    def _create_workers(self) -> None:
        for zone in self.zones:
            if not zone.get("active", True):
                continue
            zid = zone["zone_id"]
            fall_event_queue = mp.Queue(maxsize=50)
            worker_configs = [
                (FallDetectionWorker, {"sample_rate": 50.0, "window_size": 100, "n_subcarriers": 52,
                                        "confidence_threshold": 0.85, "confirmation_window_seconds": 3.0,
                                        "inactivity_threshold": 0.15}),
                (VitalSignsWorker, {"sample_rate": 50.0, "n_subcarriers": 52,
                                     "fft_window_seconds": 30.0, "update_interval_seconds": 5.0}),
                (SleepWorker, {"sample_rate": 50.0, "epoch_duration_minutes": 1}),
                (ActivityWorker, {"sample_rate": 50.0, "window_seconds": 30.0,
                                   "threshold_active": 0.5, "threshold_still": 0.15,
                                   "inactivity_timeout_seconds": 7200.0,
                                   "daytime_start_hour": 6, "daytime_end_hour": 22,
                                   "recovery_timeout_seconds": 30.0}),
            ]
            for wcls, wcfg in worker_configs:
                worker_name = f"{wcls.__name__.replace('Worker', '')}_{zid}"
                wq = mp.Queue(maxsize=2000)
                kwargs: dict[str, Any] = {
                    "name": worker_name, "zone_id": zid, "input_queue": wq,
                    "output_queue": self.output_queue, "stop_event": self.stop_event, "config": wcfg,
                }
                if wcls in (FallDetectionWorker, ActivityWorker):
                    kwargs["fall_event_queue"] = fall_event_queue
                self.workers.append(wcls(**kwargs))
                self._zone_queues.setdefault(zid, []).append(wq)

    def start(self) -> None:
        logger.info(f"Starting inference engine: {len(self.workers)} workers, {len(self._zone_queues)} zones")
        for w in self.workers:
            w.start()
        self._orchestrator = _ResultOrchestrator(self, self.output_queue, self.stop_event)
        self._orchestrator.start()

    def stop(self) -> None:
        logger.info("Stopping inference engine...")
        self.stop_event.set()
        if self._orchestrator:
            self._orchestrator.join(timeout=3.0)
        for w in self.workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()
        logger.info("Inference engine stopped")

    def feed_packet(self, zone_id: str, packet: dict) -> None:
        worker_queues = self._zone_queues.get(zone_id)
        if not worker_queues:
            return
        for wq in worker_queues:
            try:
                wq.put(packet, timeout=1.0)
            except queue.Full:
                pass

    def get_results(self, timeout: float = 0.1) -> list[dict]:
        results: list[dict] = []
        for zs in data_store.get_all_zone_statuses():
            entry = {
                "zone_id": zs.zone_id, "model_name": "activity", "timestamp": zs.last_seen or 0,
                "data": {"state": zs.activity_state, "alert": None},
            }
            results.append(entry)
        return results

    @property
    def zone_configs(self) -> dict:
        return self._zone_configs


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="ElderCare Inference Engine")
    parser.add_argument("--config", default="configs/zones.yaml")
    args = parser.parse_args()
    engine = InferenceEngine(config_path=args.config)
    engine.start()
    logger.info(f"Inference engine running. {len(engine.workers)} workers active.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
