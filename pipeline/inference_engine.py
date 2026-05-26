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
from pipeline.watchdog import WorkerWatchdog

from alerts.i18n import locale

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
        heartbeat_dict: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(name=name)
        self.model_name = name
        self.zone_id = zone_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.config = config
        self._heartbeat_dict = heartbeat_dict
        self._last_heartbeat: float = 0.0

    def run(self) -> None:
        logger.info(f"[{self.model_name}] Worker started for zone={self.zone_id}")
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while not self.stop_event.is_set():
            try:
                self._send_heartbeat()
                packet = self.input_queue.get(timeout=0.5)
                result = self.process(packet)
                if result is not None:
                    self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception:
                logger.exception(f"[{self.model_name}] Error in inference loop")
        logger.info(f"[{self.model_name}] Worker stopped")

    def _send_heartbeat(self) -> None:
        """Send heartbeat timestamp to shared dict every 5 seconds."""
        now = time.time()
        if now - self._last_heartbeat >= 5.0:
            self._last_heartbeat = now
            if self._heartbeat_dict is not None:
                self._heartbeat_dict[self.model_name] = now

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
        from pipeline.degradation import degradation
        self._degradation = degradation
        try:
            from models.vital_signs.estimator import VitalsAdapter
            self._adapter = VitalsAdapter(
                n_subcarriers=self._n_subcarriers, sample_rate=self._sample_rate,
                respiration_window_secs=self.config.get("respiration_window_seconds", 30.0),
                heart_rate_window_secs=self.config.get("heart_rate_window_seconds", 15.0),
            )
            # VitalsAdapter auto-tries Rust, falls back to Python internally.
            # Check which backend actually loaded.
            if getattr(self._adapter, '_using_fallback', False):
                self._degradation.degrade("vital_signs")
                logger.warning(f"[{self.model_name}] Using Python scipy fallback for vitals")
        except Exception as e:
            self._degradation.mark_error("vital_signs", str(e))
            msg = self._degradation.degrade("vital_signs")
            if msg:
                logger.warning(f"[{self.model_name}] {msg}")
            # Create fallback directly
            from models.vital_signs.python_fallback import PythonVitalsFallback
            self._adapter = PythonVitalsFallback(
                n_subcarriers=self._n_subcarriers, sample_rate=self._sample_rate,
            )
            self._degradation.degrade("vital_signs")

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
        self._model = SleepLSTM(n_features=6, hidden_dim=64, n_layers=2, n_classes=3, dropout=0.3)
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
        from pipeline.multi_person import MultiPersonDetector
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
        self._multi_person_detector = MultiPersonDetector(
            window_frames=self.config.get("multi_person_window_frames", 250),
            spread_threshold=self.config.get("multi_person_spread_threshold", 0.45),
            entropy_threshold=self.config.get("multi_person_entropy_threshold", 2.5),
            confirmation_frames=self.config.get("multi_person_confirm_frames", 50),
            cooldown_seconds=self.config.get("multi_person_cooldown_seconds", 30.0),
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
        occupancy = self._multi_person_detector.analyze(amp)
        data = {"state": state, "alert": alert, "occupancy": occupancy.state,
                "occupancy_confidence": occupancy.confidence}
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

        # Multi-person gate: suppress alerts for multi-occupant zones
        multi_occupant = r.data.get("occupancy") if r.model_name == "activity" else None
        suppress_alerts = multi_occupant == "multi"

        if r.model_name == "fall_detection":
            fd = r.data
            data_store.update_fall(zid, fd["fall_detected"], fd["fall_confidence"])
            if fd["fall_detected"] and not suppress_alerts:
                self._send_alert(zid, zone_name, "EMERGENCY",
                                 locale.t("alerts.fall_detected"),
                                 {"confidence": fd["fall_confidence"]},
                                 event_type="fall")

        elif r.model_name == "vital_signs":
            data_store.update_vitals(
                zid, r.data["respiration_bpm"], r.data["respiration_confidence"],
                r.data["heart_rate_bpm"], r.data["heart_rate_confidence"],
            )

        elif r.model_name == "sleep":
            data_store.update_sleep(zid, r.data["sleep_stage"], r.data["sleep_score"])

        elif r.model_name == "activity":
            occupancy = r.data.get("occupancy", "single")
            occ_conf = r.data.get("occupancy_confidence", 0.0)
            data_store.update_activity(zid, r.data["state"], r.data.get("alert"),
                                       occupancy=occupancy, occupancy_confidence=occ_conf)
            if not suppress_alerts:
                if r.data.get("alert") == "WARNING":
                    self._send_alert(zid, zone_name, "WARNING",
                                     locale.t("alerts.inactivity_detected", zone_id=zid),
                                     event_type="inactivity")
                if r.data.get("post_fall_alert") == "EMERGENCY":
                    self._send_alert(zid, zone_name, "EMERGENCY",
                                     locale.t("alerts.post_fall_emergency", zone_name=zone_name),
                                     event_type="fall")

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

    def _send_alert(self, zone_id: str, zone_name: str, level: str, desc: str,
                    data: dict = None, event_type: str = "alert") -> None:
        self._ensure_alert_mgr()
        lvl = getattr(self.AlertLevel, level, self.AlertLevel.INFO)
        msg = self.AlertMessage(
            zone_id=zone_id, zone_name=zone_name, level=lvl,
            event_type=event_type, timestamp=time.time(),
            description=desc, data=data or {},
        )
        dispatched = self._alert_mgr.send_alert(msg)
        data_store.add_alert({
            "zone_id": zone_id, "zone_name": zone_name, "level": level,
            "event_type": event_type,
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
        self._heartbeat_dict: dict[str, float] = mp.Manager().dict()
        self._watchdog: Optional[WorkerWatchdog] = None
        self._watchdog_stop: threading.Event = threading.Event()
        self._start_time: float = 0.0
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
                    "heartbeat_dict": self._heartbeat_dict,
                }
                if wcls in (FallDetectionWorker, ActivityWorker):
                    kwargs["fall_event_queue"] = fall_event_queue
                self.workers.append(wcls(**kwargs))
                self._zone_queues.setdefault(zid, []).append(wq)
                # Initialize heartbeat entry
                self._heartbeat_dict[worker_name] = 0.0

    def start(self) -> None:
        logger.info(f"Starting inference engine: {len(self.workers)} workers, {len(self._zone_queues)} zones")
        self._start_time = time.time()
        for w in self.workers:
            w.start()
        self._orchestrator = _ResultOrchestrator(self, self.output_queue, self.stop_event)
        self._orchestrator.start()
        # Start watchdog
        self._watchdog_stop.clear()
        self._watchdog = WorkerWatchdog(
            heartbeat_dict=self._heartbeat_dict,
            stop_event=self._watchdog_stop,
            engine=self,
        )
        self._watchdog.start()

    def stop(self) -> None:
        logger.info("Stopping inference engine...")
        self.stop_event.set()
        # Stop watchdog first
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None
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

    def restart_worker(self, worker_name: str) -> None:
        """Terminate an unhealthy worker and start a replacement.

        Finds the worker by name, terminates the process, then creates
        and starts a new worker with the same configuration.
        """
        target: Optional[InferenceWorker] = None
        for w in self.workers:
            if w.model_name == worker_name:
                target = w
                break

        if target is None:
            logger.warning("Cannot restart unknown worker '%s'", worker_name)
            return

        logger.info("Restarting worker '%s' (pid=%d)", worker_name, target.pid or 0)
        target.terminate()
        target.join(timeout=5.0)

        # Create replacement with same config
        extra_kwargs: dict[str, Any] = {
            "name": target.model_name,
            "zone_id": target.zone_id,
            "input_queue": target.input_queue,
            "output_queue": target.output_queue,
            "stop_event": target.stop_event,
            "config": target.config,
            "heartbeat_dict": self._heartbeat_dict,
        }
        if hasattr(target, "_fall_event_queue") and target._fall_event_queue is not None:
            extra_kwargs["fall_event_queue"] = target._fall_event_queue
        new_worker = type(target)(**extra_kwargs)

        # Replace in workers list
        idx = self.workers.index(target)
        self.workers[idx] = new_worker
        # Reset heartbeat for this worker
        self._heartbeat_dict[worker_name] = 0.0
        new_worker.start()
        logger.info("Worker '%s' restarted as pid=%d", worker_name, new_worker.pid or 0)

    def get_system_health(self, mqtt_connected: bool = False) -> dict:
        """Return system health including resource metrics and worker status."""
        from pipeline.watchdog import get_system_health as _get_system_health
        worker_status = self._watchdog.get_all_worker_status() if self._watchdog else {}
        return _get_system_health(
            worker_status=worker_status,
            mqtt_connected=mqtt_connected,
            start_time=self._start_time,
        )


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
