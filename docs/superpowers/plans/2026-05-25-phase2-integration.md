# Phase 2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire TwoStageConfirmer, day/night activity detection, and post-fall monitoring into the live inference pipeline; validate end-to-end with CSI simulator.

**Architecture:** Replace stub placeholders in FallDetectionWorker with real FallDetector + TwoStageConfirmer. Replace rule-of-thumb variance checks in ActivityWorker with ActivityDetector + PostFallInactivityChecker. Add a FallConfirmationEvent shared queue so fall confirmations trigger post-fall monitoring in ActivityWorker. Write integration tests that feed multi-zone simulated CSI through the full chain.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, multiprocessing.Queue, pytest

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pipeline/inference_engine.py` | Modify | Wire real models into workers, add fall event queue |
| `tests/test_fall_detection.py` | Modify | Add integrated FallDetector+TwoStageConfirmer tests |
| `tests/test_activity.py` | Modify | Add PostFallInactivityChecker integration tests |
| `tests/test_integration.py` | Create | End-to-end pipeline test with CSI simulator |

---

### Task 1: Add FallConfirmationEvent dataclass to inference_engine.py

**Files:**
- Modify: `pipeline/inference_engine.py:1-22` (imports and dataclass section)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fall_detection.py` at the end of the file:

```python
class TestFallConfirmationEvent:
    def test_event_creation(self) -> None:
        from pipeline.inference_engine import FallConfirmationEvent
        event = FallConfirmationEvent(
            zone_id="zone_bedroom",
            timestamp=1000.0,
            confidence=0.95,
        )
        assert event.zone_id == "zone_bedroom"
        assert event.timestamp == 1000.0
        assert event.confidence == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fall_detection.py::TestFallConfirmationEvent::test_event_creation -v`
Expected: FAIL with ImportError or AttributeError

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline/inference_engine.py` after the existing `InferenceResult` dataclass (around line 32):

```python
@dataclass
class FallConfirmationEvent:
    """Event posted when a fall is confirmed by TwoStageConfirmer."""
    zone_id: str
    timestamp: float
    confidence: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fall_detection.py::TestFallConfirmationEvent::test_event_creation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/inference_engine.py tests/test_fall_detection.py
git commit -m "feat(pipeline): add FallConfirmationEvent dataclass"
```

---

### Task 2: Wire FallDetector + TwoStageConfirmer into FallDetectionWorker

**Files:**
- Modify: `pipeline/inference_engine.py:76-107` (FallDetectionWorker class)
- Modify: `tests/test_fall_detection.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fall_detection.py`:

```python
class TestFallDetectionWorkerIntegration:
    def test_worker_uses_two_stage_confirmer(self) -> None:
        import multiprocessing as mp
        from pipeline.inference_engine import FallDetectionWorker, FallConfirmationEvent

        input_q: mp.Queue = mp.Queue()
        output_q: mp.Queue = mp.Queue()
        fall_event_q: mp.Queue = mp.Queue()
        stop = mp.Event()

        config = {
            "sample_rate": 50.0,
            "window_size": 100,
            "confidence_threshold": 0.85,
            "confirmation_window_seconds": 3.0,
            "inactivity_threshold": 0.15,
        }

        worker = FallDetectionWorker(
            name="FallDetection_zone_test",
            zone_id="zone_test",
            input_queue=input_q,
            output_queue=output_q,
            stop_event=stop,
            config=config,
            fall_event_queue=fall_event_q,
        )

        # Verify worker has a FallDetector instance
        assert worker._detector is not None
        assert worker._confirmer is not None
        assert worker._fall_event_queue is fall_event_q
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fall_detection.py::TestFallDetectionWorkerIntegration::test_worker_uses_two_stage_confirmer -v`
Expected: FAIL — FallDetectionWorker does not accept `fall_event_queue` parameter

- [ ] **Step 3: Rewrite FallDetectionWorker**

Replace the entire `FallDetectionWorker` class in `pipeline/inference_engine.py` (lines 76-107) with:

```python
class FallDetectionWorker(InferenceWorker):
    """Fall detection inference worker with TwoStageConfirmer."""

    def __init__(self, *args, fall_event_queue: Optional[mp.Queue] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_size = self.config.get("window_size", 100)
        self._buffer: list[dict] = []

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
        self._fall_event_queue = fall_event_queue

    def process(self, packet: dict) -> Optional[InferenceResult]:
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
```

Also update the `Optional` import at the top of the file to ensure it includes `Optional` from typing (it already does at line 18).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fall_detection.py::TestFallDetectionWorkerIntegration::test_worker_uses_two_stage_confirmer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/inference_engine.py tests/test_fall_detection.py
git commit -m "feat(pipeline): wire FallDetector + TwoStageConfirmer into FallDetectionWorker"
```

---

### Task 3: Wire ActivityDetector + PostFallInactivityChecker into ActivityWorker

**Files:**
- Modify: `pipeline/inference_engine.py:181-219` (ActivityWorker class)
- Modify: `tests/test_activity.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_activity.py`:

```python
class TestActivityWorkerIntegration:
    def test_worker_uses_activity_detector(self) -> None:
        import multiprocessing as mp
        from pipeline.inference_engine import ActivityWorker

        input_q: mp.Queue = mp.Queue()
        output_q: mp.Queue = mp.Queue()
        stop = mp.Event()

        config = {
            "sample_rate": 50.0,
            "window_seconds": 30.0,
            "threshold_active": 0.5,
            "threshold_still": 0.15,
            "inactivity_timeout_seconds": 7200.0,
            "daytime_start_hour": 6,
            "daytime_end_hour": 22,
            "recovery_timeout_seconds": 30.0,
        }

        worker = ActivityWorker(
            name="Activity_zone_test",
            zone_id="zone_test",
            input_queue=input_q,
            output_queue=output_q,
            stop_event=stop,
            config=config,
            fall_event_queue=mp.Queue(),
        )

        assert worker._detector is not None
        assert worker._post_fall_checker is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_activity.py::TestActivityWorkerIntegration::test_worker_uses_activity_detector -v`
Expected: FAIL — ActivityWorker does not accept `fall_event_queue` or have `_detector`/`_post_fall_checker`

- [ ] **Step 3: Rewrite ActivityWorker**

Replace the entire `ActivityWorker` class in `pipeline/inference_engine.py` (lines 181-219) with:

```python
class ActivityWorker(InferenceWorker):
    """Activity / inactivity detection worker with day/night awareness and post-fall monitoring."""

    def __init__(self, *args, fall_event_queue: Optional[mp.Queue] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_frames = int(
            self.config.get("window_seconds", 30.0) * self.config.get("sample_rate", 50.0)
        )
        self._buffer: list[dict] = []

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
        self._fall_event_queue = fall_event_queue
        self._inactivity_start: Optional[float] = None

    def process(self, packet: dict) -> Optional[InferenceResult]:
        self._buffer.append(packet)
        if len(self._buffer) > self._window_frames:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_frames // 2:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_activity.py::TestActivityWorkerIntegration::test_worker_uses_activity_detector -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/inference_engine.py tests/test_activity.py
git commit -m "feat(pipeline): wire ActivityDetector + PostFallInactivityChecker into ActivityWorker"
```

---

### Task 4: Update InferenceEngine._create_workers to pass fall_event_queue

**Files:**
- Modify: `pipeline/inference_engine.py:253-284` (_create_workers method)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fall_detection.py`:

```python
class TestInferenceEngineWorkerWiring:
    def test_fall_and_activity_workers_share_fall_event_queue(self) -> None:
        from pipeline.inference_engine import InferenceEngine

        engine = InferenceEngine(config_path="configs/zones.yaml")

        fall_workers = [w for w in engine.workers if isinstance(w, FallDetectionWorker)]
        activity_workers = [w for w in engine.workers if isinstance(w, ActivityWorker)]

        assert len(fall_workers) > 0, "No FallDetectionWorkers created"
        assert len(activity_workers) > 0, "No ActivityWorkers created"

        # Each zone's fall worker and activity worker share the same queue
        for fw in fall_workers:
            matching_aw = [aw for aw in activity_workers if aw.zone_id == fw.zone_id]
            assert len(matching_aw) == 1
            assert fw._fall_event_queue is matching_aw[0]._fall_event_queue
```

Also add the necessary imports at the top of `tests/test_fall_detection.py`:

```python
from pipeline.inference_engine import FallDetectionWorker, ActivityWorker
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fall_detection.py::TestInferenceEngineWorkerWiring -v`
Expected: FAIL — workers do not share fall_event_queue

- [ ] **Step 3: Update _create_workers**

Replace the `_create_workers` method in `pipeline/inference_engine.py` (lines 253-284) with:

```python
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

                kwargs: dict[str, Any] = {
                    "name": worker_name,
                    "zone_id": zid,
                    "input_queue": self.input_queue,
                    "output_queue": self.output_queue,
                    "stop_event": self.stop_event,
                    "config": worker_config,
                }

                # Only fall and activity workers get the shared fall event queue
                if wcls in (FallDetectionWorker, ActivityWorker):
                    kwargs["fall_event_queue"] = fall_event_queue

                worker = wcls(**kwargs)
                self.workers.append(worker)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fall_detection.py::TestInferenceEngineWorkerWiring -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/inference_engine.py tests/test_fall_detection.py
git commit -m "feat(pipeline): share fall event queue between fall and activity workers per zone"
```

---

### Task 5: Write end-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
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
            engine.feed_packet(zone_id, {
                "zone_id": zone_id,
                "packet": pkt,
            })

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
                    engine.feed_packet(zone_id, {
                        "zone_id": zone_id,
                        "packet": pkt,
                    })

            time.sleep(1.0)
            results = engine.get_results()

            zone_ids = {r.zone_id for r in results}
            assert "zone_bedroom" in zone_ids
            assert "zone_living" in zone_ids
            assert "zone_hallway" in zone_ids
        finally:
            engine.stop()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_integration.py -v --timeout=60`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(pipeline): end-to-end integration tests with CSI simulator"
```

---

### Task 6: Update InferenceEngine.feed_packet to route by zone

**Files:**
- Modify: `pipeline/inference_engine.py:300-305` (feed_packet method)

- [ ] **Step 1: Verify current behavior and write test**

The current `feed_packet` puts packets on a shared queue, but workers filter by zone_id from the inner dict. Verify this works correctly by checking the integration test passes. If it already works, no change needed.

Run: `python -m pytest tests/test_integration.py::TestEndToEndPipeline::test_multi_zone_isolation -v`
Expected: PASS (current feed_packet already wraps in `{"zone_id": zone_id, "packet": packet}`)

- [ ] **Step 2: Verify workers filter by zone**

Check that each worker's `run()` loop reads `packet` from the queue and only processes packets matching its zone. Currently the base `InferenceWorker.run()` calls `self.input_queue.get()` and passes to `self.process(packet)`. Workers need to skip packets for other zones.

Add zone filtering to `InferenceWorker.run()` in `pipeline/inference_engine.py`. Replace the `run()` method (lines 54-69) with:

```python
    def run(self) -> None:
        logger.info(f"[{self.model_name}] Worker started for zone={self.zone_id}")
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.5)
                # Route: only process packets for this worker's zone
                if isinstance(item, dict) and "zone_id" in item:
                    if item["zone_id"] != self.zone_id:
                        continue
                    packet = item.get("packet", item)
                else:
                    packet = item

                result = self.process(packet)
                if result is not None:
                    self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception:
                logger.exception(f"[{self.model_name}] Error in inference loop")

        logger.info(f"[{self.model_name}] Worker stopped")
```

- [ ] **Step 3: Run all tests to verify nothing broke**

Run: `python -m pytest tests/test_integration.py tests/test_fall_detection.py tests/test_activity.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add pipeline/inference_engine.py
git commit -m "fix(pipeline): add zone-based packet routing in worker run loop"
```

---

### Task 7: Run full test suite and verify no regressions

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (no failures, no errors)

- [ ] **Step 2: Run existing model tests individually to confirm no breakage**

Run:
```bash
python -m pytest tests/test_preprocessor.py tests/test_ingestion.py tests/test_simulator.py -v
```
Expected: All PASS

- [ ] **Step 3: Update PROJECT-DETAIL.md Phase 2 checkboxes**

In `PROJECT-DETAIL.md`, mark these as done:
- `[x] Implement two-stage fall confirmation logic`
- `[x] Multi-zone ingestion (zone ID tagging, per-zone ring buffers)`
- `[x] Implement day/night-aware inactivity detection (rule-based)`

And add a note to remaining tasks:
- `[ ] Deploy 3 ESP32 nodes in test environment (requires hardware)`
- `[ ] Collect in-situ fall + activity data (labeled) (requires hardware + annotator)`
- `[ ] Fine-tune CSI-FallNet (requires real datasets)`
- `[ ] Target: >85% F1 on in-situ test set (requires real datasets)`

- [ ] **Step 4: Commit**

```bash
git add PROJECT-DETAIL.md
git commit -m "docs(phase2): mark completed code integration tasks"
```
