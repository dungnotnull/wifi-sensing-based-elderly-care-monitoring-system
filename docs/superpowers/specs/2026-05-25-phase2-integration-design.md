# Phase 2 Integration Design

**Date:** 2026-05-25
**Scope:** Wire Phase 2 components (TwoStageConfirmer, day/night activity, post-fall monitoring) into the live inference pipeline. Validate end-to-end with CSI simulator.

---

## Context

Phase 2 components are already implemented in the codebase:
- `TwoStageConfirmer` in `models/fall_detection/model.py`
- `ActivityDetector` with day/night awareness in `models/activity/detector.py`
- `PostFallInactivityChecker` in `models/activity/detector.py`
- Multi-zone ingestion in `ingestion/receiver.py`

The gap is integration: these components exist in isolation but are not wired into the inference pipeline. No end-to-end test proves the full chain works.

## Approach

Integration-first: wire all components, run full-stack simulator test, write integration tests.

## Architecture

```
CSI Simulator (3 zones)
    |  MQTT topics: eldercare/csi/{zone_id}
    v
ingestion/receiver.py  (per-zone ring buffers)
    |  validated, zone-tagged packets
    v
pipeline/preprocessor.py  (Hampel, bandpass, normalize)
    |  preprocessed tensors
    v
pipeline/inference_engine.py
    |
    +-- FallDetectionWorker
    |     FallDetector -> TwoStageConfirmer
    |     On confirmation: post FallConfirmationEvent to shared queue
    |
    +-- ActivityWorker
    |     ActivityDetector (day/night aware, receives timestamps)
    |     Reads FallConfirmationEvent -> activates PostFallInactivityChecker (30s)
    |     On 30s no movement: escalate to EMERGENCY
    |
    +-- VitalSignsWorker (unchanged)
    +-- SleepWorker (unchanged)
    |
    v
Alert callback -> alert_manager.py
```

### Key design decisions

1. **FallConfirmationEvent** — typed event posted to a `multiprocessing.Queue` shared between FallDetectionWorker and ActivityWorker. Avoids tight coupling while enabling post-fall monitoring.

2. **Day/night timestamps** — inference engine passes current timestamp to ActivityWorker on each cycle. ActivityDetector already has `is_daytime()` logic; it just needs the timestamp fed in.

3. **Post-fall coordination** — when ActivityWorker receives a FallConfirmationEvent, it starts a 30-second window where it monitors for recovery movement. If none detected, it triggers EMERGENCY alert.

## Files Changed

### Modified

| File | Change |
|---|---|
| `pipeline/inference_engine.py` | Wire TwoStageConfirmer into FallDetectionWorker. Add shared event queue. Feed CSI variance to ActivityWorker. Pass timestamps for day/night logic. Post-fall escalation path. |

### New

| File | Change |
|---|---|
| `tests/test_integration.py` | End-to-end test: CSI simulator -> ingestion -> preprocessing -> inference -> verify fall confirmation, activity states, post-fall escalation, multi-zone isolation. |

### Updated

| File | Change |
|---|---|
| `tests/test_fall_detection.py` | Add tests for TwoStageConfirmer integrated with FallDetector (raw fall -> confirmed fall flow). |
| `tests/test_activity.py` | Add tests for PostFallInactivityChecker integration with FallConfirmationEvent. |

### Not changed

- `models/fall_detection/model.py` — TwoStageConfirmer already complete
- `models/activity/detector.py` — ActivityDetector + PostFallInactivityChecker already complete
- `ingestion/receiver.py` — multi-zone already complete
- `pipeline/preprocessor.py` — signal chain already complete
- No new models, no new config files, no dashboard changes

## Deferred (requires hardware/real data)

- Deploy 3 ESP32 nodes in test environment
- Collect in-situ fall + activity data (labeled)
- Fine-tune CSI-FallNet on real datasets (CSI-Bench, ElderAL-CSI)
- Target >85% F1 on in-situ test set

## Success criteria

1. TwoStageConfirmer wraps FallDetector in inference pipeline — falls require confidence > 0.85 + 3s inactivity check
2. ActivityDetector receives timestamps and suppresses inactivity alerts during sleep hours (10PM-6AM)
3. Post-fall monitoring: confirmed fall triggers 30s watch, no movement escalates to EMERGENCY
4. Multi-zone: 3 simulated zones run simultaneously without cross-contamination
5. All integration tests pass: `pytest tests/test_integration.py -v`
6. All existing tests still pass: `pytest tests/ -v`
