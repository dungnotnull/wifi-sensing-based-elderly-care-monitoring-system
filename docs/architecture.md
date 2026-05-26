# ElderCare -- System Architecture (v0.3)

Design document covering system architecture, data flow, component interactions, and design decisions. Updated for v0.2 with 19 new modules spanning quality validation, persistence, observability, shadow mode, adaptive thresholds, and edge integrations.

---

## 1. High-Level Architecture

```
+--------------------------------------------------------------+
|                      ESP32-S3 NODES                           |
|  (up to 3 zones) CSI capture @ 50Hz, MQTT publish            |
+-------+----------------------------+-------------------------+
        | MQTT                        | MQTT
        | eldercare/csi/              | eldercare/csi/
        | zone_bedroom                | zone_living
        v                             v
+--------------------------------------------------------------+
|               MOSQUITTO MQTT BROKER                           |
|  (topic routing, QoS 1, persistent sessions)                 |
+------------------------------+-------------------------------+
                               |
+------------------------------v-------------------------------+
|                 INGESTION LAYER                                |
|  ingestion/receiver.py                                       |
|  - Per-zone MQTT subscription                               |
|  - JSON + binary packet parsing                             |
|  - Ring buffers (5s @ 50Hz = 250 frames)                    |
|  - CSIRingBuffer + IngestionStats per zone                   |
+------------------------------+-------------------------------+
                               | per-zone CSI packets
+------------------------------v-------------------------------+
|            CSI QUALITY CHECKER                                 |
|  pipeline/csi_quality.py                                     |
|  - Per-packet SNR estimation                                |
|  - Packet-loss detection (sequence gaps)                     |
|  - Null / stuck subcarrier detection                         |
|  - Rolling averages (60s window)                             |
|  - Quality flags: GOOD / DEGRADED / POOR                     |
+------------------------------+-------------------------------+
                               | quality-gated packets
+------------------------------v-------------------------------+
|               PREPROCESSING PIPELINE                           |
|  pipeline/preprocessor.py                                    |
|  - Hampel filter (outlier removal)                           |
|  - Butterworth bandpass (4th order)                          |
|  - Phase sanitization (unwrap + linear detrend)              |
|  - Per-subcarrier z-score normalization (EMA)                |
|  - Edge subcarrier removal (+-2)                             |
+------------------------------+-------------------------------+
                               | preprocessed (amplitude, phase)
+------------------------------v-------------------------------+
|               INFERENCE ENGINE                                |
|  pipeline/inference_engine.py (multiprocessing)              |
|                                                               |
|  ModelRegistry (pipeline/model_registry.py):                  |
|    Config-driven per-zone model selection                     |
|                                                               |
|  Per zone (x3):                                               |
|  +----------------------------------------------------------+|
|  | FallDetectionWorker                                       ||
|  |  CSI-FallNet (1D-CNN -> BiLSTM -> Attention)              ||
|  |  + TemperatureScaling calibration                         ||
|  |  + ConfidenceSmoother (window=3)                          ||
|  |  + TwoStageConfirmer (0.85 conf + 3s inactivity)          ||
|  |  -> FallConfirmationEvent to shared queue                 ||
|  +----------------------------------------------------------+|
|  | VitalSignsWorker                                          ||
|  |  VitalsAdapter (wifi_densepose Rust PyO3)                 ||
|  |  Python fallback (scipy FFT) if Rust unavailable          ||
|  |  BreathingExtractor + HeartRateExtractor                  ||
|  |  -> respiration_bpm, heart_rate_bpm                       ||
|  +----------------------------------------------------------+|
|  | SleepWorker                                               ||
|  |  SleepFeatureExtractor -> SleepLSTM                       ||
|  |  n_features=5, FocalLoss, balanced training               ||
|  |  -> sleep_stage, SleepScorer (0-100)                      ||
|  +----------------------------------------------------------+|
|  | ActivityWorker                                            ||
|  |  ActivityDetector (rule-based, day/night)                 ||
|  |  + PostFallInactivityChecker (30s recovery)               ||
|  |  -> active/still/inactivity, alert level                  ||
|  +----------------------------------------------------------+|
|                                                               |
|  12 total workers (3 zones x 4 models)                       |
|                                                               |
|  +----------------------------------------------------------+|
|  | WorkerWatchdog (pipeline/watchdog.py)                     ||
|  |  - Heartbeat checks every 30s                             ||
|  |  - Auto-restart dead workers                              ||
|  |  - System health (CPU, memory, queue depth)               ||
|  +----------------------------------------------------------+|
|                                                               |
|  +----------------------------------------------------------+|
|  | AdaptiveThreshold (pipeline/adaptive_thresholds.py)       ||
|  |  - Observes per-zone metric variance                      ||
|  |  - Auto-tunes detection thresholds                        ||
|  +----------------------------------------------------------+|
+------------------------------+-------------------------------+
                               | output queue (InferenceResult)
| +----------------------------------------------------------+ |
| |          CORRELATION TRACKER (end-to-end)                 | |
| |  pipeline/correlation.py                                  | |
| |  - Assigns correlation_id at ingestion                    | |
| |  - Tracks latency through each pipeline stage             | |
| |  - Aggregates per-zone latency statistics                 | |
| +----------------------------------------------------------+ |
                               |
+------------------------------v-------------------------------+
|            _RESULTORCHESTRATOR (thread)                        |
|  Consumes InferenceResult from output queue                   |
|  - Updates InferenceDataStore                                |
|  - Checks Shadow Mode gate before alert dispatch             |
|  - Dispatches alerts via AlertManager (if not shadow)        |
|  - Checks daily summary schedule (7:00 AM)                   |
+-----+--------------------------------+-----------------------+
      |                                |
      |    +---------------------------+
      |    | SHADOW MODE GATE          |
      |    | pipeline/shadow_mode.py   |
      |    | - Suppress alerts         |
      |    | - Log events              |
      |    | - Label ground truth      |
      |    | - Track FP/FN counts      |
      |    +-------+-------------------+
      |            |
+-----v----------+ |  +------------------------------------------+
| AlertManager   |<-+  | InferenceDataStore                       |
| alerts/        |      | pipeline/data_store.py                   |
|                |      |                                          |
| - Telegram     |      | - ZoneStatus (per zone)                  |
|   dispatch     |      | - VitalsHistory (3600 pts)               |
| - 3-level      |      | - Alerts (500 recent)                    |
|   cooldown     |      | - SleepRecords (90 nights)               |
| - VN messages  |      | - Thread-safe (lock)                     |
| - File log     |      +----+------------------+------------------+
| - Daily        |           |                  |
|   summary      |           |                  |
+----------------+           |                  |
                             v                  v
               +-----------------------------+  +---------------------------+
               | SQLite Persistence          |  | InfluxDB Writer            |
               | pipeline/persistence.py     |  | pipeline/influx_writer.py  |
               | - Write-behind (1s batch)   |  | - Batch writes (5s)        |
               | - WAL mode for concurrent   |  | - Graceful degradation     |
               |   reads                     |  |   if down                  |
               | - Restore latest state on   |  +---------------------------+
               |   startup                   |
               +-------------+---------------+
                             |
               +-------------v-------------------------------+
               | Dashboard (FastAPI)                          |
               | dashboard/backend/main.py                    |
               |                                              |
               | REST endpoints:                              |
               |   GET /api/zones                             |
               |   GET /api/vitals?zone_id=...                |
               |   GET /api/alerts?limit=50                   |
               |   GET /api/sleep?zone_id=...                 |
               |   GET /api/daily-summary                     |
               |   GET /api/health                            |
               |   GET /api/events (SSE stream)               |
               |   GET /api/telemetry                         |
               |   GET /api/correlation/traces                |
               |   GET /api/correlation/stats                 |
               |   GET /api/shadow-mode/report                |
               |   POST /api/shadow-mode/go-live              |
               |   GET /api/csi-quality/{zone_id}             |
               |   GET /api/adaptive-thresholds               |
               +-------------+-------------------------------+
                             | HTTP + SSE
               +-------------v-------------------------------+
               | React Frontend                                |
               | dashboard/frontend/                           |
               | - Zone cards (per-room status)                |
               | - Vitals history graphs                       |
               | - Sleep stage charts                          |
               | - Alert log (live via SSE)                    |
               | - System health panel                         |
               | - 16px minimum font                           |
               +----------------------------------------------+

    +-----------------------------------------------------------+
    |                 ANCILLARY MODULES                           |
    |                                                             |
    |  CSIRecorder / Replayer / Labeler                           |
    |    pipeline/record_replay.py                                |
    |    - Record live CSI streams to disk                        |
    |    - Replay recorded sessions for testing                   |
    |    - Annotate ground truth labels                           |
    |                                                             |
    |  HomeAssistant Discovery                                    |
    |    pipeline/homeassistant.py                                |
    |    - MQTT auto-discovery for HA entities                    |
    |    - Binary sensor (fall), sensor (vitals, sleep)           |
    |                                                             |
    |  Model Quantizer                                            |
    |    pipeline/quantization.py                                 |
    |    - INT8 post-training static quantization                 |
    |    - Size/speed benchmarks before deployment                |
    |                                                             |
    |  Telemetry Collector                                        |
    |    pipeline/telemetry.py                                    |
    |    - Local metrics collection (Prometheus format)           |
    |    - Pipeline latency, throughput, error rates              |
    |                                                             |
    |  Model Calibration                                          |
    |    models/calibration.py                                    |
    |    - Temperature scaling for confidence calibration         |
    +-----------------------------------------------------------+
```

---

## 2. Data Flow

### 2.1 CSI Packet Lifecycle

```
ESP32-S3 firmware (esp-idf CSI API)
  |  Reads CSI at 50 Hz (HT20, 52 subcarriers)
  |  Packs: {timestamp, sequence, csi_amplitude[52], csi_phase[52], rssi}
  v
MQTT publish to eldercare/csi/{zone_id}
  |  JSON or binary struct encoding
  v
Ingestion (receiver.py)
  |  CSIIngestion._on_message -> _parse_packet -> _validate_packet
  |  Pushes to CSIRingBuffer (250 frames, 5s window)
  |  Calls registered callbacks
  v
CSI Quality Checker (csi_quality.py)
  |  Estimates SNR per packet
  |  Detects sequence gaps (packet loss)
  |  Flags null / stuck subcarriers
  |  Rolling 60s averages
  v
Preprocessor (preprocessor.py)
  |  Preprocessor.process(amplitude[100,52], phase[100,52])
  |  Hampel -> Butterworth -> Phase sanitize -> Z-score -> Edge removal
  |  Returns (processed_amp[100,48], processed_phase[100,48])
  v
InferenceEngine.feed_packet()
  |  Fans out one copy per worker queue (4 queues per zone)
  v
Workers process independently, post InferenceResult to output queue
  |
  v
_ResultOrchestrator thread
  |  Updates data store
  |  Checks shadow mode gate
  |  Dispatches alerts (if not in shadow mode)
  |  Checks daily summary schedule
```

### 2.2 Fall Detection Flow (Two-Stage)

```
Stage 1: Model inference
  CSI amplitude (T=100, C=52) -> FallDetector
  -> raw logits (2 classes)

Calibration:
  TemperatureScaling scales logits to calibrated probabilities
  -> calibrated_confidence in [0,1]

Smoothing:
  ConfidenceSmoother maintains sliding window (size=3)
  -> smoothed_confidence = mean(recent 3 confidence values)

Stage 2: Confirmation
  IF smoothed_confidence >= 0.85:
    -> _pending_confirmation = True
    -> Wait for next 3 seconds of CSI data
    -> Check CSI variance < inactivity_threshold (0.15)
    -> YES: confirm fall -> post FallConfirmationEvent
    -> NO: dismiss as false positive

FallConfirmationEvent
  -> ActivityWorker.post_fall_checker.on_fall_detected()
  -> Checks 30s recovery window
  -> No movement -> "EMERGENCY"
  -> Movement detected -> reset
```

### 2.3 Vital Signs Flow

```
Per subcarrier amplitude residuals -> wifi_densepose Rust extractors

Primary path (Rust PyO3):
  BreathingExtractor (0.1-0.5 Hz bandpass, zero-crossing)
    -> VitalEstimate {value_bpm, confidence, status}

  HeartRateExtractor (0.8-2.0 Hz bandpass, autocorrelation)
    -> VitalEstimate {value_bpm, confidence, status}

Fallback path (Python/scipy, if Rust unavailable):
  PythonFallbackVitals (models/vital_signs/python_fallback.py)
    -> scipy.signal FFT-based frequency extraction
    -> Same output format: VitalEstimate {value_bpm, confidence, status}

VitalsAdapter.feed_frame() per frame
  -> Respiration/HR updated every 5 seconds
  -> Stored in InferenceDataStore.vitals_history
```

### 2.4 Sleep Monitoring Flow

```
CSI data accumulates in SleepWorker buffer (60 * 50 = 3000 frames per minute)

SleepFeatureExtractor.extract(amplitude[N, C], respiration_bpm)
  -> 5 features: [mean_respiration, respiration_std, movement_index,
                  burst_count, signal_quality]
    (signal_quality sourced from CSIQualityChecker metrics)

SleepLSTM(features[1,5])
  -> probs: [p_awake, p_light, p_deep]
  -> argmax -> sleep stage

Training improvements (v0.2):
  - FocalLoss addresses class imbalance (deep sleep underrepresented)
  - Balanced class sampling during training
  - n_features=5 (was 4), added signal quality

Nightly accumulation:
  -> Epochs collected in _night_epochs (max 480 = 8 hours)
  -> When >= 240 epochs (4 hours):
    SleepScorer.compute_score(stage_probs, session_hours)
    -> 0-100 score from: deep% (40%) + efficiency% (40%) + weighted_quality (20%)
```

### 2.5 Alert Dispatch Flow

```
_ResultOrchestrator._handle_result(InferenceResult)

  Shadow mode gate check:
    IF shadow_mode.is_active():
      -> Log event with correlation_id
      -> Skip alert dispatch
      -> Record for FP/FN tracking
    ELSE:
      -> Proceed with normal alert dispatch

fall_detection result:
  if fall_detected:
    -> AlertManager.send_emergency("Phat hien te nga!")
    -> data_store.update_fall(zone, True, confidence)
    -> data_store.add_alert(EMERGENCY)

vital_signs result:
  -> data_store.update_vitals(zone, bpm, conf, hr, hrconf)

sleep result:
  -> data_store.update_sleep(zone, stage, score)

activity result:
  -> data_store.update_activity(zone, state, alert)
  if alert == "WARNING":
    -> AlertManager.send_warning("Khong hoat dong keo dai")
  if post_fall_alert == "EMERGENCY":
    -> AlertManager.send_emergency("KHAN CAP: Khong cu dong sau te nga!")

AlertManager.send_alert():
  1. Cooldown check (per level + zone)
  2. Telegram dispatch (if configured)
  3. Log to data/alerts.log
  4. Add to data_store._alerts
```

### 2.6 CSI Quality Validation Flow

```
Every incoming CSI packet is evaluated by CSIQualityChecker:

  SNR estimation:
    signal_power = mean(|amplitude|^2)
    noise_power = var(amplitude_diff) along time axis
    snr_db = 10 * log10(signal_power / noise_power)

  Packet loss detection:
    Check sequence_number gap from previous packet
    loss_rate = gaps / expected_packets (rolling 60s)

  Null / stuck subcarrier detection:
    subcarrier_std = std(amplitude[:, sub]) over rolling window
    flag if std < stuck_threshold (default 1e-6)

  Quality classification:
    GOOD:     snr > 15 dB AND loss_rate < 0.05 AND null_subs == 0
    DEGRADED: snr > 8 dB  OR  loss_rate < 0.15  OR  null_subs < 3
    POOR:     anything worse

  Action:
    GOOD / DEGRADED -> packet passes through to preprocessing
    POOR -> packet dropped, warning logged
    Quality metrics exposed via /api/csi-quality/{zone_id}
```

### 2.7 Persistence Flow

```
InferenceResult events flow through two persistence paths:

SQLite (pipeline/persistence.py):
  - Write-behind buffer: collect up to 50 events or 1 second
  - Batch INSERT into SQLite WAL-mode database
  - Tables: events, vitals_snapshot, sleep_sessions, system_state
  - On startup: restore latest ZoneStatus and alert history from DB
  - Enables crash recovery and historical queries
  - File: data/eldercare.db

InfluxDB (pipeline/influx_writer.py):
  - Batch writes every 5 seconds
  - Time-series points: vitals_bpm, heart_rate_bpm, sleep_stage,
    movement_index, signal_quality, alert_fired
  - Tags: zone_id, model_name
  - Graceful degradation: if InfluxDB unreachable, buffer in memory
    (up to 10,000 points), retry with exponential backoff
  - Optional: only enabled if configured in configs/models.yaml
```

### 2.8 Correlation Tracking Flow

```
CorrelationTracker assigns and propagates a correlation_id
through the entire pipeline:

  Ingestion:
    receiver.py assigns correlation_id (UUID) to each CSI batch
    Records: {correlation_id, zone_id, ingest_timestamp}

  Preprocessing:
    Records: {correlation_id, preprocess_start, preprocess_end}
    Computes: preprocess_latency_ms

  Inference:
    Each worker records: {correlation_id, worker_type, inference_start, inference_end}
    Computes: inference_latency_ms per model

  Orchestration:
    Records: {correlation_id, orchestrate_timestamp}
    Computes: end_to_end_latency_ms = orchestrate - ingest

  Statistics aggregation:
    Rolling 5-minute windows, per zone and per model:
    - p50, p95, p99 latency
    - throughput (events/second)
    - error count

  API exposure:
    GET /api/correlation/traces?correlation_id=...
    GET /api/correlation/stats?zone_id=...
```

### 2.9 Shadow Mode Flow

```
ShadowMode allows gradual rollout and validation before going live:

  Activation:
    Config-driven: shadow_mode.enabled = true in configs/models.yaml
    Or API: POST /api/shadow-mode/go-live (transitions to live mode)

  Behavior when active:
    1. Inference runs normally (all models process data)
    2. All events logged to shadow_events table in SQLite
    3. Alert dispatch is suppressed (no Telegram, no logs)
    4. Caregiver UI shows shadow-mode banner
    5. Ground truth labeling interface available

  Ground truth labeling:
    CSIRecorder captures live sessions
    CSIReplayer replays them for manual annotation
    CSILabeler stores annotations: {correlation_id, true_label, annotator}
    Compares model predictions vs. ground truth

  Reporting:
    FP rate, FN rate, precision, recall per model
    Confidence calibration error
    Latency statistics
    GET /api/shadow-mode/report returns full analysis

  Go-live decision:
    Review report, confirm metrics meet thresholds
    POST /api/shadow-mode/go-live transitions to production mode
    All subsequent events trigger real alerts
```

### 2.10 Adaptive Threshold Flow

```
AdaptiveThreshold observes per-zone metric distributions and
auto-tunes detection thresholds:

  Metrics observed (rolling 30-minute window):
    - Per-zone CSI variance
    - Per-zone vital signs baseline (respiration, HR)
    - Activity level distribution
    - Fall detector confidence distribution

  Adaptation rules:
    IF variance(zone_csi) increases by > 2x baseline:
      -> Raise fall_confidence_threshold (e.g., 0.85 -> 0.90)
      -> Widen inactivity confirmation window

    IF baseline_respiration shifts > 20%:
      -> Update vital_signs normal range
      -> Adjust WARNING thresholds

    IF false_positive_rate(last_24h) > 0.1:
      -> Increase confidence threshold for trigger model
      -> Log threshold change with justification

  Constraints:
    Minimum thresholds enforced (never drop below safety floor)
    Changes logged with old_value, new_value, reason
    Manual override via configs/thresholds.yaml always wins
    GET /api/adaptive-thresholds returns current values + history
```

---

## 3. Component Design

### 3.1 InferenceEngine

| Aspect | Detail |
|---|---|
| Pattern | Fan-out with per-worker input queues |
| Process model | `multiprocessing.Process` (12 workers) |
| Inter-worker comm | `mp.Queue` (input + output queues) |
| Fall to Activity bridge | Per-zone `fall_event_queue` (FallWorker to ActivityWorker) |
| Result consumption | `_ResultOrchestrator` thread reads output queue |
| GIL handling | Each worker is a separate process -- no GIL contention |
| Model loading | `ModelRegistry` provides config-driven per-zone model selection |

### 3.2 InferenceDataStore

| Aspect | Detail |
|---|---|
| Thread safety | `threading.Lock` on all mutations |
| Vitals history | `deque(maxlen=3600)` per zone (~1 hour at 1/sec) |
| Alerts | List, capped at 500 |
| Sleep records | List, capped at 90 nights |
| Zone status | Dict of `ZoneStatus` dataclasses |
| Persistence | Writes through to SQLitePersistence, reads from memory |

### 3.3 AlertManager

| Aspect | Detail |
|---|---|
| Cooldown key | `{level}_{zone_id}` |
| Cooldown periods | EMERGENCY: 60s, WARNING: 300s, INFO: 3600s |
| Telegram | Optional -- skips silently if not configured |
| Log persistence | `data/alerts.log` (pipe-delimited) |
| VN formatting | `AlertMessage.format_vn()` with emoji + datetime |
| Shadow gate | Checks `ShadowMode.is_active()` before dispatch |

### 3.4 Daily Summary Generator

| Aspect | Detail |
|---|---|
| Sections | Night summary, current status, 24h alerts, daily advice |
| Data source | `InferenceDataStore` (real) or dummy mode |
| Dummy mode | Plausible mock data for testing/demo |
| Language | Vietnamese |
| Schedule | Checked every event loop cycle, fires at `alerts.yaml: daily_summary.time_local` (default 07:00) |

### 3.5 WorkerWatchdog

| Aspect | Detail |
|---|---|
| Module | `pipeline/watchdog.py` |
| Heartbeat interval | 30 seconds per worker |
| Detection | Worker process `is_alive()` check + last-result timestamp |
| Auto-restart | Terminates dead process, spawns replacement with same config |
| Health metrics | CPU usage, memory usage, queue depth per worker |
| Alerting | Logs WARNING on restart, EMERGENCY if >3 restarts in 5 minutes |
| Thread model | Runs in `_ResultOrchestrator` thread alongside result handling |

### 3.6 SQLitePersistence

| Aspect | Detail |
|---|---|
| Module | `pipeline/persistence.py` |
| Write strategy | Write-behind buffer: batch INSERT every 1 second or 50 events |
| Journal mode | WAL (Write-Ahead Logging) for concurrent read/write |
| Tables | events, vitals_snapshot, sleep_sessions, system_state, shadow_events |
| Crash recovery | On startup, restores latest ZoneStatus and alert history from DB |
| File location | `data/eldercare.db` (gitignored) |
| Cleanup | Retention policy: events > 30 days archived, > 90 days deleted |

### 3.7 InfluxWriter

| Aspect | Detail |
|---|---|
| Module | `pipeline/influx_writer.py` |
| Write interval | 5-second batch flush |
| Points | vitals_bpm, heart_rate_bpm, sleep_stage, movement_index, signal_quality, alert_fired |
| Tags | zone_id, model_name, correlation_id |
| Degradation | If InfluxDB unreachable, buffer up to 10,000 points in memory |
| Retry | Exponential backoff: 5s, 15s, 45s, then disable for 5 minutes |
| Optional | Only enabled if `influxdb.url` is set in configs/models.yaml |

### 3.8 CSIQualityChecker

| Aspect | Detail |
|---|---|
| Module | `pipeline/csi_quality.py` |
| Metrics per packet | SNR (dB), packet loss rate, null/stuck subcarrier count |
| Windowing | Rolling 60-second averages per zone |
| Classification | GOOD / DEGRADED / POOR |
| Action | Drops POOR packets, passes GOOD/DEGRADED through |
| API exposure | `GET /api/csi-quality/{zone_id}` returns current + historical metrics |
| Integration | Quality flag attached to correlation context for tracing |

### 3.9 ModelRegistry

| Aspect | Detail |
|---|---|
| Module | `pipeline/model_registry.py` |
| Config source | `configs/models.yaml` per-zone overrides |
| Default behavior | If no per-zone config, uses global model settings |
| Supported overrides | checkpoint_path, device, batch_size, confidence_threshold |
| Hot reload | Watches config file for changes, reloads without restart |
| Fallback | Falls back to default model if zone-specific checkpoint missing |

### 3.10 CorrelationTracker

| Aspect | Detail |
|---|---|
| Module | `pipeline/correlation.py` |
| ID format | UUID v4, assigned at ingestion |
| Tracking points | Ingestion, preprocessing, inference (per model), orchestration |
| Latency stats | p50, p95, p99 per zone and per model (5-minute rolling) |
| Storage | In-memory ring buffer (10,000 traces) + SQLite for persistence |
| API | `GET /api/correlation/traces` and `GET /api/correlation/stats` |

### 3.11 AdaptiveThreshold

| Aspect | Detail |
|---|---|
| Module | `pipeline/adaptive_thresholds.py` |
| Observation window | 30-minute rolling |
| Safety floor | Minimum thresholds enforced (from configs/thresholds.yaml) |
| Override | Manual config values always take precedence |
| Logging | Every threshold change logged with old/new values and reason |
| API | `GET /api/adaptive-thresholds` returns current + adjustment history |

### 3.12 CSIRecorder / Replayer / Labeler

| Aspect | Detail |
|---|---|
| Module | `pipeline/record_replay.py` |
| Recorder | Captures raw CSI packets to JSONL files (one per session) |
| Replayer | Reads JSONL, publishes to internal pipeline at configurable speed |
| Labeler | TUI tool for annotating ground truth on recorded sessions |
| Output | Annotations stored in SQLite: {correlation_id, true_label, timestamp, annotator} |
| Use case | Shadow mode validation, model fine-tuning data collection |

### 3.13 HomeAssistantDiscovery

| Aspect | Detail |
|---|---|
| Module | `pipeline/homeassistant.py` |
| Protocol | MQTT Home Assistant auto-discovery (config topics) |
| Entities | Binary sensor (fall), sensor (respiration, heart_rate, sleep_score, activity) |
| Topics | `homeassistant/binary_sensor/eldercare/{zone_id}/config` etc. |
| State updates | Published on every InferenceResult cycle |
| Optional | Only enabled if `homeassistant.enabled = true` in configs/models.yaml |

### 3.14 ModelQuantizer

| Aspect | Detail |
|---|---|
| Module | `pipeline/quantization.py` |
| Method | INT8 post-training static quantization (PyTorch) |
| Calibration | Uses 100 representative batches from calibration dataset |
| Benchmark | Compares original vs. quantized: size, inference time, accuracy delta |
| Output | Quantized model saved as `{model_name}_int8.pt` |
| Target | Raspberry Pi 5 deployment -- keep models < 5M params, minimize latency |

### 3.15 ShadowMode

| Aspect | Detail |
|---|---|
| Module | `pipeline/shadow_mode.py` |
| Activation | Config: `shadow_mode.enabled = true` |
| Deactivation | API: `POST /api/shadow-mode/go-live` |
| Alert behavior | All alert dispatch suppressed; events logged to shadow_events table |
| FP/FN tracking | Compares predictions against ground truth labels |
| Report | Precision, recall, F1 per model; confidence calibration error; latency stats |
| Dashboard | Shows "SHADOW MODE" banner; events visible but marked as shadow |

### 3.16 TelemetryCollector

| Aspect | Detail |
|---|---|
| Module | `pipeline/telemetry.py` |
| Format | Prometheus-compatible metrics (exposed via `/api/telemetry`) |
| Metrics | Pipeline latency, throughput (events/s), error rate, queue depth, |
|         | Worker health, model inference time, CSI quality score |
| Aggregation | Per-zone and global rollups |
| Storage | In-memory counters + histograms, flushed to SQLite periodically |
| Scrape | Can be scraped by Prometheus or read via API |

---

## 4. Model Architecture Details

### 4.1 CSI-FallNet (FallDetector)

```
Input: (B, T=100, C=52)

Conv1D(52->64, k5) + BN + ReLU
Conv1D(64->128, k3) + BN + ReLU + MaxPool(2) -> (B, 50, 128)
Conv1D(128->128, k3) + BN + ReLU + MaxPool(2) -> (B, 25, 128)

BiLSTM(128->256, bidirectional) -> (B, 25, 512)
AttentionPooling -> (B, 512)

FC(512->256) + Dropout(0.5) + ReLU
FC(256->64) + ReLU
FC(64->2) -> raw logits

Post-processing (v0.2):
  TemperatureScaling(logits) -> calibrated probabilities
    Learned temperature parameter T, optimized on validation set
    Cross-entropy with T-scaled softmax

  ConfidenceSmoother(window=3) -> smoothed confidence
    Maintains sliding buffer of last 3 confidence values
    Output: mean of buffer (reduces flicker from transient spikes)

  Softmax -> {non-fall, fall}
  calibrated_confidence = softmax[fall_class]

Parameters: 2,607,235 (2.6M) + 1 (temperature)
```

### 4.2 SleepLSTM

```
Input: (B, N_epochs, 5) -- 5 features per 1-minute epoch
  Features: [mean_respiration, respiration_std, movement_index,
             burst_count, signal_quality]
  (v0.2: added signal_quality from CSIQualityChecker)

LSTM(5->64, 2 layers, dropout=0.3) -> (B, N_epochs, 64)
FC(64->32) + ReLU + Dropout(0.3)
FC(32->3) + Softmax -> {awake, light, deep}

Training improvements (v0.2):
  Loss: FocalLoss (gamma=2.0, alpha=[0.3, 0.3, 0.4])
    Addresses class imbalance (deep sleep epochs underrepresented)
  Sampling: Balanced class sampler during training
    Ensures equal representation of awake/light/deep per batch
  Features: n_features=5 (was 4), added signal_quality

Parameters: 55,427
```

### 4.3 VitalsAdapter

```
Primary path -- wraps wifi_densepose Rust-native extractors (PyO3):

  BreathingExtractor.esp32_default()  -- 56 subcarriers, 100 Hz, 30s window
  HeartRateExtractor.esp32_default()  -- 56 subcarriers, 100 Hz, 15s window

  Stateful: feed one frame at a time, GIL released during DSP loop.

Fallback path (v0.2) -- pure Python/scipy:
  PythonFallbackVitals (models/vital_signs/python_fallback.py)

  Respiration estimation:
    scipy.signal.butter(4, [0.1, 0.5], btype='band') bandpass filter
    FFT peak detection in 0.1-0.5 Hz range
    -> respiration_bpm = dominant_frequency * 60

  Heart rate estimation:
    scipy.signal.butter(4, [0.8, 2.0], btype='band') bandpass filter
    Autocorrelation-based period detection
    -> heart_rate_bpm = 60 / dominant_period

  Auto-activation:
    IF import of wifi_densepose fails:
      -> Log WARNING
      -> Instantiate PythonFallbackVitals
      -> Same interface (feed_frame, get_estimate)
      -> Slightly higher latency, no Rust optimization
```

---

## 5. Dataset Mappers

### 5.1 ElderAL-CSI Mapper

```
Input: CSV files (2 TX x 3 RX x 512 subcarriers = 3076 cols)
  [0-3] Metadata: activityID, sujectID, positionID, timestamp
  [4-3075] Amplitude columns: amp_tx{0,1}_rx{0,1,2}_sub{0..511}

Pipeline:
  load_elderal_csv()
    -> Select TX-RX pair (e.g. tx0_rx0)
    -> Downsample 512 -> 52 subcarriers
    -> Extract label from path (action2_fall_new/ -> 1)

  sliding_windows()
    -> window_size=100, stride=50
    -> Output: (N, 100, 52) with labels

  load_elderal_directory()
    -> Bulk CSV loading
    -> Shuffle
    -> 80/20 train/val split
```

### 5.2 CSI-Bench Mapper

```
Input: .npz (keys: data, labels) -- shape (N, 100, 52) int64 labels

Fallback: .mat adapter
  preprocess_csibench_mat()
    -> scipy.io.loadmat
    -> Print available keys for manual configuration
    -> amplitude extraction -> subcarrier selection -> sliding windows
    -> Output shape: (N, 100, 52)
```

---

## 6. Dashboard API

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/zones` | Current status for all zones |
| GET | `/api/vitals?zone_id=...` | Vitals history for a zone |
| GET | `/api/alerts?limit=50` | Recent alert history |
| GET | `/api/sleep?zone_id=...` | Sleep data for a zone |
| GET | `/api/daily-summary` | Latest daily summary |
| GET | `/api/health` | System health check |

### v0.2 Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/events` | SSE stream of real-time inference events |
| GET | `/api/telemetry` | Prometheus-format metrics for scraping |
| GET | `/api/correlation/traces?correlation_id=...` | Full trace for a specific correlation ID |
| GET | `/api/correlation/stats?zone_id=...` | Aggregated latency/throughput statistics |
| GET | `/api/shadow-mode/report` | Shadow mode analysis (FP, FN, precision, recall) |
| POST | `/api/shadow-mode/go-live` | Transition from shadow mode to production |
| GET | `/api/csi-quality/{zone_id}` | CSI signal quality metrics per zone |
| GET | `/api/adaptive-thresholds` | Current adaptive threshold values + history |

### SSE Event Format

```
GET /api/events
Content-Type: text/event-stream

event: inference
data: {"zone_id": "zone_bedroom", "model": "fall_detection", "result": {...},
       "correlation_id": "...", "timestamp": "..."}

event: alert
data: {"level": "WARNING", "zone_id": "zone_living", "message": "..."}

event: health
data: {"workers_alive": 12, "queue_depth": 3, "cpu_pct": 45.2}
```

---

## 7. Configuration Files

| File | Purpose |
|---|---|
| `configs/zones.yaml` | Zone definitions (ID, name, MAC, MQTT topic) |
| `configs/thresholds.yaml` | Detection thresholds (inactivity, fall, vitals, sleep, activity) |
| `configs/alerts.yaml` | Alert cooldowns, Telegram settings, daily summary time |
| `configs/models.yaml` | Checkpoint paths, device, input shapes, per-zone overrides, shadow mode, quantization |
| `.env` | Secrets (Telegram token, chat IDs, InfluxDB credentials) |

---

## 8. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| CSI Capture | ESP32-S3, esp-idf CSI API | Low cost, documented, 50 Hz capture |
| Transport | MQTT (Mosquitto) | Per-zone topic routing, persistent sessions |
| Signal Processing | Python/scipy | Faithful reimplementation of RuView's Rust chain |
| Vitals Extraction | wifi_densepose (Rust PyO3) | Battle-tested RuView algorithms, GIL-released |
| Fallback Vitals | Python/scipy FFT | Graceful degradation if Rust unavailable |
| Deep Learning | PyTorch (CNN, BiLSTM, LSTM) | Standard ML framework, INT8 quantization |
| Calibration | Temperature scaling | Confidence calibration for fall detector |
| API Server | FastAPI + uvicorn | Async, auto-docs, production-ready, SSE support |
| Frontend | React | Responsive, caregiver-friendly |
| Alerting | python-telegram-bot | Free, reliable, Vietnamese support |
| Persistence | SQLite (WAL) | Local, zero-config, crash recovery |
| Time-series | InfluxDB (optional) | Metrics storage for historical analysis |
| Container | Docker + docker-compose | Reproducible deployment |
| Home Automation | Home Assistant MQTT discovery | Integration with smart home ecosystem |
| Telemetry | Prometheus format | Standard observability, local collection |

---

## 9. Design Decisions

### 9.1 MQTT over UDP
RuView's Rust server uses UDP for CSI transport. ElderCare uses MQTT because:
- Topic-based routing gives natural per-zone isolation (`eldercare/csi/{zone_id}`)
- QoS guarantees prevent packet loss
- Mosquitto runs on the same server -- no extra hardware
- ESP32 has native MQTT support in esp-idf

### 9.2 Custom Preprocessor (not RuView import)
RuView's signal processing chain lives in the Rust server binary, not the Python package. Reimplementing in scipy:
- No Rust toolchain dependency for the Python inference server
- Easier to modify and debug (Python vs Rust FFI)
- Functionally equivalent -- same Hampel, Butterworth, phase sanitization algorithms

### 9.3 Stateful Vitals Extraction
The `wifi_densepose` `BreathingExtractor` and `HeartRateExtractor` are stateful (internal circular buffers). The `VitalsAdapter` feeds one frame at a time -- no batching needed. This allows real-time streaming inference. A Python fallback using scipy FFT is available when the Rust package is not installed.

### 9.4 Multiprocessing over Threading
Workers run as `multiprocessing.Process` (not threads) because:
- PyTorch inference benefits from separate processes (no GIL)
- A crashed worker doesn't bring down the orchestrator
- Per-worker queues avoid shared memory contention

### 9.5 Lazy Model Initialization
Workers import models lazily in `_ensure_model()` to avoid blocking process spawn. This means workers start quickly and only load models (which can be heavy) when the first packet arrives.

### 9.6 Shadow Mode for Gradual Rollout
New deployments start in shadow mode to validate model behavior against real data before enabling live alerts. This prevents false alarm fatigue during initial tuning and provides structured ground truth for model improvement.

### 9.7 SQLite for Local Persistence
SQLite with WAL mode provides crash recovery without external database dependencies. Write-behind batching avoids blocking the inference pipeline. InfluxDB remains optional for deployments that need long-term time-series storage.

### 9.8 Adaptive Thresholds
Static thresholds work poorly across different environments (apartment layouts, furniture, number of occupants). Adaptive thresholds observe per-zone metric distributions and self-tune within safety floors, reducing false positives without sacrificing detection sensitivity.

---

## 10. Directory Structure

```
eldercare/
├── firmware/esp32_csi/          # ESP32-S3 firmware spec + CSI simulator
├── ingestion/                   # MQTT ingestion (receiver + ring buffers)
│   └── receiver.py              # CSIIngestion, CSIRingBuffer, IngestionStats
├── pipeline/                    # Signal preprocessing + inference engine
│   ├── preprocessor.py          # Hampel + Butterworth + phase + z-score
│   ├── inference_engine.py      # 12-worker multiprocessing orchestrator
│   ├── data_store.py            # Thread-safe inference data store
│   ├── persistence.py           # SQLite write-behind persistence (WAL mode)
│   ├── influx_writer.py         # InfluxDB batch writer with graceful degradation
│   ├── csi_quality.py           # CSI signal quality validation (SNR, packet loss)
│   ├── model_registry.py        # Config-driven per-zone model selection
│   ├── correlation.py           # End-to-end correlation tracking + latency stats
│   ├── adaptive_thresholds.py   # Self-tuning detection thresholds
│   ├── watchdog.py              # Worker health monitor + auto-restart
│   ├── shadow_mode.py           # Shadow mode gate + FP/FN tracking
│   ├── record_replay.py         # CSI recorder, replayer, and labeler
│   ├── homeassistant.py         # Home Assistant MQTT auto-discovery
│   ├── quantization.py          # INT8 post-training model quantization
│   ├── telemetry.py             # Local metrics collection (Prometheus format)
│   └── evaluate.py              # Model evaluation module
├── models/
│   ├── fall_detection/          # CSI-FallNet + TemperatureScaling + TwoStageConfirmer
│   │   └── model.py             # FallDetector, ConfidenceSmoother
│   ├── vital_signs/             # VitalsAdapter (wifi_densepose wrapper) + Python fallback
│   │   ├── estimator.py         # VitalsAdapter
│   │   └── python_fallback.py   # scipy FFT-based fallback extractors
│   ├── sleep/                   # SleepLSTM + SleepScorer
│   │   └── model.py             # SleepLSTM (5 features, FocalLoss)
│   ├── activity/                # ActivityDetector + PostFallInactivityChecker
│   │   └── detector.py          # Day/night-aware activity detection
│   └── calibration.py           # TemperatureScaling confidence calibration
├── alerts/                      # AlertManager + DailySummary
│   ├── alert_manager.py         # Telegram dispatch, 3-level cooldown
│   ├── daily_summary.py         # Vietnamese daily summary generator
│   └── i18n.py                  # Internationalization strings
├── dashboard/
│   ├── backend/                 # FastAPI API server
│   │   └── main.py              # REST + SSE endpoints
│   └── frontend/                # React frontend
│       # Zone cards, vitals graphs, sleep charts, alert log, health panel
├── training/
│   ├── dataset_mappers/         # ElderAL-CSI + CSI-Bench mappers
│   ├── dataset.py               # PyTorch Dataset classes
│   ├── annotator.py             # Ground truth annotation tool
│   ├── train_fall_detection.py  # Fall detection training
│   ├── train_sleep.py           # SleepLSTM training (FocalLoss, balanced)
│   └── mock_data.py             # Mock dataset generator
├── configs/                     # YAML configs (zones, thresholds, alerts, models)
├── docker/                      # Docker Compose + Dockerfile
├── data/                        # Gitignored -- datasets, logs, checkpoints, SQLite DB
├── tests/                       # Unit + integration tests
├── docs/                        # Architecture, installation, user guide
├── CLAUDE.md                    # AI assistant context
├── PROJECT-DETAIL.md            # Full project specification
└── README.md                    # Project overview
```

---

## 11. Upgrade History

### v0.2 (2026-05-25) -- 19 New Modules

| # | Module | File | Purpose |
|---|---|---|---|
| 1 | CSIQualityChecker | `pipeline/csi_quality.py` | SNR, packet loss, null subcarrier detection |
| 2 | WorkerWatchdog | `pipeline/watchdog.py` | Heartbeat checks, auto-restart, system health |
| 3 | SQLitePersistence | `pipeline/persistence.py` | Write-behind SQLite storage, WAL mode, crash recovery |
| 4 | InfluxWriter | `pipeline/influx_writer.py` | Batch writes to InfluxDB, graceful degradation |
| 5 | ModelRegistry | `pipeline/model_registry.py` | Config-driven per-zone model selection |
| 6 | CorrelationTracker | `pipeline/correlation.py` | End-to-end pipeline tracing, latency stats |
| 7 | AdaptiveThreshold | `pipeline/adaptive_thresholds.py` | Self-tuning detection from observed data |
| 8 | ShadowMode | `pipeline/shadow_mode.py` | Gradual rollout, alert suppression, FP/FN tracking |
| 9 | CSIRecorder | `pipeline/record_replay.py` | Record live CSI streams to JSONL |
| 10 | CSIReplayer | `pipeline/record_replay.py` | Replay recorded sessions for testing |
| 11 | CSILabeler | `pipeline/record_replay.py` | Annotate ground truth on recorded sessions |
| 12 | HomeAssistantDiscovery | `pipeline/homeassistant.py` | MQTT auto-discovery for HA entities |
| 13 | ModelQuantizer | `pipeline/quantization.py` | INT8 post-training static quantization |
| 14 | TelemetryCollector | `pipeline/telemetry.py` | Prometheus-format local metrics |
| 15 | TemperatureScaling | `models/calibration.py` | Confidence calibration for fall detector |
| 16 | ConfidenceSmoother | `models/fall_detection/model.py` | Window-3 sliding average on confidence |
| 17 | PythonFallbackVitals | `models/vital_signs/python_fallback.py` | scipy FFT fallback when Rust unavailable |
| 18 | FocalLoss | `models/sleep/model.py` | Class-balanced loss for sleep staging |
| 19 | SignalQualityFeature | `models/sleep/model.py` | 5th input feature from CSIQualityChecker |

### v0.1 (2025-06-01) -- Initial Release

Core pipeline: MQTT ingestion, scipy preprocessing, 12-worker multiprocessing inference engine (fall, vitals, sleep, activity), AlertManager with Telegram, FastAPI dashboard, ElderAL-CSI and CSI-Bench dataset mappers, training scripts.
