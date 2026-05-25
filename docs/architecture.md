# ElderCare — System Architecture

Design document covering system architecture, data flow, component interactions, and design decisions.

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                     ESP32-S3 NODES                     │
│  (up to 3 zones) CSI capture @ 50Hz, MQTT publish     │
└───────┬──────────────────────┬────────────────────────┘
        │ MQTT                 │ MQTT
        │ eldercare/csi/       │ eldercare/csi/
        │ zone_bedroom         │ zone_living
        ▼                      ▼
┌──────────────────────────────────────────────────────┐
│              MOSQUITTO MQTT BROKER                     │
│  (topic routing, QoS 1, persistent sessions)          │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│                INGESTION LAYER                         │
│  ingestion/receiver.py                                │
│  - Per-zone MQTT subscription                        │
│  - JSON + binary packet parsing                      │
│  - Ring buffers (5s @ 50Hz = 250 frames)             │
│  - CSIRingBuffer + IngestionStats per zone            │
└───────────────────────┬──────────────────────────────┘
                        │ per-zone CSI packets
┌───────────────────────▼──────────────────────────────┐
│              PREPROCESSING PIPELINE                    │
│  pipeline/preprocessor.py                             │
│  - Hampel filter (outlier removal)                    │
│  - Butterworth bandpass (4th order)                   │
│  - Phase sanitization (unwrap + linear detrend)       │
│  - Per-subcarrier z-score normalization (EMA)         │
│  - Edge subcarrier removal (±2)                        │
└───────────────────────┬──────────────────────────────┘
                        │ preprocessed (amplitude, phase)
┌───────────────────────▼──────────────────────────────┐
│              INFERENCE ENGINE                          │
│  pipeline/inference_engine.py (multiprocessing)       │
│                                                        │
│  Per zone (x3):                                        │
│  ┌──────────────────────────────────────────────────┐│
│  │ FallDetectionWorker                               ││
│  │  CSI-FallNet (1D-CNN → BiLSTM → Attention)       ││
│  │  + TwoStageConfirmer (0.85 conf + 3s inactivity)  ││
│  │  -> FallConfirmationEvent to shared queue         ││
│  ├──────────────────────────────────────────────────┤│
│  │ VitalSignsWorker                                  ││
│  │  VitalsAdapter (wifi_densepose Rust PyO3)         ││
│  │  BreathingExtractor + HeartRateExtractor          ││
│  │  -> respiration_bpm, heart_rate_bpm               ││
│  ├──────────────────────────────────────────────────┤│
│  │ SleepWorker                                       ││
│  │  SleepFeatureExtractor -> SleepLSTM              ││
│  │  -> sleep_stage, SleepScorer (0-100)              ││
│  ├──────────────────────────────────────────────────┤│
│  │ ActivityWorker                                    ││
│  │  ActivityDetector (rule-based, day/night)         ││
│  │  + PostFallInactivityChecker (30s recovery)      ││
│  │  -> active/still/inactivity, alert level          ││
│  └──────────────────────────────────────────────────┘│
│                                                        │
│  12 total workers (3 zones x 4 models)                │
└───────────────────────┬──────────────────────────────┘
                        │ output queue (InferenceResult)
┌───────────────────────▼──────────────────────────────┐
│           _RESULTORCHESTRATOR (thread)                 │
│  Consumes InferenceResult from output queue            │
│  - Updates InferenceDataStore                         │
│  - Dispatches alerts via AlertManager                 │
│  - Checks daily summary schedule (7:00 AM)            │
└─────┬─────────────────────────────┬──────────────────┘
      │                             │
┌─────▼──────────┐        ┌────────▼────────────────────┐
│ AlertManager   │        │ InferenceDataStore            │
│ alerts/        │        │ pipeline/data_store.py        │
│                │        │                               │
│ - Telegram     │        │ - ZoneStatus (per zone)       │
│   dispatch     │        │ - VitalsHistory (3600 pts)    │
│ - 3-level      │        │ - Alerts (500 recent)         │
│   cooldown     │        │ - SleepRecords (90 nights)    │
│ - VN messages  │        │ - Thread-safe (lock)          │
│ - File log     │        └────────────┬──────────────────┘
│ - Daily        │                     │
│   summary      │        ┌────────────▼──────────────────┐
└────────────────┘        │ Dashboard (FastAPI)             │
                          │ dashboard/backend/main.py       │
                          │                                 │
                          │ GET /api/zones                  │
                          │ GET /api/vitals?zone_id=...     │
                          │ GET /api/alerts?limit=50        │
                          │ GET /api/sleep?zone_id=...      │
                          │ GET /api/daily-summary          │
                          │ GET /api/health                 │
                          └────────────┬────────────────────┘
                                       │ HTTP
                          ┌────────────▼────────────────────┐
                          │ React Frontend                   │
                          │ dashboard/frontend/              │
                          │ - Zone cards                    │
                          │ - Vitals history chart           │
                          │ - Alert log                     │
                          │ - Sleep score trend              │
                          │ - 16px minimum font              │
                          └─────────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 CSI Packet Lifecycle

```
ESP32-S3 firmware (esp-idf CSI API)
  │  Reads CSI at 50 Hz (HT20, 52 subcarriers)
  │  Packs: {timestamp, sequence, csi_amplitude[52], csi_phase[52], rssi}
  ▼
MQTT publish to eldercare/csi/{zone_id}
  │  JSON or binary struct encoding
  ▼
Ingestion (receiver.py)
  │  CSIIngestion._on_message → _parse_packet → _validate_packet
  │  Pushes to CSIRingBuffer (250 frames, 5s window)
  │  Calls registered callbacks
  ▼
Preprocessor (preprocessor.py)
  │  Preprocessor.process(amplitude[100,52], phase[100,52])
  │  Hampel → Butterworth → Phase sanitize → Z-score → Edge removal
  │  Returns (processed_amp[100,48], processed_phase[100,48])
  ▼
InferenceEngine.feed_packet()
  │  Fans out one copy per worker queue (4 queues per zone)
  ▼
Workers process independently, post InferenceResult to output queue
  │
  ▼
_ResultOrchestrator thread
  │  Updates data store
  │  Dispatches alerts
  │  Checks daily summary schedule
```

### 2.2 Fall Detection Flow (Two-Stage)

```
Stage 1: Model inference
  CSI amplitude (T=100, C=52) → FallDetector
  → confidence ∈ [0,1]

Stage 2: Confirmation
  IF confidence >= 0.85:
    → _pending_confirmation = True
    → Wait for next 3 seconds of CSI data
    → Check CSI variance < inactivity_threshold (0.15)
    → YES: confirm fall → post FallConfirmationEvent
    → NO: dismiss as false positive

FallConfirmationEvent
  → ActivityWorker.post_fall_checker.on_fall_detected()
  → Checks 30s recovery window
  → No movement → "EMERGENCY"
  → Movement detected → reset
```

### 2.3 Vital Signs Flow

```
Per subcarrier amplitude residuals → wifi_densepose Rust extractors

BreathingExtractor (0.1-0.5 Hz bandpass, zero-crossing)
  → VitalEstimate {value_bpm, confidence, status}

HeartRateExtractor (0.8-2.0 Hz bandpass, autocorrelation)
  → VitalEstimate {value_bpm, confidence, status}

VitalsAdapter.feed_frame() per frame
  → Respiration/HR updated every 5 seconds
  → Stored in InferenceDataStore.vitals_history
```

### 2.4 Sleep Monitoring Flow

```
CSI data accumulates in SleepWorker buffer (60 * 50 = 3000 frames per minute)

SleepFeatureExtractor.extract(amplitude[N, C], respiration_bpm)
  → 4 features: [mean_respiration, respiration_std, movement_index, burst_count]

SleepLSTM(features[1,4])
  → probs: [p_awake, p_light, p_deep]
  → argmax → sleep stage

Nightly accumulation:
  → Epochs collected in _night_epochs (max 480 = 8 hours)
  → When >= 240 epochs (4 hours):
    SleepScorer.compute_score(stage_probs, session_hours)
    → 0-100 score from: deep% (40%) + efficiency% (40%) + weighted_quality (20%)
```

### 2.5 Alert Dispatch Flow

```
_ResultOrchestrator._handle_result(InferenceResult)

fall_detection result:
  if fall_detected:
    → AlertManager.send_emergency("Phát hiện té ngã!")
    → data_store.update_fall(zone, True, confidence)
    → data_store.add_alert(EMERGENCY)

vital_signs result:
  → data_store.update_vitals(zone, bpm, conf, hr, hrconf)

sleep result:
  → data_store.update_sleep(zone, stage, score)

activity result:
  → data_store.update_activity(zone, state, alert)
  if alert == "WARNING":
    → AlertManager.send_warning("Không hoạt động kéo dài")
  if post_fall_alert == "EMERGENCY":
    → AlertManager.send_emergency("KHẨN CẤP: Không cử động sau té ngã!")

AlertManager.send_alert():
  1. Cooldown check (per level + zone)
  2. Telegram dispatch (if configured)
  3. Log to data/alerts.log
  4. Add to data_store._alerts
```

---

## 3. Component Design

### 3.1 InferenceEngine

| Aspect | Detail |
|---|---|
| Pattern | Fan-out with per-worker input queues |
| Process model | `multiprocessing.Process` (12 workers) |
| Inter-worker comm | `mp.Queue` (input + output queues) |
| Fall→Activity bridge | Per-zone `fall_event_queue` (FallWorker → ActivityWorker) |
| Result consumption | `_ResultOrchestrator` thread reads output queue |
| GIL handling | Each worker is a separate process — no GIL contention |

### 3.2 InferenceDataStore

| Aspect | Detail |
|---|---|
| Thread safety | `threading.Lock` on all mutations |
| Vitals history | `deque(maxlen=3600)` per zone (~1 hour at 1/sec) |
| Alerts | List, capped at 500 |
| Sleep records | List, capped at 90 nights |
| Zone status | Dict of `ZoneStatus` dataclasses |

### 3.3 AlertManager

| Aspect | Detail |
|---|---|
| Cooldown key | `{level}_{zone_id}` |
| Cooldown periods | EMERGENCY: 60s, WARNING: 300s, INFO: 3600s |
| Telegram | Optional — skips silently if not configured |
| Log persistence | `data/alerts.log` (pipe-delimited) |
| VN formatting | `AlertMessage.format_vn()` with emoji + datetime |

### 3.4 Daily Summary Generator

| Aspect | Detail |
|---|---|
| Sections | Night summary, current status, 24h alerts, daily advice |
| Data source | `InferenceDataStore` (real) or dummy mode |
| Dummy mode | Plausible mock data for testing/demo |
| Language | Vietnamese |
| Schedule | Checked every event loop cycle, fires at `alerts.yaml: daily_summary.time_local` (default 07:00) |

---

## 4. Model Architecture Details

### 4.1 CSI-FallNet (FallDetector)

```
Input: (B, T=100, C=52)

Conv1D(52→64, k5) + BN + ReLU
Conv1D(64→128, k3) + BN + ReLU + MaxPool(2) → (B, 50, 128)
Conv1D(128→128, k3) + BN + ReLU + MaxPool(2) → (B, 25, 128)

BiLSTM(128→256, bidirectional) → (B, 25, 512)
AttentionPooling → (B, 512)

FC(512→256) + Dropout(0.5) + ReLU
FC(256→64) + ReLU
FC(64→2) + Softmax → {non-fall, fall}

Parameters: 2,607,235 (2.6M)
```

### 4.2 SleepLSTM

```
Input: (B, N_epochs, 4) — 4 features per 1-minute epoch

LSTM(4→64, 2 layers, dropout=0.3) → (B, N_epochs, 64)
FC(64→32) + ReLU + Dropout(0.3)
FC(32→3) + Softmax → {awake, light, deep}

Parameters: 53,379
```

### 4.3 VitalsAdapter

```
Wraps wifi_densepose Rust-native extractors (PyO3):

BreathingExtractor.esp32_default()  — 56 subcarriers, 100 Hz, 30s window
HeartRateExtractor.esp32_default()  — 56 subcarriers, 100 Hz, 15s window

Stateful: feed one frame at a time, GIL released during DSP loop.
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
    → Select TX-RX pair (e.g. tx0_rx0)
    → Downsample 512 → 52 subcarriers
    → Extract label from path (action2_fall_new/ → 1)

  sliding_windows()
    → window_size=100, stride=50
    → Output: (N, 100, 52) with labels

  load_elderal_directory()
    → Bulk CSV loading
    → Shuffle
    → 80/20 train/val split
```

### 5.2 CSI-Bench Mapper

```
Input: .npz (keys: data, labels) — shape (N, 100, 52) int64 labels

Fallback: .mat adapter
  preprocess_csibench_mat()
    → scipy.io.loadmat
    → Print available keys for manual configuration
    → amplitude extraction → subcarrier selection → sliding windows
    → Output shape: (N, 100, 52)
```

---

## 6. Configuration Files

| File | Purpose |
|---|---|
| `configs/zones.yaml` | Zone definitions (ID, name, MAC, MQTT topic) |
| `configs/thresholds.yaml` | Detection thresholds (inactivity, fall, vitals, sleep, activity) |
| `configs/alerts.yaml` | Alert cooldowns, Telegram settings, daily summary time |
| `configs/models.yaml` | Checkpoint paths, device, input shapes |
| `.env` | Secrets (Telegram token, chat IDs, InfluxDB) |

---

## 7. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| CSI Capture | ESP32-S3, esp-idf CSI API | Low cost, documented, 50 Hz capture |
| Transport | MQTT (Mosquitto) | Per-zone topic routing, persistent sessions |
| Signal Processing | Python/scipy | Faithful reimplementation of RuView's Rust chain |
| Vitals Extraction | wifi_densepose (Rust PyO3) | Battle-tested RuView algorithms, GIL-released |
| Deep Learning | PyTorch (CNN, BiLSTM, LSTM) | Standard ML framework, INT8 quantization |
| API Server | FastAPI + uvicorn | Async, auto-docs, production-ready |
| Frontend | React | Responsive, caregiver-friendly |
| Alerting | python-telegram-bot | Free, reliable, Vietnamese support |
| Container | Docker + docker-compose | Reproducible deployment |
| Time-series | InfluxDB (optional) | Metrics storage for historical analysis |

---

## 8. Design Decisions

### 8.1 MQTT over UDP
RuView's Rust server uses UDP for CSI transport. ElderCare uses MQTT because:
- Topic-based routing gives natural per-zone isolation (`eldercare/csi/{zone_id}`)
- QoS guarantees prevent packet loss
- Mosquitto runs on the same server — no extra hardware
- ESP32 has native MQTT support in esp-idf

### 8.2 Custom Preprocessor (not RuView import)
RuView's signal processing chain lives in the Rust server binary, not the Python package. Reimplementing in scipy:
- No Rust toolchain dependency for the Python inference server
- Easier to modify and debug (Python vs Rust FFI)
- Functionally equivalent — same Hampel, Butterworth, phase sanitization algorithms

### 8.3 Stateful Vitals Extraction
The `wifi_densepose` `BreathingExtractor` and `HeartRateExtractor` are stateful (internal circular buffers). The `VitalsAdapter` feeds one frame at a time — no batching needed. This allows real-time streaming inference.

### 8.4 Multiprocessing over Threading
Workers run as `multiprocessing.Process` (not threads) because:
- PyTorch inference benefits from separate processes (no GIL)
- A crashed worker doesn't bring down the orchestrator
- Per-worker queues avoid shared memory contention

### 8.5 Lazy Model Initialization
Workers import models lazily in `_ensure_model()` to avoid blocking process spawn. This means workers start quickly and only load models (which can be heavy) when the first packet arrives.

---

## 9. Directory Structure

```
eldercare/
├── firmware/esp32_csi/          # ESP32-S3 firmware spec + CSI simulator
├── ingestion/                   # MQTT ingestion (receiver + ring buffers)
├── pipeline/                    # Signal preprocessing + inference engine
│   ├── preprocessor.py          # Hampel + Butterworth + phase + z-score
│   ├── inference_engine.py      # 12-worker multiprocessing orchestrator
│   ├── data_store.py            # Thread-safe inference data store
│   └── evaluate.py              # Model evaluation module
├── models/
│   ├── fall_detection/          # CSI-FallNet + TwoStageConfirmer
│   ├── vital_signs/             # VitalsAdapter (wifi_densepose wrapper)
│   ├── sleep/                   # SleepLSTM + SleepScorer
│   └── activity/                # ActivityDetector + PostFallInactivityChecker
├── alerts/                      # AlertManager + DailySummary
├── dashboard/
│   ├── backend/                 # FastAPI API server
│   └── frontend/                # React frontend
├── training/
│   ├── dataset_mappers/         # ElderAL-CSI + CSI-Bench mappers
│   ├── train_fall_detection.py  # Fall detection training
│   ├── train_sleep.py           # SleepLSTM training
│   └── mock_data.py             # Mock dataset generator
├── configs/                     # YAML configs (zones, thresholds, alerts, models)
├── docker/                      # Docker Compose + Dockerfile
├── data/                        # Gitignored — datasets, logs, checkpoints
├── tests/                       # 51 unit + integration tests
├── docs/                        # Architecture, installation, user guide
├── CLAUDE.md                    # AI assistant context
├── PROJECT-DETAIL.md            # Full project specification
└── README.md                    # Project overview
```
