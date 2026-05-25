# CLAUDE.md -- ElderCare Project

> Context and guidance for AI assistants working on the ElderCare codebase. Read before making any changes.

---

## Project Identity

**ElderCare** is a privacy-preserving, WiFi-based elderly monitoring system. Forked from [RuView](https://github.com/ruvnet/RuView) and heavily customized for eldercare use cases in Vietnamese households.

- **Primary language:** Python (backend, ML pipeline)
- **Hardware target:** ESP32-S3 nodes + Raspberry Pi 5 / Mini PC server
- **Deployment:** 100% local, no cloud dependency
- **Remote:** `origin` = `dungnotnull/WiFi-Sensing-Based-Elderly-Care-Monitoring-System`

---

## Repository Structure

```
eldercare/
├── firmware/esp32_csi/     # ESP32-S3 firmware spec + CSI simulator
├── ingestion/              # MQTT ingestion wrapper (zone routing for ElderCare topics)
├── models/                 # ElderCare-specific models
│   ├── fall_detection/     # CSI-FallNet, TwoStageConfirmer, TemperatureScaling, ConfidenceSmoother
│   ├── vital_signs/        # Adapter over RuView vitals engine + Python fallback
│   │   └── python_fallback.py  # Pure-Python vital signs (scipy) when Rust unavailable
│   ├── sleep/              # SleepLSTM + SleepScorer (ElderCare-specific)
│   └── activity/           # Day/night-aware inactivity detection
├── pipeline/               # RuView preprocessing + ElderCare inference engine
│   ├── preprocessor.py     # Hampel, Butterworth, phase sanitization, z-score
│   ├── inference_engine.py # Multiprocessing inference (one worker per model)
│   ├── data_store.py       # SQLite persistence layer for events and state
│   ├── evaluate.py         # Batch evaluation against labeled datasets
│   ├── persistence.py      # State snapshot/restore for graceful restart
│   ├── watchdog.py         # WorkerWatchdog -- liveness monitoring for inference workers
│   ├── csi_quality.py      # CSI signal quality metrics (SNR, packet loss, subcarrier health)
│   ├── influx_writer.py    # Async InfluxDB time-series writer for vitals and events
│   ├── model_registry.py   # Model versioning, checkpoint discovery, load/swap
│   ├── correlation.py      # Correlation-ID propagation across pipeline stages
│   ├── adaptive_thresholds.py  # Runtime threshold tuning from recent event history
│   ├── record_replay.py    # CSIRecorder/CSIReplayer for capture and deterministic replay
│   ├── homeassistant.py    # Home Assistant MQTT discovery and state publishing
│   ├── quantization.py     # INT8/FP16 model quantization for edge deployment
│   ├── shadow_mode.py      # Shadow-mode gate -- log predictions without alerting
│   └── telemetry.py        # Performance and health metrics collection
├── alerts/                 # Alert manager (Telegram, Vietnamese, 3-level cooldown)
│   └── i18n.py             # YAML-based locale templates for VI/EN alert messages
├── dashboard/              # Caregiver web dashboard (FastAPI + React)
├── data/                   # Local datasets (gitignored)
│   ├── raw/                # CSI captures + ElderAL-CSI dataset
│   ├── processed/          # Preprocessed tensors/numpy arrays
│   ├── annotations/        # Manual labels for fine-tuning
│   ├── recordings/         # Recorded CSI sessions from CSIRecorder
│   ├── shadow_mode/        # Shadow-mode prediction logs for evaluation
│   └── telemetry/          # Telemetry snapshots and performance reports
├── training/               # Fine-tuning scripts (ElderAL-CSI + in-situ)
├── configs/                # YAML configs per zone/room
│   └── locales/            # Internationalization message templates
│       ├── vi.yaml         # Vietnamese alert and UI message templates
│       └── en.yaml         # English alert and UI message templates
├── docker/                 # Dockerfiles and docker-compose
├── tests/                  # Unit tests
├── docs/                   # Architecture diagrams, API docs
├── CLAUDE.md               # <-- You are here
└── PROJECT-DETAIL.md       # Full project specification
```

---

## RuView Dependencies -- What We Inherit vs. What We Build

ElderCare leverages RuView where it provides battle-tested infrastructure: **use wifi_densepose Rust extractors for vitals; build signal processing, ingestion, and models as custom Python.**

### From RuView (`wifi_densepose` Python package, v2.0.0a1+)

| RuView Component | How Used |
|---|---|
| **BreathingExtractor** | Rust-native 0.1-0.5 Hz bandpass + zero-crossing respiration detector via PyO3 |
| **HeartRateExtractor** | Rust-native 0.8-2.0 Hz bandpass + autocorrelation heart rate detector via PyO3 |
| ESP32-S3 CSI Firmware | Raw CSI capture at 50 Hz, HT20 mode, 52 subcarriers |

### ElderCare Custom (Python)

| ElderCare Module | Why Custom / Not in RuView |
|---|---|
| **Preprocessor** (`pipeline/preprocessor.py`) | Hampel, Butterworth, phase sanitization, z-score. Faithful scipy reimplementation of RuView's Rust signal chain |
| **MQTT Ingestion** (`ingestion/receiver.py`) | Per-zone MQTT routing. RuView uses UDP -- MQTT chosen for multi-zone topic separation |
| **VitalsAdapter** (`models/vital_signs/estimator.py`) | Stateful Python wrapper feeding per-frame amplitude residuals to wifi_densepose Rust extractors |
| **Python Vitals Fallback** (`models/vital_signs/python_fallback.py`) | Pure-scipy vital signs estimator for environments where wifi_densepose Rust wheel is unavailable |
| **CSI-FallNet** (`models/fall_detection/`) | 1D-CNN to BiLSTM, fine-tunable on ElderAL-CSI |
| **TwoStageConfirmer** | 0.85 confidence + 3-second inactivity re-check |
| **TemperatureScaling** (`models/fall_detection/calibration.py`) | Post-hoc confidence calibration for well-calibrated fall probabilities |
| **ConfidenceSmoother** | Exponential moving average on raw confidence to reduce transient spikes |
| **SleepLSTM + SleepScorer** | No RuView equivalent -- sleep staging and quality score |
| **Alert Manager** (`alerts/`) | Vietnamese Telegram, 3-level cooldown |
| **Activity Detector** | Day/night context, post-fall recovery monitoring |
| **Caregiver Dashboard** (`dashboard/`) | FastAPI + React, 16px min font, VN labels |
| **DataStore** (`pipeline/data_store.py`) | SQLite persistence for events, model state, and deduplication across restarts |
| **Persistence** (`pipeline/persistence.py`) | State snapshot/restore for inference workers -- enables graceful restart without data loss |
| **WorkerWatchdog** (`pipeline/watchdog.py`) | Liveness monitoring for inference worker processes with automatic restart on hang |
| **CSI Quality Checker** (`pipeline/csi_quality.py`) | Signal quality metrics (SNR, packet loss, subcarrier health) -- rejects degraded frames |
| **InfluxDB Writer** (`pipeline/influx_writer.py`) | Async time-series writer for vitals and events to InfluxDB |
| **Model Registry** (`pipeline/model_registry.py`) | Model versioning, checkpoint discovery, hot-swap without restart |
| **Correlation ID** (`pipeline/correlation.py`) | UUID-based correlation tracking across all pipeline stages for traceability |
| **Adaptive Thresholds** (`pipeline/adaptive_thresholds.py`) | Runtime threshold tuning from recent event history to reduce false positives over time |
| **CSIRecorder/Replayer** (`pipeline/record_replay.py`) | Record live CSI streams and replay them deterministically for testing and evaluation |
| **Home Assistant** (`pipeline/homeassistant.py`) | MQTT discovery and state publishing for Home Assistant integration |
| **Quantization** (`pipeline/quantization.py`) | INT8/FP16 model quantization pipeline for Raspberry Pi 5 edge deployment |
| **Shadow Mode** (`pipeline/shadow_mode.py`) | Log predictions alongside real alerts without firing them -- safe first-deployment evaluation |
| **Telemetry** (`pipeline/telemetry.py`) | Performance and health metrics collection (latency, throughput, memory) |
| **i18n** (`alerts/i18n.py`) | YAML-based locale templates for Vietnamese and English alert messages |
| **Evaluate** (`pipeline/evaluate.py`) | Batch evaluation runner against labeled datasets with per-model metric reporting |

**Rule of thumb:** Vitals extraction goes through `wifi_densepose` (Rust) with a Python fallback. Signal preprocessing and ingestion are faithful Python reimplementations -- the former matching RuView's Rust signal chain, the latter using MQTT (an architectural choice over RuView's UDP). Everything elderly-care-specific (models, alerts, dashboard, persistence, monitoring, quantization, shadow mode, telemetry) is custom.

---

## Core Technologies

| Layer | Technology |
|---|---|
| CSI Capture | ESP32-S3 (esp-idf CSI API) |
| Transport | MQTT (Mosquitto broker) |
| Signal Processing | Custom Python (scipy reimplementation of RuView signal chain) |
| Deep Learning | PyTorch -- CNN, BiLSTM (FallNet), LSTM (Sleep) |
| Inference Server | FastAPI (async) + multiprocessing queue |
| Dashboard | React (caregiver UI, SSE for real-time updates) |
| Alerting | python-telegram-bot, Vietnamese localization |
| Persistence | SQLite (events/state) + InfluxDB (time-series vitals) |
| Monitoring | psutil + WorkerWatchdog (liveness and health) |
| i18n | YAML-based locale templates (VI/EN) |
| Edge Optimization | PyTorch quantization (INT8/FP16) for RPi5 deployment |
| Containerization | Docker + docker-compose |

---

## ML Models -- Key Details

### 1. Fall Detection (`models/fall_detection/`)
- **Architecture:** 1D-CNN to BiLSTM to FC classifier (CSI-FallNet)
- **Input:** CSI amplitude matrix, shape `(T=100, N_subcarriers=52)` -- 2-second sliding window at 50 Hz
- **Output:** Binary `{fall, non-fall}`, with calibrated confidence score
- **Calibration:** TemperatureScaling post-hoc calibration ensures confidence scores reflect true probabilities
- **Smoothing:** ConfidenceSmoother applies exponential moving average to suppress transient spikes
- **Target accuracy:** > 85% F1 on held-out ElderAL-CSI test set
- **False positive strategy:** Two-stage confirmation -- initial trigger + 3-second inactivity re-check
- **Training pipeline:** CSI-Bench pre-training to ElderAL-CSI fine-tuning to in-situ data

### 2. Vital Signs (`models/vital_signs/`)
- **Respiration:** `wifi_densepose.BreathingExtractor` -- Rust-native 0.1-0.5 Hz bandpass + zero-crossing
- **Heart Rate:** `wifi_densepose.HeartRateExtractor` -- Rust-native 0.8-2.0 Hz bandpass + autocorrelation (experimental)
- **Python fallback:** `python_fallback.py` provides pure-scipy vital signs estimation when the Rust wheel is unavailable (e.g., non-Linux hosts, CI)
- **Output:** BPM estimates updated every 5 seconds

### 3. Sleep Quality (`models/sleep/`)
- **Architecture:** LSTM sequence classifier
- **Input:** 5 features per 1-minute epoch -- respiration rate, movement index, movement rate of change, heart rate, time-of-day encoding (n_features=5)
- **Output:** Sleep stage {awake, light, deep} + Sleep Score (0-100)
- **Loss function:** FocalLoss for class imbalance (deep sleep underrepresented in training data)
- **Training:** Balanced sampling across sleep stages to prevent majority-class bias
- **ElderCare-specific:** No RuView equivalent exists

### 4. Activity / Inactivity Detection (`models/activity/`)
- **Purpose:** Detect prolonged inactivity (> 2 hours) vs. normal stillness (sleeping)
- **Day/night awareness:** Suppress inactivity alerts during sleep window (default 10PM-6AM)
- **Post-fall checker:** 30-second recovery window triggers EMERGENCY if no movement
- **Adaptive thresholds:** Runtime tuning from recent event history to adapt to individual patterns

---

## Data Pipeline

```
ESP32-S3 (RuView firmware, CSI capture @ 50 Hz)
    |  MQTT -- topic: eldercare/csi/{zone_id}
    v
ingestion/receiver.py  --> MQTT client + ring buffer (5s)
    |
    v
pipeline/csi_quality.py --> Signal quality check (SNR, packet loss, subcarrier health)
    |  Rejects degraded frames before processing
    v
ElderCare Signal Processing (scipy, faithful to RuView Rust chain)
    |-- Hampel filter (outlier removal)
    |-- Butterworth bandpass
    |-- Phase sanitization (unwrapping + linear detrend)
    +-- Amplitude normalization (per-subcarrier z-score)
    |
    v
pipeline/correlation.py --> Assign correlation ID for end-to-end tracing
    |
    v
pipeline/inference_engine.py  (multiprocessing, one worker per model)
    |-- FallDetector (CSI-FallNet + TemperatureScaling + ConfidenceSmoother + TwoStageConfirmer)
    |-- VitalSignsEstimator (wifi_densepose Rust / Python fallback)
    |-- SleepMonitor (ElderCare SleepLSTM, n_features=5)
    +-- ActivityTracker (rule-based + adaptive thresholds + day/night context)
    |
    +--> pipeline/data_store.py --> SQLite (events, deduplication, state)
    +--> pipeline/influx_writer.py --> InfluxDB (time-series vitals and events)
    +--> pipeline/telemetry.py --> Performance metrics collection
    |
    v
pipeline/shadow_mode.py --> Gate: log predictions silently OR pass to alerts
    |  Shadow mode enabled: predictions logged to data/shadow_mode/ only
    |  Shadow mode disabled: predictions flow to alert manager
    v
alerts/alert_manager.py + alerts/i18n.py --> Telegram (VI/EN templates) / log
    |
    +--> pipeline/homeassistant.py --> Home Assistant MQTT state updates
    v
dashboard/ --> Browser UI (SSE for real-time event streaming)
```

---

## Alert Levels

| Level | Trigger | Action |
|---|---|---|
| **INFO** | Normal daily summary | Telegram message (scheduled, e.g., 7 AM) |
| **WARNING** | Inactivity > 2 hours, abnormal breathing | Telegram alert to caregiver |
| **EMERGENCY** | Fall detected + no recovery in 30s | Immediate Telegram + escalation |

Alert cooldown: minimum 5 minutes between same-level alerts (configurable in `configs/alerts.yaml`).

---

## Configuration

All runtime parameters live in `configs/`. Never hardcode thresholds in model files.

Key config files:
- `configs/zones.yaml` -- room/zone definitions, ESP32 MAC assignments
- `configs/thresholds.yaml` -- inactivity timeout, fall confirmation window, breathing rate bounds, adaptive threshold settings, shadow_mode config, telemetry settings
- `configs/alerts.yaml` -- Telegram bot token, chat IDs, cooldown periods
- `configs/models.yaml` -- model checkpoint paths, inference batch size, device (cpu/cuda)
- `configs/locales/vi.yaml` -- Vietnamese message templates for alerts and UI
- `configs/locales/en.yaml` -- English message templates for alerts and UI

---

## Development Rules

1. **Leverage RuView where it fits.** Vitals extraction uses `wifi_densepose` Rust PyO3 bindings (with Python fallback). Signal preprocessing and MQTT ingestion are faithful Python reimplementations optimized for ElderCare's multi-zone architecture.
2. **Never commit secrets.** Use `.env` files. Telegram tokens, API keys must never appear in git history.
3. **Preprocessing is RuView's domain.** Do not modify signal processing chain without updating both CLAUDE.md and PROJECT-DETAIL.md.
4. **Use type hints.** All Python functions must have type annotations.
5. **Log, don't print.** Use Python `logging` module. `print()` statements are not allowed.
6. **No raw data in git.** The `data/` directory is gitignored. Document dataset setup in docs.
7. **Docker first.** New services must have a corresponding `docker-compose.yml` entry.
8. **Keep sleep module.** SleepLSTM and SleepScorer are ElderCare-specific -- RuView has no equivalent.
9. **Shadow mode first deployment.** When deploying to a new site, always start with `shadow_mode.enabled: true` to validate predictions before enabling real alerts.
10. **Persistence for restarts.** All stateful pipeline components must support snapshot/restore via `persistence.py` to survive graceful restarts without data loss.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only model tests
pytest tests/models/ -v

# Run with coverage
pytest tests/ --cov=pipeline --cov=models --cov-report=html
```

Minimum coverage target: **70%** for `pipeline/` and `models/` modules.

---

## Common Commands

```bash
# Start full stack (local dev)
docker-compose -f docker/docker-compose.yml up --build

# Run inference pipeline standalone
python -m pipeline.inference_engine --config configs/zones.yaml

# Fine-tune fall detection model
python training/train_fall_detection.py \
  --dataset data/processed/csibench+elderal \
  --epochs 50 \
  --output models/fall_detection/checkpoints/

# Send a test Telegram alert
python -m alerts.alert_manager --test --level emergency

# Run dashboard
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000

# Record CSI data for offline testing
python -c "from pipeline.record_replay import CSIRecorder; CSIRecorder(output_dir='data/recordings/session_01', zone_id='living_room').start()"

# Replay recorded CSI data
python -c "from pipeline.record_replay import CSIReplayer; CSIReplayer(recording_dir='data/recordings/session_01', speed=1.0).start()"

# Quantize model for edge deployment (RPi5)
python -c "from pipeline.quantization import quantize_from_checkpoint; quantize_from_checkpoint('models/fall_detection/checkpoints/best.pt', output_path='models/fall_detection/checkpoints/best_int8.pt', dtype='int8')"

# Run in shadow mode (first deployment)
# Set shadow_mode.enabled: true in configs/thresholds.yaml
# Predictions logged to data/shadow_mode/ without triggering alerts
```

---

## Datasets

- **[CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)** -- Free, public, Kaggle. Large-scale CSI dataset for pre-training.
- **ElderAL-CSI** -- Local files (to be provided). Elderly-specific CSI dataset. Place in `data/raw/elderal-csi/`.
- **In-Situ Data** -- Collected on-site at deployment (3-5 hours per zone). Stored in `data/raw/`.
- **Recorded Sessions** -- CSI captures from CSIRecorder for deterministic replay and evaluation. Stored in `data/recordings/`.

---

## Known Limitations

- **Single-person assumption:** Current models assume a single occupant per zone. Multi-person CSI is unsupported in MVP.
- **Thick concrete walls:** CSI propagation degrades significantly through reinforced concrete.
- **ESP32-S3 CSI API:** Subcarrier count varies (52 for HT20, 114 for HT40). Firmware config must match model input shape.
- **Phase noise:** ESP32 phase data is noisy. Always apply RuView's phase sanitization before frequency-domain analysis.
- **Raspberry Pi 5 inference:** INT8 quantization pipeline now available (`pipeline/quantization.py`) to reduce model size and latency on edge hardware. Keep models under 5M params.
- **Real-world accuracy:** All reported metrics are on mock/synthetic data. In-situ evaluation is needed to validate real-world performance before clinical reliance.

---

## References

- Base repo: [ruvnet/RuView](https://github.com/ruvnet/RuView)
- [Kaggle -- CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)
- [ESP-IDF CSI API docs](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information)
