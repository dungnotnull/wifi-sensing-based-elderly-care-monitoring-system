# CLAUDE.md — ElderCare Project

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
│   ├── fall_detection/     # CSI-FallNet architecture + TwoStageConfirmer
│   ├── vital_signs/        # Adapter over RuView vitals engine
│   ├── sleep/              # SleepLSTM + SleepScorer (ElderCare-specific)
│   └── activity/           # Day/night-aware inactivity detection
├── pipeline/               # RuView preprocessing + ElderCare inference engine
├── alerts/                 # Alert manager (Telegram, Vietnamese, 3-level cooldown)
├── dashboard/              # Caregiver web dashboard (FastAPI + React)
├── data/                   # Local datasets (gitignored)
│   ├── raw/                # CSI captures + ElderAL-CSI dataset
│   ├── processed/          # Preprocessed tensors/numpy arrays
│   └── annotations/        # Manual labels for fine-tuning
├── training/               # Fine-tuning scripts (ElderAL-CSI + in-situ)
├── configs/                # YAML configs per zone/room
├── docker/                 # Dockerfiles and docker-compose
├── tests/                  # Unit tests
├── docs/                   # Architecture diagrams, API docs
├── CLAUDE.md               # ← You are here
└── PROJECT-DETAIL.md       # Full project specification
```

---

## RuView Dependencies — What We Inherit vs. What We Build

ElderCare leverages RuView where it provides battle-tested infrastructure: **use wifi_densepose Rust extractors for vitals; build signal processing, ingestion, and models as custom Python.**

### ✅ From RuView (`wifi_densepose` Python package, v2.0.0a1+)

| RuView Component | How Used |
|---|---|
| **BreathingExtractor** | Rust-native 0.1-0.5 Hz bandpass + zero-crossing respiration detector via PyO3 |
| **HeartRateExtractor** | Rust-native 0.8-2.0 Hz bandpass + autocorrelation heart rate detector via PyO3 |
| ESP32-S3 CSI Firmware | Raw CSI capture at 50 Hz, HT20 mode, 52 subcarriers |

### 🔧 ElderCare Custom (Python)

| ElderCare Module | Why Custom / Not in RuView |
|---|---|
| **Preprocessor** (`pipeline/preprocessor.py`) | Hampel, Butterworth, phase sanitization, z-score. Faithful scipy reimplementation of RuView's Rust signal chain |
| **MQTT Ingestion** (`ingestion/receiver.py`) | Per-zone MQTT routing. RuView uses UDP — MQTT chosen for multi-zone topic separation |
| **VitalsAdapter** (`models/vital_signs/estimator.py`) | Stateful Python wrapper feeding per-frame amplitude residuals to wifi_densepose Rust extractors |
| **CSI-FallNet** (`models/fall_detection/`) | 1D-CNN→BiLSTM, fine-tunable on ElderAL-CSI |
| **TwoStageConfirmer** | 0.85 confidence + 3-second inactivity re-check |
| **SleepLSTM + SleepScorer** | No RuView equivalent — sleep staging and quality score |
| **Alert Manager** (`alerts/`) | Vietnamese Telegram, 3-level cooldown |
| **Activity Detector** | Day/night context, post-fall recovery monitoring |
| **Caregiver Dashboard** (`dashboard/`) | FastAPI + React, 16px min font, VN labels |

**Rule of thumb:** Vitals extraction goes through `wifi_densepose` (Rust). Signal preprocessing and ingestion are faithful Python reimplementations — the former matching RuView's Rust signal chain, the latter using MQTT (an architectural choice over RuView's UDP). Everything elderly-care-specific (models, alerts, dashboard) is custom.

---

## Core Technologies

| Layer | Technology |
|---|---|
| CSI Capture | ESP32-S3 (esp-idf CSI API) |
| Transport | MQTT (Mosquitto broker) |
| Signal Processing | Custom Python (scipy reimplementation of RuView signal chain) |
| Deep Learning | PyTorch — CNN, BiLSTM (FallNet), LSTM (Sleep) |
| Inference Server | FastAPI (async) + multiprocessing queue |
| Dashboard | React (caregiver UI) |
| Alerting | python-telegram-bot, Vietnamese localization |
| Containerization | Docker + docker-compose |
| Local DB | SQLite (events/logs) + InfluxDB (time-series vitals) |

---

## ML Models — Key Details

### 1. Fall Detection (`models/fall_detection/`)
- **Architecture:** 1D-CNN → BiLSTM → FC classifier (CSI-FallNet)
- **Input:** CSI amplitude matrix, shape `(T=100, N_subcarriers=52)` — 2-second sliding window at 50 Hz
- **Output:** Binary `{fall, non-fall}`, with confidence score
- **Target accuracy:** > 85% F1 on held-out ElderAL-CSI test set
- **False positive strategy:** Two-stage confirmation — initial trigger + 3-second inactivity re-check
- **Training pipeline:** CSI-Bench pre-training → ElderAL-CSI fine-tuning → in-situ data

### 2. Vital Signs (`models/vital_signs/`)
- **Respiration:** `wifi_densepose.BreathingExtractor` — Rust-native 0.1-0.5 Hz bandpass + zero-crossing
- **Heart Rate:** `wifi_densepose.HeartRateExtractor` — Rust-native 0.8-2.0 Hz bandpass + autocorrelation (experimental)
- **Output:** BPM estimates updated every 5 seconds

### 3. Sleep Quality (`models/sleep/`)
- **Architecture:** LSTM sequence classifier
- **Input:** Respiration rate + movement index over 1-minute epochs
- **Output:** Sleep stage {awake, light, deep} + Sleep Score (0–100)
- **ElderCare-specific:** No RuView equivalent exists

### 4. Activity / Inactivity Detection (`models/activity/`)
- **Purpose:** Detect prolonged inactivity (> 2 hours) vs. normal stillness (sleeping)
- **Day/night awareness:** Suppress inactivity alerts during sleep window (default 10PM–6AM)
- **Post-fall checker:** 30-second recovery window triggers EMERGENCY if no movement

---

## Data Pipeline

```
ESP32-S3 (RuView firmware, CSI capture @ 50 Hz)
    │  MQTT — topic: eldercare/csi/{zone_id}
    ▼
ingestion/receiver.py  ──► MQTT client + ring buffer (5s)
    │
    ▼
ElderCare Signal Processing (scipy, faithful to RuView Rust chain)
    ├── Hampel filter (outlier removal)
    ├── Butterworth bandpass
    ├── Phase sanitization (unwrapping + linear detrend)
    └── Amplitude normalization (per-subcarrier z-score)
    │
    ▼
pipeline/inference_engine.py  (multiprocessing, one process per model)
    ├── FallDetector (CSI-FallNet + TwoStageConfirmer)
    ├── VitalSignsEstimator (wifi_densepose Rust extractors)
    ├── SleepMonitor (ElderCare SleepLSTM)
    └── ActivityTracker (rule-based + day/night context)
    │
    ▼
alerts/alert_manager.py  ──► Telegram / log / InfluxDB write
    │
    ▼
dashboard/  ──► Browser UI (polling or WebSocket)
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
- `configs/zones.yaml` — room/zone definitions, ESP32 MAC assignments
- `configs/thresholds.yaml` — inactivity timeout, fall confirmation window, breathing rate bounds
- `configs/alerts.yaml` — Telegram bot token, chat IDs, cooldown periods
- `configs/models.yaml` — model checkpoint paths, inference batch size, device (cpu/cuda)

---

## Development Rules

1. **Leverage RuView where it fits.** Vitals extraction uses `wifi_densepose` Rust PyO3 bindings. Signal preprocessing and MQTT ingestion are faithful Python reimplementations optimized for ElderCare's multi-zone architecture.
2. **Never commit secrets.** Use `.env` files. Telegram tokens, API keys must never appear in git history.
3. **Preprocessing is RuView's domain.** Do not modify signal processing chain without updating both CLAUDE.md and PROJECT-DETAIL.md.
4. **Use type hints.** All Python functions must have type annotations.
5. **Log, don't print.** Use Python `logging` module. `print()` statements are not allowed.
6. **No raw data in git.** The `data/` directory is gitignored. Document dataset setup in docs.
7. **Docker first.** New services must have a corresponding `docker-compose.yml` entry.
8. **Keep sleep module.** SleepLSTM and SleepScorer are ElderCare-specific — RuView has no equivalent.

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
```

---

## Datasets

- **[CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)** — Free, public, Kaggle. Large-scale CSI dataset for pre-training.
- **ElderAL-CSI** — Local files (to be provided). Elderly-specific CSI dataset. Place in `data/raw/elderal-csi/`.
- **In-Situ Data** — Collected on-site at deployment (3–5 hours per zone). Stored in `data/raw/`.

---

## Known Limitations

- **Multi-person scenarios:** Current models assume a single occupant. Multi-person CSI is unsupported in MVP.
- **Thick concrete walls:** CSI propagation degrades significantly through reinforced concrete.
- **ESP32-S3 CSI API:** Subcarrier count varies (52 for HT20, 114 for HT40). Firmware config must match model input shape.
- **Phase noise:** ESP32 phase data is noisy. Always apply RuView's phase sanitization before frequency-domain analysis.
- **Raspberry Pi 5 inference:** Keep models small (< 5M params). Use INT8 quantization.

---

## References

- Base repo: [ruvnet/RuView](https://github.com/ruvnet/RuView)
- [Kaggle — CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)
- [ESP-IDF CSI API docs](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information)
