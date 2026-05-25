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

ElderCare leverages RuView's battle-tested infrastructure. The rule is: **import from RuView for signal processing, ingestion, and vital signs; keep only what is ElderCare-specific.**

### ✅ Imported from RuView (`wifi_densepose` package)

| RuView Component | Used For |
|---|---|
| ESP32-S3 CSI Firmware | Raw CSI capture at 50 Hz, HT20 mode, 52 subcarriers |
| Hampel Filter | Impulse noise removal from CSI amplitude |
| Phase Sanitizer | Unwrapping + linear detrend + Butterworth bandpass |
| Amplitude Normalization | Per-subcarrier z-score with EMA running stats |
| Breathing Extractor | FFT-based respiration rate (0.1–0.5 Hz band) |
| Heart Rate Extractor | FFT-based HR (0.8–2.0 Hz band, experimental) |
| MQTT Client + Ring Buffers | CSI packet ingestion + validation |
| Vital Anomaly Detector | Apnea, tachypnea, bradycardia detection |

### 🔧 ElderCare-Specific (Custom Code)

| ElderCare Module | Why Custom |
|---|---|
| **CSI-FallNet** (`models/fall_detection/`) | Fine-tuned on ElderAL-CSI for elderly motion patterns |
| **TwoStageConfirmer** | Confidence threshold (0.85) + 3-second inactivity re-check |
| **SleepLSTM + SleepScorer** | No RuView equivalent — sleep staging and quality score are elderly-care-specific |
| **Alert Manager** (`alerts/`) | Vietnamese Telegram messages, 3-level cooldown, caregiver-oriented |
| **Activity Detector** | Day/night context (suppress alerts during sleep hours 10PM-6AM) |
| **Caregiver Dashboard** (`dashboard/`) | Simplified UI, 16px minimum font, Vietnamese labels |
| **Configs** (`configs/`) | Elderly-appropriate thresholds, Vietnamese zone names, MQTT topics |

**Rule of thumb:** If a module implements pure signal processing (filters, FFT, phase math), it should be imported from RuView. If it encodes elderly-care-specific logic (alert levels, sleep scoring, daytime awareness), it belongs in ElderCare.

---

## Core Technologies

| Layer | Technology |
|---|---|
| CSI Capture | ESP32-S3 (esp-idf CSI API) — RuView firmware |
| Transport | MQTT (Mosquitto broker) |
| Signal Processing | RuView signal chain (Hampel, bandpass, phase sanitization) |
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
- **Respiration:** RuView FFT engine on phase variance (0.1–0.5 Hz band)
- **Heart Rate:** RuView engine (0.8–2.0 Hz, experimental — marked as such in UI)
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
ingestion/receiver.py  ──► RuView MQTT client + ring buffer (5s)
    │
    ▼
RuView Signal Processing
    ├── Hampel filter (outlier removal)
    ├── Butterworth bandpass
    ├── Phase sanitization (unwrapping + linear detrend)
    └── Amplitude normalization (per-subcarrier z-score)
    │
    ▼
pipeline/inference_engine.py  (multiprocessing, one process per model)
    ├── FallDetector (CSI-FallNet + TwoStageConfirmer)
    ├── VitalSignsEstimator (RuView FFT engine)
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

1. **Import RuView, don't rebuild.** If RuView provides it (signal processing, vitals, ingestion), import it. Only build ElderCare-specific logic.
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
