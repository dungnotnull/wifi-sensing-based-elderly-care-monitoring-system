# 🏥 ElderCare — WiFi-Based Elderly Monitoring System

<p align="center">
  <em>Fall Detection · Vital Signs · Sleep Monitoring · Emergency Alerts</em><br>
  <em>No cameras · No wearables · 100% Local · ~$60 USD BOM</em>
</p>

---

## Overview

**ElderCare** is a privacy-preserving elderly monitoring system that uses **WiFi CSI (Channel State Information)** — subtle disturbances in ambient WiFi signals caused by human movement and breathing — to detect falls, estimate vital signs, and track activity patterns without cameras or wearables.

Built on and extending the open-source [RuView](https://github.com/ruvnet/RuView) WiFi sensing platform, ElderCare is customized for elderly care in Vietnamese households. It leverages RuView's Rust-native vitals extractors (`wifi_densepose` package) while providing custom signal preprocessing, MQTT ingestion, and elder-specific AI models, alerting, and a caregiver-friendly dashboard.

### Why WiFi CSI?

WiFi signals already fill every home. Human bodies — even breathing — subtly alter these signals. ElderCare reads those alterations from low-cost ESP32-S3 boards and turns them into actionable caregiver information, all without cloud dependency or privacy-invasive cameras.

---

### Key Features

| Feature | Description |
|---|---|
| 🚨 **Fall Detection** | Two-stage CNN+BiLSTM with confirmation window. Target F1 > 85% |
| 🫁 **Respiration Monitoring** | FFT-based breathing rate (6–30 breaths/min), refreshed every 5 seconds |
| 😴 **Sleep Quality** | LSTM sleep staging (awake/light/deep) + Sleep Score (0–100) |
| ⚠️ **Alerting** | Three levels (INFO/WARNING/EMERGENCY) via Telegram, with cooldown |
| 📊 **Dashboard** | Real-time web UI, mobile-responsive, 16px minimum font for older caregivers |
| 🏠 **Multi-Zone** | Up to 3 zones (bedroom, living room, hallway) from 3 ESP32 nodes |
| 🔒 **Privacy-First** | No cameras, no cloud, all inference runs locally |

### How ElderCare Differs

| | ElderCare | Cameras | Wearables |
|---|---|---|---|
| Privacy preserving | ✅ | ❌ | ✅ |
| No device on body | ✅ | ✅ | ❌ |
| Works in darkness | ✅ | ❌ (needs IR) | ✅ |
| Low cost (~$60) | ✅ | ❌ | ❌ |
| 100% local (no cloud) | ✅ | Varies | Varies |
| Fall detection | ✅ | ✅ | ✅ |
| Vital signs | ✅ (breathing) | ❌ | ✅ |
| Sleep monitoring | ✅ | ❌ | ✅ |

---

## System Architecture

```
ESP32-S3 (CSI capture @ 50Hz)
    │  MQTT (eldercare/csi/{zone_id})
    ▼
ElderCare Ingestion Layer (MQTT receiver + ring buffers)
    │
    ▼
ElderCare Signal Processing (Hampel → Bandpass → Phase sanitization → Z-score)
    │
    ▼
ElderCare Inference Engine
    ├── FallDetector (two-stage: CNN+BiLSTM + confirmation)
    ├── VitalSignsEstimator (wifi_densepose Rust extractors)
    ├── SleepMonitor (ElderCare SleepLSTM → Sleep Score)
    └── ActivityTracker (rule-based + day/night context)
    │
    ▼
ElderCare Alert Manager ──► Telegram / Log / InfluxDB
    │
    ▼
ElderCare Dashboard (FastAPI + React) ──► Browser UI
```

### RuView Components We Use

| RuView Component | How Used |
|---|---|
| ESP32-S3 CSI Firmware | Raw CSI capture at 50 Hz, 52 subcarriers |
| **BreathingExtractor** | Rust-native 0.1-0.5 Hz bandpass + zero-crossing (PyO3) |
| **HeartRateExtractor** | Rust-native 0.8-2.0 Hz bandpass + autocorrelation (PyO3) |

### ElderCare Custom Components

| ElderCare Component | Value Add |
|---|---|
| **Signal Preprocessor** | Hampel + Butterworth + phase sanitization (faithful scipy reimplementation) |
| **MQTT Ingestion** | Per-zone MQTT routing with ring buffers |
| **CSI-FallNet Model** | Fine-tuned on ElderAL-CSI for elderly fall detection |
| **Two-Stage Confirmer** | 0.85 confidence + 3-second inactivity window |
| **SleepLSTM + Sleep Scorer** | Sleep staging + quality score (0–100) |
| **Alert Manager** | Vietnamese Telegram alerts, 3-level cooldown system |
| **Caregiver Dashboard** | Simplified UI, 16px font, Vietnamese labels |

---

## Hardware

| Component | Qty | Cost (USD) |
|---|---|---|
| ESP32-S3 (LoLin S3 or DevKitC) | 4 | ~$16 |
| Raspberry Pi 5 (4GB) or Mini PC | 1 | ~$55 |
| MicroSD 64GB (SanDisk Endurance) | 1 | ~$8 |
| Cables, adapters, power supplies | — | ~$6 |
| **Total** | | **~$85** |

---

## Quick Start

```bash
# Clone
git clone https://github.com/dungnotnull/WiFi-Sensing-Based-Elderly-Care-Monitoring-System.git
cd WiFi-Sensing-Based-Elderly-Care-Monitoring-System

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.template .env           # Edit .env with Telegram Bot Token + Chat IDs
# Edit configs/zones.yaml       # Set MAC addresses and zone names

# Run full stack with Docker
docker-compose -f docker/docker-compose.yml up --build

# Or run standalone
python -m pipeline.inference_engine --config configs/zones.yaml

# Dashboard
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
eldercare/
├── firmware/esp32_csi/    # ESP32-S3 CSI capture spec
├── ingestion/             # MQTT ingestion wrapper (RuView-powered)
├── models/
│   ├── fall_detection/    # ElderCare CSI-FallNet + TwoStageConfirmer
│   ├── vital_signs/       # RuView FFT vital signs adapter
│   ├── sleep/             # ElderCare SleepLSTM + Sleep Scorer
│   └── activity/          # ElderCare day/night-aware inactivity detection
├── pipeline/              # Preprocessing (RuView) + ElderCare inference engine
├── alerts/                # ElderCare alert manager (Telegram, VN)
├── dashboard/
│   ├── backend/           # FastAPI REST API
│   └── frontend/          # Caregiver React UI
├── configs/               # ElderCare YAML configs
├── training/              # Fine-tuning scripts
├── tests/                 # Test suite
├── docker/                # Docker Compose deployment
└── data/                  # CSI datasets + annotations (gitignored)
```

---

## Datasets

All datasets are free and publicly available:

- **[CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)** — Large-scale CSI dataset for pre-training
- **ElderAL-CSI** — Elderly-specific CSI dataset (local files, to be provided; place in `data/raw/elderal-csi/`)
- **In-Situ Data** — On-site collection at deployment (3–5 hours per room)

---

## Development Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| **Phase 0** | Fork RuView, establish project baseline, audit reusable components | ✅ Complete |
| **Phase 1** | Preprocessing integration (RuView signal chain), baseline model training on CSI-Bench | 🔜 Next |
| **Phase 2** | Fine-tune CSI-FallNet on ElderAL-CSI, multi-zone ingest, two-stage fall confirmation | ⏳ Planned |
| **Phase 3** | Alert Manager (Telegram integration), Dashboard backend + frontend | ⏳ Planned |
| **Phase 4** | Sleep monitoring (SleepLSTM training), system hardening, 24h stress test | ⏳ Planned |
| **Phase 5** | Documentation, accuracy evaluation, handoff | ⏳ Planned |

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=pipeline --cov=models --cov-report=html
```

---

## License

MIT License — see [LICENSE](LICENSE).

Forked from [RuView](https://github.com/ruvnet/RuView) (MIT License).

---

**ElderCare** — Privacy-preserving, low-cost elderly monitoring for Vietnamese households.
