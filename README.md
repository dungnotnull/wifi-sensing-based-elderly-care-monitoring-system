# ElderCare — WiFi-Based Elderly Monitoring System

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
| 😴 **Sleep Quality** | LSTM sleep staging (awake/light/deep) + Sleep Score (0–100). Awake F1: 0.996 |
| ⚠️ **Alerting** | Three levels (INFO/WARNING/EMERGENCY) via Telegram + Webhook, with cooldown |
| 📊 **Dashboard** | SSE-based real-time web UI, JWT authentication, vitals graphs, sleep charts, alert log, health panel |
| 🏠 **Multi-Zone** | Up to 3 zones (bedroom, living room, hallway) from 3 ESP32 nodes |
| 🔒 **Privacy-First** | No cameras, no cloud, all inference runs locally. Dashboard auth-protected |
| 🔁 **Record/Replay** | Offline CSI data recording and replay for testing |
| 📡 **Home Assistant** | MQTT auto-discovery integration for smart home setups |
| 🧠 **Adaptive Thresholds** | Self-tuning from deployment data |
| 🔬 **Shadow Mode** | Gradual model rollout with false-positive tracking, confusion matrix panel |
| 👥 **Multi-Person Detection** | Occupancy-aware alert suppression via subcarrier variance entropy |
| 🛡️ **Graceful Degradation** | 4-level fallback hierarchy (FULL/DEGRADED/MINIMAL/OFFLINE) with 7 components |
| 🌐 **Realistic CSI Simulator** | 6 noise layers: mains hum, bursty interference, multi-path fading, neighbor AP switching, thermal drift, clock jitter |

### How ElderCare Differs

| | ElderCare | Cameras | Wearables |
|---|---|---|---|
| Privacy preserving | Yes | No | Yes |
| No device on body | Yes | Yes | No |
| Works in darkness | Yes | No (needs IR) | Yes |
| Low cost (~$60) | Yes | No | No |
| 100% local (no cloud) | Yes | Varies | Varies |
| Fall detection | Yes | Yes | Yes |
| Vital signs | Yes (breathing) | No | Yes |
| Sleep monitoring | Yes | No | Yes |

---

## System Architecture

```
ESP32-S3 (RuView firmware, CSI capture @ ~20 Hz)
    │  UDP (ADR-018 binary, port 5005)
    ▼
ElderCare UDP-to-MQTT Bridge ──► MQTT (eldercare/csi/{zone_id})
    │
    ▼
ElderCare Ingestion Layer (MQTT receiver + ring buffers + CSI Quality Checker)
    │
    ▼
ElderCare Signal Processing (Hampel → Bandpass → Phase sanitization → Z-score)
    │  [Correlation ID tracking starts here]
    ▼
ElderCare Inference Engine (WorkerWatchdog health monitoring + auto-restart)
    ├── FallDetector (two-stage: CNN+BiLSTM + confirmation)
    ├── VitalSignsEstimator (wifi_densepose Rust / Python scipy fallback)
    ├── SleepMonitor (ElderCare SleepLSTM → Sleep Score, 6 features)
    ├── ActivityTracker (rule-based + day/night context)
    ├── MultiPersonDetector (subcarrier spread ratio + variance entropy)
    ├── Adaptive Thresholds (self-tuning from deployment data)
    └── Shadow Mode (gradual rollout with FP tracking)
    │
    ├──► SQLite Persistence (crash recovery)
    ├──► InfluxDB Writer (time-series storage)
    ├──► Local Telemetry (inference latency, system metrics)
    ├──► Performance Tracker (confusion matrix, per-class F1, confidence distributions)
    └──► Degradation Manager (FULL → DEGRADED → MINIMAL → OFFLINE)
    │
    ▼
ElderCare Alert Manager ──► Telegram / Webhook / Log (i18n: Vietnamese + English)
    │
    ▼
ElderCare Dashboard (FastAPI + SSE + React, JWT auth) ──► Browser UI
```

### RuView Components We Use

| RuView Component | How Used |
|---|---|
| ESP32-S3 CSI Firmware (`esp32-csi-node`, v0.6.5) | Raw CSI capture via ADR-018 binary protocol over UDP |
| `wifi_densepose` v2.0.0a1 | `BreathingExtractor` + `HeartRateExtractor` (Rust PyO3) |
| UDP-to-MQTT Bridge | Custom integration layer — converts ADR-018 → MQTT for multi-zone routing |

> **Audit note:** `wifi_densepose` exposes 13 types but only the two extractors are used. RuView's signal processing chain (Hampel/Butterworth/phase) lives in a Rust binary — not importable as Python. Our `pipeline/preprocessor.py` is a faithful scipy reimplementation (130x vectorized). No other RuView Python utilities exist.

### ElderCare Custom Components

| ElderCare Component | Value Add |
|---|---|
| **Signal Preprocessor** | Hampel + Butterworth + phase sanitization (faithful scipy reimplementation) |
| **MQTT Ingestion** | Per-zone MQTT routing with ring buffers |
| **CSI-FallNet Model** | Fine-tuned on ElderAL-CSI for elderly fall detection |
| **Two-Stage Confirmer** | 0.85 confidence + 3-second inactivity window |
| **SleepLSTM + Sleep Scorer** | Sleep staging + quality score (0–100), 6 features incl. wakefulness index |
| **Alert Manager** | Vietnamese Telegram + Webhook alerts, 3-level cooldown system |
| **Caregiver Dashboard** | Full React UI with JWT auth, vitals graphs, sleep charts, alert log, health panel |
| **Python Vitals Fallback** | Pure scipy vital signs when wifi_densepose unavailable |
| **SQLite Persistence** | Crash recovery — events survive process restarts |
| **InfluxDB Writer** | Time-series storage for vitals and system metrics |
| **Worker Watchdog** | Auto-restart failed inference workers |
| **CSI Quality Checker** | SNR, packet loss, and null subcarrier detection |
| **Model Registry** | Config-driven worker creation for all model types |
| **Correlation Tracker** | End-to-end latency tracing through the pipeline |
| **Adaptive Thresholds** | Self-tuning from deployment data |
| **Record/Replay** | Offline CSI data recording and replay for testing |
| **Home Assistant Integration** | MQTT auto-discovery for smart home setups |
| **Model Quantization** | INT8 quantization for Raspberry Pi 5 deployment |
| **Shadow Mode** | Gradual model rollout with false-positive tracking |
| **Local Telemetry** | Inference latency, system metrics collection |
| **i18n** | Vietnamese + English locale support for alerts and UI |
| **Focal Loss Sleep Training** | Balanced class detection for sleep staging |
| **Confidence Calibration** | Temperature scaling + temporal smoothing |
| **SSE Real-time Dashboard** | Server-Sent Events replacing HTTP polling |
| **Realistic CSI Simulator** | 6 noise layers (mains hum, bursty interference, multi-path fading, neighbor AP switching, thermal drift, clock jitter) |
| **Multi-Person Detector** | Occupancy-aware alert suppression via subcarrier variance entropy |
| **Webhook Alert Channel** | Generic webhook dispatch with circuit breaker and retry |
| **Dashboard Authentication** | JWT-based login/logout for caregiver dashboard |
| **Performance Tracker** | Per-model confusion matrix + confidence distribution (API endpoint) |
| **Degradation Manager** | 4-level fallback hierarchy (FULL/DEGRADED/MINIMAL/OFFLINE) with 7 components |
| **Docker E2E Tests** | Automated docker-compose integration test suite |

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

# Run full stack with Docker (3 services: MQTT + UDP bridge + server)
docker-compose -f docker/docker-compose.yml up --build

# Or run standalone
# Terminal 1: Start UDP→MQTT bridge (for ESP32 hardware)
python -m ingestion.udp_mqtt_bridge --config configs/zones.yaml
# Terminal 2: Start inference engine
python -m pipeline.inference_engine --config configs/zones.yaml
# Terminal 3: Dashboard
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000

# Without hardware: simulate CSI data for development
python -c "from firmware.esp32_csi.csi_simulator import CSISimulator; \
  sim = CSISimulator(); sim.set_state('breathing'); \
  [print(sim.generate_packet()) for _ in range(5)]"

# Record CSI data for offline replay
python -c "from pipeline.record_replay import CSIRecorder; \
  recorder = CSIRecorder(output_dir='data/raw/session'); \
  recorder.start()"

# Quantize model for RPi5
python -c "from pipeline.quantization import quantize_from_checkpoint; \
  quantize_from_checkpoint('models/fall_detection/checkpoints/best.pt', output_path='models/fall_detection/quantized/')"

# Evaluate with calibration
python -m pipeline.evaluate
```

---

## Project Structure

```
eldercare/
├── firmware/
│   ├── esp32_csi/         # CSI packet spec + realistic simulator (6 noise layers)
│   └── esp32-csi-node/    # RuView ESP32-S3 firmware (C, esp-idf v5.2, v0.6.5)
├── ingestion/
│   ├── udp_mqtt_bridge.py # UDP-to-MQTT bridge (ADR-018 binary → MQTT topics)
│   └── receiver.py        # MQTT ingestion + ring buffers + quality gate
├── models/
│   ├── fall_detection/    # ElderCare CSI-FallNet + TwoStageConfirmer
│   ├── vital_signs/       # wifi_densepose adapter + Python scipy fallback
│   │   └── python_fallback.py  # Pure scipy vitals when wifi_densepose unavailable
│   ├── sleep/             # ElderCare SleepLSTM (6-feature) + Sleep Scorer
│   ├── activity/          # ElderCare day/night-aware inactivity detection
│   └── calibration.py     # Confidence calibration (temperature scaling + smoothing)
├── pipeline/
│   ├── preprocessor.py    # Vectorized signal processing (130x speedup)
│   ├── inference_engine.py  # Multiprocessing inference (12 workers, 3 zones)
│   ├── data_store.py      # Thread-safe inference data store
│   ├── persistence.py     # SQLite persistence for crash recovery
│   ├── watchdog.py        # WorkerWatchdog health monitoring + auto-restart
│   ├── csi_quality.py     # CSI signal quality validation (SNR, packet loss)
│   ├── influx_writer.py   # InfluxDB time-series writer
│   ├── model_registry.py  # Config-driven model worker creation
│   ├── correlation.py     # End-to-end latency tracing (Correlation IDs)
│   ├── adaptive_thresholds.py  # Self-tuning thresholds
│   ├── record_replay.py   # CSI data recording and replay
│   ├── homeassistant.py   # Home Assistant MQTT auto-discovery
│   ├── quantization.py    # INT8 model quantization for RPi5
│   ├── shadow_mode.py     # Shadow mode gradual rollout
│   ├── telemetry.py       # Local telemetry (Prometheus format)
│   ├── multi_person.py    # Multi-person occupancy detection
│   ├── performance_tracker.py  # Confusion matrix + per-class metrics
│   ├── degradation.py     # Graceful degradation hierarchy (7 components)
├── alerts/
│   ├── alert_manager.py   # Telegram + Webhook dispatch (circuit breaker)
│   └── i18n.py            # Vietnamese + English locale support
├── dashboard/
│   ├── backend/
│   │   ├── main.py        # FastAPI REST API + SSE (JWT-protected)
│   │   └── auth.py        # JWT authentication module
│   └── frontend/          # React caregiver UI (login + dashboard)
├── configs/               # YAML configs (zones, thresholds, alerts, models)
│   └── locales/           # i18n locale files (vi.yaml, en.yaml)
├── training/              # Fine-tuning scripts + dataset mappers
├── tests/                 # Test suite (60 tests: 51 unit + 9 bridge + 1 Docker E2E)
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
| **Phase 0** | Fork RuView, establish project baseline, audit reusable components | Complete |
| **Phase 1** | Preprocessing integration, baseline model training on mock data | Complete |
| **Phase 2** | Fine-tune CSI-FallNet, multi-zone ingest, two-stage fall confirmation | Complete |
| **Phase 3** | Alert Manager (Telegram/Webhook), Dashboard backend + frontend + auth | Complete |
| **Phase 4** | Sleep monitoring (SleepLSTM 6-feature), system hardening, 24h stress test | Complete |
| **Phase 5** | Documentation, accuracy evaluation, handoff | Complete |
| **Phase 6** | System Upgrade v0.2: 19 modules (persistence, watchdog, SSE, shadow mode, etc.) | Complete |
| **Phase 7** | System Hardening v0.3: realistic simulator, sleep fix, auth, webhook, multi-person, confusion matrix, E2E tests, degradation | Complete |

---

## Running Tests

```bash
# Run all 60 tests
pytest tests/ -v

# Run with coverage (target: 70% for pipeline/ and models/)
pytest tests/ --cov=pipeline --cov=models --cov-report=html
```

---

## License

MIT License — see [LICENSE](LICENSE).

Forked from [RuView](https://github.com/ruvnet/RuView) (MIT License).

---

**ElderCare v0.3.0** — Privacy-preserving, low-cost elderly monitoring for Vietnamese households.
