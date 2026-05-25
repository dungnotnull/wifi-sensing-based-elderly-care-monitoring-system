# PROJECT-DETAIL.md — ElderCare

**ElderCare: WiFi Sensing-Based Elderly Care Monitoring System**
Version: 0.1 (MVP Specification)
Last Updated: 2026-05-25
Author: ElderCare Development Team

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technical Assessment](#2-technical-assessment)
3. [System Architecture](#3-system-architecture)
4. [Deep Learning Design](#4-deep-learning-design)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Hardware Specification](#7-hardware-specification)
8. [Dataset Strategy](#8-dataset-strategy)
9. [Development Roadmap](#9-development-roadmap)
10. [Risk Analysis](#10-risk-analysis)
11. [Budget Breakdown](#11-budget-breakdown)
12. [Expected Outputs](#12-expected-outputs)
13. [Future Roadmap](#13-future-roadmap)

---

## 1. Project Overview

### 1.1 Problem Statement

Elderly individuals living alone face significant safety risks from falls, sudden health deterioration, and prolonged inactivity. Traditional monitoring solutions either require cameras (privacy invasion) or expensive medical-grade sensors (cost prohibitive for Vietnamese families). The gap between safety and privacy is the core problem this project addresses.

### 1.2 Solution

ElderCare leverages **WiFi Channel State Information (CSI)** — fluctuations in the WiFi signal caused by human body movement — to detect falls, estimate vital signs, and monitor activity patterns, all without any camera or wearable device.

The system is forked from [RuView](https://github.com/ruvnet/RuView) (a general-purpose WiFi sensing framework) and customized with:
- Eldercare-specific ML models trained on relevant datasets
- A simplified, caregiver-friendly dashboard
- Multi-zone support for Vietnamese home layouts
- Telegram-based alerting for Vietnamese users

**Development Strategy:** ElderCare imports RuView for all signal processing infrastructure (Hampel filter, phase sanitizer, Butterworth bandpass, breathing/heart rate extraction, MQTT ingestion). Only elderly-care-specific components are custom-built: CSI-FallNet fine-tuning on ElderAL-CSI, SleepLSTM + Sleep Scorer, TwoStageConfirmer, alert manager with Vietnamese localization, and caregiver dashboard. This avoids reinventing the wheel on logic already built into RuView.

### 1.3 Key Differentiators

| Feature | ElderCare | Camera Systems | Wearables |
|---|---|---|---|
| Privacy preserving | ✅ | ❌ | ✅ |
| No device on body | ✅ | ✅ | ❌ |
| Works in dark/night | ✅ | ❌ (IR needed) | ✅ |
| Low cost (~$50 USD) | ✅ | ❌ | ❌ |
| 100% local | ✅ | Varies | Varies |
| Fall detection | ✅ | ✅ | ✅ |
| Vital signs | ✅ (breathing) | ❌ | ✅ |
| Sleep monitoring | ✅ | ❌ | ✅ |

---

## 2. Technical Assessment

### 2.1 Project Feasibility

**Overall assessment: Technically feasible with moderate-to-high complexity.**

This is a serious deep learning systems project. The following assessment identifies key challenges and mitigations.

### 2.2 Strengths of the Approach

- **WiFi CSI is well-researched.** Academic literature since 2015 (WIFALL, FallDeFi, etc.) confirms CSI can detect falls with >90% accuracy in controlled settings.
- **RuView provides a solid foundation.** The base repo already handles CSI ingestion, basic signal processing, and a web dashboard, reducing development time significantly. ElderCare imports these components directly — no need to rebuild.
- **ESP32-S3 is mature hardware.** The CSI API in esp-idf is documented and community-supported. MicroPython alternatives also exist.
- **Datasets exist.** CSI-Bench and similar public datasets provide labeled data for pre-training, reducing annotation burden.

### 2.3 Key Technical Challenges

**Challenge 1: Domain gap between lab datasets and real homes**
- Academic CSI datasets are captured in controlled environments (empty rooms, specific distances).
- Real Vietnamese homes have furniture, thick walls, multiple WiFi interference sources.
- **Mitigation:** Fine-tune on in-situ data captured at deployment site. Plan 3-5 hours of data collection per zone at deployment.

**Challenge 2: Phase noise on ESP32**
- ESP32 does not have hardware phase calibration unlike Intel 5300/Atheros CSI tools.
- Phase data requires aggressive preprocessing (sanitization, linear detrending).
- **Mitigation:** RuView's phase sanitization pipeline handles this. Rely primarily on **amplitude-based features** for fall detection. Use phase only for vital signs after sanitization pipeline.

**Challenge 3: Fall vs. non-fall discrimination**
- Fast sitting down, dropping objects, or pets can trigger false positives.
- **Mitigation:** Two-stage confirmation (initial trigger + 3-second inactivity check). Collect negative examples (lying down slowly, sitting) during fine-tuning.

**Challenge 4: Multi-zone CSI interference**
- Running 3-4 ESP32 nodes simultaneously can cause inter-device interference.
- **Mitigation:** Use time-division CSI capture or assign non-overlapping channels per zone. Implement zone isolation in ingestion pipeline.

**Challenge 5: Raspberry Pi 5 inference throughput**
- Running 4 models at 50 Hz on CPU is demanding.
- **Mitigation:** Use inference batching (process 1-second windows), quantize models to INT8, keep architectures small (< 5M parameters). Profile early.

### 2.4 Realism Check on Targets

| Target | Assessment | Comment |
|---|---|---|
| Fall detection > 85% F1 | Achievable | Requires in-situ fine-tuning; lab-only weights typically reach 70-75% in real homes |
| Latency < 5 seconds | Achievable | 2s window + ~200ms inference on RPi5 = well within target |
| 24/7 stability | Requires effort | Must handle ESP32 WiFi reconnects, queue backpressure, memory leaks in long-running processes |
| Breathing monitoring | Achievable | FFT-based, well-established. Accuracy degrades during movement. RuView provides this directly |
| Heart rate monitoring | Difficult | ESP32 amplitude noise makes fine HR extraction unreliable. Mark as "experimental" for MVP |
| Sleep quality scoring | Partially achievable | Movement + breathing proxy works; EEG-level accuracy impossible; set expectation correctly |
| 6-8 week timeline | Tight but realistic | Only with scoped MVP; full feature set needs 12+ weeks |

---

## 3. System Architecture

### 3.1 High-Level Diagram

```
ESP32-S3 Zone 1     ESP32-S3 Zone 2     ESP32-S3 Zone 3
 (Bedroom)           (Living Room)        (Hallway)
     |                   |                   |
     +-----------WiFi / MQTT------------------+
                         |
            Raspberry Pi 5 / Mini PC
                         |
            +----------- Ingestion Layer (RuView) ----------+
            |                                                |
      RuView Signal Processing                    ElderCare Inference Engine
      (Hampel, Bandpass, Phase, Normalize)        (Fall, Vitals, Sleep, Activity)
            |                                                |
            +----------- Alert Manager (ElderCare) ---------+
                         | (Telegram, Log, InfluxDB)
                    Dashboard (FastAPI + React)
                         |
                    Caregiver Browser
```

### 3.2 Component Descriptions

**ESP32-S3 Firmware (RuView)**
Each node runs RuView's CSI capture firmware that samples WiFi channel state at ~50 Hz and transmits structured packets via MQTT to the local server.

**Ingestion Layer (RuView)**
- MQTT subscription per zone (topic: `eldercare/csi/{zone_id}`)
- Packet validation and ring buffer management
- Zone-ID tagging, timestamp, sequence numbers

**Preprocessing Pipeline (RuView)**
- Hampel filter for impulse noise removal
- Butterworth bandpass filter
- Phase unwrapping + linear detrend
- Subcarrier selection and z-score normalization

**ML Inference Engine (ElderCare)**
Four parallel processes, one per model. Uses RuView's preprocessing output.

**Alert Manager (ElderCare)**
Three-level alert system (INFO/WARNING/EMERGENCY) with Vietnamese-language Telegram messages, cooldown management, and log persistence.

**Dashboard (ElderCare)**
FastAPI backend + React frontend, mobile-responsive, caregiver-optimized with 16px minimum font.

---

## 4. Deep Learning Design

### 4.1 Fall Detection Model

**Problem type:** Binary time-series classification (`fall` vs. `non-fall`)

**Architecture: CSI-FallNet**
```
Input: (Batch, T=100, C=52)  <- 2-second window, 52 subcarriers
  |
  +-- Conv1D(52->64, kernel=5, padding=same) + BatchNorm + ReLU
  +-- Conv1D(64->128, kernel=3, padding=same) + BatchNorm + ReLU
  +-- MaxPool1D(2) -> (Batch, 50, 128)
  +-- Conv1D(128->128, kernel=3, padding=same) + BatchNorm + ReLU
  +-- MaxPool1D(2) -> (Batch, 25, 128)
  |
  +-- BiLSTM(128, hidden=256, bidirectional=True) -> (Batch, 25, 512)
  +-- Attention pooling -> (Batch, 512)
  |
  +-- FC(512->256) + Dropout(0.5) + ReLU
  +-- FC(256->64) + ReLU
  +-- FC(64->2) + Softmax
```

**Training details:**
- Loss: Weighted CrossEntropy (weight falls heavier due to class imbalance ~1:10)
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR
- Augmentation: Time shift, Gaussian noise injection, subcarrier dropout
- Pre-training: CSI-Bench (multi-activity classification task)
- Fine-tuning: ElderAL-CSI + in-situ captured data

**Two-stage confirmation:**
```python
# Stage 1: model confidence > 0.85
# Stage 2: if fall triggered, check inactivity for next 3 seconds
#           if CSI variance drops below inactivity_threshold -> confirm fall
#           else -> dismiss as false positive
```

### 4.2 Vital Signs Estimation

**Respiration Rate (RuView FFT Engine):**
```
Input: 30-second CSI phase segment (per subcarrier)
  |
  +-- Phase sanitization (unwrap + detrend) [RuView]
  +-- Select top-K subcarriers by signal variance
  +-- Average across subcarriers
  +-- FFT -> power spectrum
  +-- Bandpass: 0.1-0.5 Hz (6-30 breaths/min) [RuView]
  +-- Peak frequency -> Respiration Rate (BPM) [RuView]
```

**Heart Rate (Experimental — RuView Engine):**
Same pipeline with 0.8-2.0 Hz bandpass. Accuracy is ESP32-hardware-limited; treated as experimental in MVP with explicit uncertainty display in dashboard.

### 4.3 Sleep Quality Monitoring

**Feature extraction (per 1-minute epoch):**
- Mean respiration rate
- Respiration rate variability (std)
- Movement index (mean CSI amplitude variance)
- Body movement burst count (threshold crossings)

**Architecture: SleepLSTM**
```
Input: (Batch, N_epochs, 4_features)
  |
  +-- LSTM(4, hidden=64, num_layers=2, dropout=0.3)
  +-- FC(64->32) + ReLU
  +-- FC(32->3) + Softmax  <- {awake, light, deep}
```

**Sleep Score:** Weighted sum of deep sleep proportion, total sleep time, and respiratory regularity (normalized 0-100). This is ElderCare-specific — RuView has no equivalent.

### 4.4 Activity / Inactivity Detection

**Rule-based (primary for MVP):**
```
CSI Amplitude Variance over 30s window:
  > threshold_active   -> ACTIVE
  > threshold_still    -> STILL (breathing only, e.g., sleeping)
  < threshold_still    -> INACTIVITY (potential problem)

If INACTIVITY persists > 120 minutes -> WARNING alert
If INACTIVITY follows a FALL event > 30 seconds -> EMERGENCY alert

Day/night context (ElderCare-specific):
  Inactivity alerts suppressed during sleep hours (10 PM - 6 AM)
```

**DL enhancement (post-MVP):** Fine-grained activity classification (walking, sitting, lying, standing) using a ResNet-style 1D-CNN.

---

## 5. Functional Requirements

### FR-01: Fall Detection
- Detect a fall event within 5 seconds of occurrence
- Two-stage confirmation to reduce false positives
- Distinguish between: fall, lying down normally, sitting down quickly

### FR-02: Post-Fall Inactivity Alert
- If no recovery movement detected within 30 seconds post-fall -> EMERGENCY alert
- Alert includes: zone name, timestamp, confidence score

### FR-03: Prolonged Inactivity Alert
- Alert if no significant movement detected for > 2 hours during daytime hours (configurable)
- No alert during configured sleep hours (e.g., 10 PM - 6 AM)

### FR-04: Vital Signs Monitoring
- Estimate respiration rate (breaths/min) updated every 5 seconds **via RuView FFT engine**
- Display respiration trend graph (last 1 hour) on dashboard
- Alert on abnormal respiration rate (< 8 or > 25 breaths/min sustained > 1 minute)

### FR-05: Sleep Quality Monitoring
- Track sleep sessions automatically (no manual start/stop)
- Produce a Sleep Score (0-100) per night **via ElderCare SleepLSTM**
- Send morning summary report at configurable time (default: 7:00 AM)

### FR-06: Multi-Zone Support
- Support 2-4 zones simultaneously
- Each zone identified by zone name and ESP32 MAC address
- Dashboard shows per-zone status independently

### FR-07: Alerting
- Telegram integration (mandatory)
  - Three alert levels: INFO, WARNING, EMERGENCY
  - Each alert message includes: zone, event type, timestamp, severity, brief description
  - **Messages in Vietnamese for caregiver accessibility**
- Daily summary report (text format, Telegram)
- Alert cooldown: configurable per-level (default: 5 min WARNING, 1 min EMERGENCY)

### FR-08: Dashboard
- Real-time status per zone (last seen, current activity, breathing rate)
- Historical graphs: breathing rate (24h), sleep score (30 days), activity heatmap (7 days)
- Alert log with acknowledgment
- System health panel (ESP32 connection status, inference latency, disk usage)
- **Minimum 16px font for readability by older caregivers**

---

## 6. Non-Functional Requirements

### NFR-01: Privacy
- No camera, no image data
- All data stored locally, never transmitted to cloud except alert text
- Alert messages must NOT contain personally identifying information beyond zone name

### NFR-02: Performance
- Fall detection end-to-end latency: < 5 seconds
- Dashboard page load: < 2 seconds on local network
- System must sustain 50 Hz CSI ingestion from 3 zones without packet loss > 1%

### NFR-03: Reliability
- Target uptime: 99.5% (< 4 hours downtime/month)
- Auto-recovery on ESP32 WiFi disconnect (firmware-level reconnect loop)
- Watchdog process to restart failed inference workers

### NFR-04: Accuracy
- Fall detection F1 score: > 85% after site-specific fine-tuning
- False positive rate: < 2 per day in normal household conditions

### NFR-05: Usability
- Dashboard accessible on mobile browser without app installation
- Font size minimum 16px for readability by older caregivers
- Alert messages in Vietnamese

### NFR-06: Security
- Dashboard accessible only on local network (no port forwarding by default)
- Telegram bot token stored in encrypted `.env` file
- No default passwords; setup wizard enforces configuration

---

## 7. Hardware Specification

### 7.1 ESP32-S3 Nodes

**Recommended board:** LoLin S3 or ESP32-S3-DevKitC-1  
**Quantity:** 3 (MVP) — 1 per zone  
**Why ESP32-S3:** Improved RF performance vs. ESP32 original; CSI API available in esp-idf v5.x; sufficient RAM for firmware

**Node placement guidelines:**
- Mount at 1.0-1.5m height (chest level when standing)
- Line-of-sight or one-wall clearance to monitored area preferred
- Avoid placement behind large metal objects or appliances
- Minimum 3m separation between nodes to reduce inter-node interference

**Per-zone configuration (1 TX + 1 RX or router-RX mode):**
- Option A: 2 ESP32-S3 per zone (one TX beacon, one RX CSI) — better isolation
- Option B: 1 ESP32-S3 per zone receiving CSI from router (if router supports CSI output) — lower cost

### 7.2 Local Server

**Recommended:** Raspberry Pi 5 (8GB RAM) or equivalent x86 Mini PC

**Minimum specs:**
- CPU: 4-core ARM64 or x86-64, 1.8 GHz+
- RAM: 4 GB (8 GB recommended for comfortable headroom)
- Storage: 32 GB microSD Class 10 (RPi5) or SSD
- Network: 100 Mbps Ethernet (preferred over WiFi to avoid interference)
- OS: Ubuntu Server 24.04 LTS or Raspberry Pi OS Lite (64-bit)

**Why local server:** All ML inference, database, and dashboard services run here. No cloud dependency.

### 7.3 Network

- Standard 2.4 GHz WiFi router (5 GHz not needed; 2.4 GHz has better wall penetration)
- MQTT broker (Mosquitto) runs on the local server
- All ESP32 nodes connect to same WiFi network as server
- Server connects to router via Ethernet

---

## 8. Dataset Strategy

All datasets are free and publicly available. No paid datasets required.

### 8.1 Primary: CSI-Bench

A large-scale WiFi CSI dataset covering multiple human activities including falls, walking, sitting, lying, and standing. Used for pre-training models on general activity recognition.

- **Source:** [Kaggle — CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)
- **License:** Free for research use
- **Download:** Via Kaggle (web UI or `kagglehub` Python library)

**Usage:** Pre-train fall detection and activity classification models. Extract learned representations as backbone weights before fine-tuning.

### 8.2 Secondary: ElderAL-CSI

An eldercare-specific CSI dataset with labeled fall events and activities of daily living recorded with elderly subjects. More representative of target population.

- **Source:** Local files (to be provided)
- **Setup:** Place downloaded dataset files in `data/raw/elderal-csi/`. Detailed instructions will be added once files are available.

**Usage:** Fine-tune fall detection model. Validation set for evaluating model performance on elderly-specific motion patterns.

### 8.3 In-Situ Data Collection (Required)

No public dataset perfectly matches a specific deployment environment. After installing hardware, collect 3-5 hours of labeled data per zone:

**Collection protocol:**
- Falls: Simulate falls (safely, with mattress/pad) in 10 different directions and speeds — 50+ examples per zone
- Non-falls: Normal activities (sitting, lying, standing, walking) — 200+ examples per zone
- Labeling: Manual annotation using a simple CLI tool (`python training/annotator.py`)

**Storage:** Raw CSI stored in `data/raw/`, processed in `data/processed/`, annotations in `data/annotations/`

### 8.4 Data Augmentation

To compensate for limited in-situ data:
- **Time shifting:** Randomly shift fall event position within the window
- **Gaussian noise injection:** Simulate SNR variation
- **Subcarrier dropout:** Randomly zero out 10% of subcarriers
- **Amplitude scaling:** Random scaling +/-20%
- **Mixup:** Blend two non-fall samples with label mixing

---

## 9. Development Roadmap

### Phase 0 — Setup (Week 1) ✅ COMPLETE

- [x] Fork RuView repository, create `feature/eldercare` branch
- [x] Audit RuView codebase: identify reusable components, strip unused features
- [x] Set up development environment (Docker, PyTorch, MQTT)
- [ ] Flash ESP32-S3 firmware, verify CSI data capture to console
- [x] Establish MQTT ingestion pipeline architecture: ESP32 -> local server
- [x] Commit baseline project structure per CLAUDE.md
- [x] **RuView component audit complete:** Signal processing (Hampel, phase sanitizer, Butterworth bandpass), vital signs extraction (FFT respiration + heart rate), MQTT ingestion, ESP32 firmware — all identified for import. ElderCare builds only: CSI-FallNet fine-tuning, SleepLSTM, TwoStageConfirmer, Alert Manager with VN localization, caregiver dashboard.

**Milestone:** Project structure established. RuView components identified and audited. ElderCare-specific code scaffolded (models, alerts, dashboard, configs, training). ✅

### Phase 1 — Signal Processing & Baseline Model (Weeks 2-3)

- [x] **CSI-FallNet training loop built** — PyTorch Dataset + DataLoader + AdamW + CosineAnnealingLR + weighted CrossEntropy + augmentation (time shift, Gaussian noise, subcarrier dropout). Model: 2.6M params (under 5M target). Checkpoint saving + metrics tracking.
- [x] **Synthetic CSI dataset generator** — CSI-Bench format (T=100, C=52), 4 classes: fall/idle/breathing/movement. Realistic class imbalance (~1:10 fall ratio). Generates correct tensor shapes verified.
- [ ] Download real CSI-Bench from Kaggle (deferred — large dataset, manual download required)
- [ ] Train baseline CSI-FallNet on CSI-Bench (blocked by dataset — synthetic-only checkpoint exists at `models/fall_detection/checkpoints/`)
- [ ] Evaluate baseline on ElderAL-CSI test split (blocked by dataset — eval function implemented in training script)
- [x] **RuView FFT respiration + heart rate estimators reimplemented** — FFT-based vitals with top-K subcarrier selection, confidence scoring. Respiration: 0.1-0.5 Hz (tested 16 BPM vs 15 BPM ground truth). Heart rate: 0.8-2.0 Hz (tested 72 BPM exact). Note: reimplemented locally rather than imported from RuView package — functionally equivalent. Includes PhaseDenoiser (optional 1D-CNN for phase enhancement).
- [x] **Unit tests for model wrappers** — test_fall_detection.py (FallDetector + TwoStageConfirmer), test_vital_signs.py (RespirationEstimator + HeartRateEstimator). Verify architectures, output shapes, parameter counts, and edge cases.

**Milestone:** Baseline fall detection at >70% F1 on public dataset. RuView signal processing and vitals engines integrated.

### Phase 2 — Fine-Tuning & Multi-Zone (Weeks 4-5)

- [ ] Deploy 3 ESP32 nodes in test environment (requires hardware)
- [ ] Collect in-situ fall + activity data (labeled) (requires hardware + annotator tool)
- [ ] Fine-tune CSI-FallNet with CSI-Bench -> ElderAL-CSI -> in-situ pipeline (requires real datasets)
- [x] **Implement two-stage fall confirmation logic** — TwoStageConfirmer wired into FallDetectionWorker. Confidence threshold 0.85 + 3-second CSI variance inactivity check. Posts FallConfirmationEvent to shared per-zone queue.
- [x] **Multi-zone ingestion (zone ID tagging, per-zone ring buffers)** — Per-worker input queues guarantee every worker receives every packet for its zone. 3 zones x 4 workers = 12 workers. Zone isolation verified by integration tests.
- [x] **Implement day/night-aware inactivity detection (rule-based)** — ActivityDetector with daytime hours (6AM-10PM) wired into ActivityWorker. Suppresses inactivity alerts during sleep hours. PostFallInactivityChecker monitors 30s post-fall recovery window, escalates to EMERGENCY if no movement.
- [ ] Target: >85% F1 on in-situ test set (requires real datasets)

**Milestone:** Fall detection meeting accuracy target. Three zones operational.

### Phase 3 — Alerting & Dashboard (Week 6)

- [x] Alert Manager scaffolded (3-level, cooldown, Vietnamese formatting, Telegram stub)
- [ ] Wire Alert Manager to real Telegram Bot API
- [x] Dashboard backend scaffolded (FastAPI, zone/vitals/alerts/sleep/health endpoints)
- [x] Dashboard frontend scaffolded (React, zone cards, mobile-responsive, 16px minimum font)
- [ ] Wire dashboard to live inference data (currently using stub/placeholder data)
- [ ] Daily summary report generation and scheduled delivery

**Milestone:** End-to-end system: fall -> Telegram alert within 5 seconds. Dashboard accessible on local network.

### Phase 4 — Sleep Monitoring & Hardening (Week 7)

- [ ] Implement SleepLSTM training and inference (ElderCare-specific — no RuView equivalent)
- [ ] Sleep score calculation and nightly session detection
- [ ] Morning report automation (scheduled Telegram message)
- [ ] System hardening: watchdog process, auto-reconnect, error handling
- [ ] Docker Compose full stack deployment
- [ ] Load testing: 3 zones x 50 Hz continuous for 24 hours

**Milestone:** Full MVP feature set operational. 24-hour stress test passed.

### Phase 5 — Documentation & Handoff (Week 8)

- [ ] Write installation guide (`docs/installation.md`)
- [ ] Write user guide for caregivers (`docs/user_guide.md`)
- [ ] Write system design document (`docs/architecture.md`)
- [ ] Record short demo video
- [ ] Final accuracy evaluation report
- [ ] Clean up code, resolve TODOs, merge to main

**Milestone:** MVP complete and documented. Ready for real-world pilot deployment.

---

## 10. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ESP32 CSI API inconsistency across firmware versions | Medium | High | Pin esp-idf version in firmware project; use RuView's firmware which is tested with specific esp-idf versions |
| Public dataset domain gap too large | High | Medium | Mandatory in-situ data collection; treat lab training as pre-training only |
| RPi5 cannot sustain real-time inference load | Medium | High | Profile early (Week 2); implement model quantization and batching; RuView's models are already optimized for edge |
| False positives too high in real home | Medium | High | Tune two-stage confirmation; add user feedback loop |
| WiFi interference in dense apartment buildings | Medium | Medium | Channel selection per zone; RSSI monitoring |
| 6-week timeline slips | High | Medium | Ruthlessly scope MVP; defer heart rate to post-MVP; leverage RuView imports to save dev time |
| ElderAL-CSI dataset unavailable/restricted | Low | Medium | Supplement with in-situ data collection; use data augmentation aggressively |
| Single developer burnout | Medium | High | Weekly milestones with go/no-go decisions; document as you build |

---

## 11. Budget Breakdown

| Item | Quantity | Unit Cost (VND) | Total (VND) |
|---|---|---|---|
| ESP32-S3 (LoLin S3 or DevKitC) | 4 | ~120,000 | ~480,000 |
| Raspberry Pi 5 (4GB) | 1 | ~1,400,000 | ~1,400,000 |
| MicroSD 64GB (SanDisk Endurance) | 1 | ~200,000 | ~200,000 |
| Jumper wires, USB cables, power adapters | — | — | ~150,000 |
| Electricity (8 weeks) | — | — | ~100,000 |
| Miscellaneous | — | — | ~150,000 |
| **Total** | | | **~2,480,000 VND** |

*Note: A used Mini PC (x86) may be substituted for RPi5 at similar cost with better inference performance.*

---

## 12. Expected Outputs

### MVP Deliverables

1. **Working system** with 3 ESP32 nodes, Raspberry Pi server, running Docker Compose
2. **Fall detection model** fine-tuned to > 85% F1 on in-situ test set
3. **Web dashboard** with real-time status, vital signs, sleep score, alert log (Vietnamese)
4. **Telegram bot** with three-level alerting and daily morning report (VN messages)
5. **Trained model checkpoints** saved in `models/*/checkpoints/`
6. **Installation guide** (`docs/installation.md`) with step-by-step setup
7. **User guide** (`docs/user_guide.md`) for non-technical caregivers
8. **System design document** (`docs/architecture.md`)
9. **Accuracy evaluation report** (`docs/evaluation_report.md`)
10. **Clean GitHub repository** on main branch with README

### Quality Gates (must pass before MVP sign-off)

- [ ] Fall detection F1 > 85% on in-situ test set (at least 50 fall events)
- [ ] False positive rate < 2 per day in 48-hour real-environment test
- [ ] End-to-end fall -> Telegram latency < 5 seconds (measured 10 times, all pass)
- [ ] Dashboard loads on mobile browser in < 2 seconds
- [ ] 24-hour continuous operation without crash or manual restart
- [ ] All unit tests passing (`pytest tests/ -v`)
- [ ] No hardcoded secrets in repository (verified by git log scan)

---

## 13. Future Roadmap

### Post-MVP (Months 3-6)

- **Heart rate estimation improvement:** Investigate cross-ESP32 phase difference techniques for better SNR
- **Multi-person support:** Extend models to handle 2-person scenarios (spouse monitoring)
- **Home Assistant integration:** Expose ElderCare as HA custom component with MQTT discovery
- **Zalo notification support:** Add Zalo OA webhook as alternative to Telegram
- **Edge model on ESP32-S3:** Port lightweight fall detection to run partially on-device for sub-1-second preliminary detection
- **Mobile app:** React Native caregiver app with push notifications and live status

### Long-Term (6-12 months)

- **Transfer learning across homes:** Federated fine-tuning from multiple deployments (privacy-preserving)
- **Anomaly detection:** Unsupervised detection of unusual behavioral patterns (e.g., bathroom visit frequency changes as early health indicator)
- **Doctor report export:** Structured sleep and activity report for physician review
- **Community dataset contribution:** Anonymized, consented CSI data contributed to public eldercare CSI benchmark

---

*This document is a living specification. Update it as the project evolves. Major architectural decisions must be reflected here within 48 hours of implementation.*
