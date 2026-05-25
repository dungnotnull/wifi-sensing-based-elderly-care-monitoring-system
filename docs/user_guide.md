# ElderCare — User Guide

A guide for caregivers and family members.

Table of Contents:
1. [Overview](#1-overview)
2. [Dashboard](#2-dashboard)
3. [Telegram Alerts](#3-telegram-alerts)
4. [Understanding Metrics](#4-understanding-metrics)
5. [Daily Report](#5-daily-report)
6. [Responding to Alerts](#6-responding-to-alerts)
7. [FAQ](#7-faq)
8. [Advanced Features for Installers](#8-advanced-features-for-installers)

---

## 1. Overview

ElderCare is an elderly monitoring system that uses **no cameras and no wearables**. It uses WiFi signals to detect:
- **Falls**
- **Breathing rate**
- **Sleep quality**
- **Activity level**

All data is processed locally on the home server — **nothing is sent to the cloud**.

---

## 2. Dashboard

Access the dashboard at: `http://<server-address>:8000`

### 2.1 Home Page

The main screen shows:

| Item | Meaning |
|---|---|
| Zone name | Room being monitored |
| Status | Active / Still / Inactive |
| Respiration | Breaths per minute (normal: 12-20) |
| Heart rate | Beats per minute (experimental) |
| Fall detected | Yes / No |
| Sleep score | 0-100 (above 80 = excellent) |
| Vitals trend graph | 24-hour respiration chart showing breathing rate over time |
| Sleep quality chart | 30-day bar chart showing nightly sleep scores |
| Alert log | Recent alerts with severity badges (INFO / WARNING / EMERGENCY) |
| System health panel | CPU usage, memory usage, disk space, worker status per model |
| Daily summary widget | Key daily stats: sleep score, activity hours, alert count |

All dashboard labels are displayed in Vietnamese.

### 2.2 Pages

| URL | Content |
|---|---|
| `/api/zones` | Per-zone status |
| `/api/vitals?zone_id=zone_bedroom` | 1-hour respiration history |
| `/api/alerts` | Last 50 alerts |
| `/api/sleep?zone_id=zone_bedroom` | 30-day sleep scores |
| `/api/daily-summary` | Daily summary report |
| `/api/health` | System health status |
| `/api/events` | Real-time Server-Sent Events (SSE) stream for live dashboard updates |
| `/api/telemetry` | System performance data (CPU, memory, disk, worker status) |
| `/api/csi-quality/{zone_id}` | Signal quality metrics for a specific zone |
| `/api/shadow-mode/report` | Deployment testing report from shadow mode |

---

## 3. Telegram Alerts

### 3.1 Setup

After the system is installed, you will receive Telegram messages for important events. Your installer will configure the Telegram bot for you.

### 3.2 Alert Levels

| Level | Icon | Meaning | Example |
|---|---|---|---|
| INFO | info | Regular update | Morning report, system status |
| WARNING | warning | Needs attention | Inactivity over 2 hours, abnormal breathing |
| EMERGENCY | alert | Act immediately | Fall detected, no movement after fall |

### 3.3 Example Message

```
EMERGENCY
Zone: Bedroom
Time: 14:30:25 25/05/2026
Event: Fall detected! (confidence: 92%)
```

---

## 4. Understanding Metrics

### 4.1 Respiration

| Range | Assessment |
|---|---|
| 12-20 BPM | Normal |
| 8-11 or 21-25 | Slightly abnormal |
| < 8 or > 25 | Abnormal -- check immediately |

### 4.2 Sleep Score

| Score | Assessment |
|---|---|
| 80-100 | Excellent |
| 60-79 | Good |
| 40-59 | Average |
| < 40 | Poor -- needs improvement |

The sleep score is based on: deep sleep time, light sleep time, awake time, and breathing regularity.

### 4.3 Activity Level

| State | Meaning |
|---|---|
| Active | Moving around |
| Still | Sitting or lying still, breathing normally |
| Inactivity | Completely still -- check if over 2 hours |

> **Note:** Activity and inactivity thresholds automatically adapt during the first 48 hours of deployment. The system learns the normal baseline for your home environment and calibrates accordingly. You may notice threshold values adjusting during this period -- this is expected behavior.

---

## 5. Daily Report

Every morning at 7:00 AM, the system sends a summary via Telegram (if configured).

The report includes:
- **Last night**: sleep score, deep/light/awake breakdown
- **Current status**: activity state, respiration, heart rate
- **24-hour alerts**: counts by level and details
- **Daily advice**: health recommendations

---

## 6. Responding to Alerts

### EMERGENCY -- Fall Detected
1. **Call the monitored person immediately**
2. If no answer, **go check in person** or call a neighbor
3. If injured, **call emergency services**
4. After the event, have the person see a doctor (internal injuries may not be visible)

### WARNING -- Prolonged Inactivity
1. Call to check on them
2. If they are fine (reading, watching TV) -- no need to worry
3. If no answer after 3 calls -- go check in person

### INFO
- No urgent action needed
- Read the daily report to stay informed

---

## 7. FAQ

### Does the system use cameras?
**No.** ElderCare only reads WiFi signals. There are no cameras. The system cannot see images or video.

### Does the person need to wear a device?
**No.** The system requires no wearable devices at all.

### Is data sent to the cloud?
**No.** All data is processed on the local home server. There is no cloud connection.

### What happens if WiFi goes down?
The system stops working until WiFi is restored. The ESP32 boards will reconnect automatically.

### Do ESP32 boards need batteries?
ESP32 boards are USB-powered. Keep them plugged in continuously.

### What maintenance is required?
- Check that the ESP32 indicator lights are on (once per week)
- Verify the dashboard is accessible
- No software maintenance needed -- the system runs on its own

---

## 8. Advanced Features for Installers

This section covers features intended for technical users who set up and maintain ElderCare installations.

### 8.1 Shadow Mode

Shadow Mode allows you to test a first deployment safely without triggering real alerts. When enabled, the inference pipeline runs normally and logs all detections, but **no Telegram alerts are sent** and **no EMERGENCY actions are triggered**.

This is useful for:
- Verifying that CSI data flows correctly from ESP32 nodes
- Checking model confidence levels before going live
- Tuning node placement without causing false alarm fatigue

Enable Shadow Mode in `configs/thresholds.yaml`:
```yaml
shadow_mode:
  enabled: true
```

Access the shadow mode report at `/api/shadow-mode/report` to review what would have been alerted. Once satisfied, set `enabled: false` and restart the pipeline.

### 8.2 CSI Quality Indicators

The dashboard shows signal quality per zone so you can verify that ESP32 nodes are positioned correctly. Navigate to `/api/csi-quality/{zone_id}` to see metrics.

**Good signal indicators:**
- SNR (Signal-to-Noise Ratio) consistently above 15 dB
- Subcarrier amplitude variance is stable (no sudden drops)
- Packet loss below 5%

**Bad signal indicators:**
- SNR below 10 dB or fluctuating wildly
- Large gaps in CSI data (packet loss above 20%)
- Amplitude values stuck at zero or saturated

If signal quality is poor, try repositioning the ESP32 nodes (see Section 2.3 of the Installation Guide for placement tips).

### 8.3 Adaptive Thresholds

The system learns your home's baseline activity patterns during the first 48 hours. After this learning period, thresholds for inactivity detection and breathing rate alerts adjust automatically based on observed patterns.

Configuration in `configs/thresholds.yaml`:
```yaml
adaptive:
  enabled: true
  learning_period_hours: 48
  inactivity_multiplier: 1.5    # alert when 1.5x baseline inactivity
  breathing_low_offset: -4      # alert when BPM drops 4 below baseline
  breathing_high_offset: 4      # alert when BPM rises 4 above baseline
```

After the learning period completes, the system stores baseline values and uses them for all future comparisons. You can review and override learned thresholds via the dashboard.

### 8.4 Record/Replay

For debugging and tuning, you can save real CSI data to disk and replay it through the pipeline later. This is useful for:
- Debugging false positives by replaying the exact CSI sequence
- Testing model changes against previously captured data
- Sharing data with developers for remote troubleshooting

See the Installation Guide (Section 3.4) for record/replay commands.

### 8.5 Home Assistant Integration

ElderCare supports automatic MQTT discovery for Home Assistant. If you run Home Assistant on the same network, ElderCare entities (sensors for respiration, sleep score, activity state, and binary sensors for fall detection) appear automatically.

Enable in `configs/thresholds.yaml`:
```yaml
home_assistant:
  enabled: true
  discovery_prefix: "homeassistant"
```

ElderCare publishes MQTT discovery messages to `homeassistant/sensor/eldercare/...` and `homeassistant/binary_sensor/eldercare/...`. No manual configuration is needed in Home Assistant -- devices appear in the integrations panel within a few minutes.
