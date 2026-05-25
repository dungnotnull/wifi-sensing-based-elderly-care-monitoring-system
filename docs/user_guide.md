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
| 🛏️ Zone name | Room being monitored |
| Status | Active / Still / Inactive |
| Respiration | Breaths per minute (normal: 12–20) |
| Heart rate | Beats per minute (experimental) |
| Fall detected | Yes / No |
| Sleep score | 0–100 (above 80 = excellent) |

### 2.2 Pages

| URL | Content |
|---|---|
| `/api/zones` | Per-zone status |
| `/api/vitals?zone_id=zone_bedroom` | 1-hour respiration history |
| `/api/alerts` | Last 50 alerts |
| `/api/sleep?zone_id=zone_bedroom` | 30-day sleep scores |
| `/api/daily-summary` | Daily summary report |
| `/api/health` | System health status |

---

## 3. Telegram Alerts

### 3.1 Setup

After the system is installed, you will receive Telegram messages for important events. Your installer will configure the Telegram bot for you.

### 3.2 Alert Levels

| Level | Icon | Meaning | Example |
|---|---|---|---|
| INFO | ℹ️ | Regular update | Morning report, system status |
| WARNING | ⚠️ | Needs attention | Inactivity over 2 hours, abnormal breathing |
| EMERGENCY | 🚨 | Act immediately | Fall detected, no movement after fall |

### 3.3 Example Message

```
🚨 EMERGENCY
📍 Zone: Bedroom
⏰ Time: 14:30:25 25/05/2026
📋 Event: Fall detected! (confidence: 92%)
```

---

## 4. Understanding Metrics

### 4.1 Respiration

| Range | Assessment |
|---|---|
| 12–20 BPM | ✅ Normal |
| 8–11 or 21–25 | ⚠️ Slightly abnormal |
| < 8 or > 25 | 🚨 Abnormal — check immediately |

### 4.2 Sleep Score

| Score | Assessment |
|---|---|
| 80–100 | ✅ Excellent |
| 60–79 | ✅ Good |
| 40–59 | ⚠️ Average |
| < 40 | ❌ Poor — needs improvement |

The sleep score is based on: deep sleep time, light sleep time, awake time, and breathing regularity.

### 4.3 Activity Level

| State | Meaning |
|---|---|
| Active | Moving around |
| Still | Sitting or lying still, breathing normally |
| Inactivity | Completely still — check if over 2 hours |

---

## 5. Daily Report

Every morning at 7:00 AM, the system sends a summary via Telegram (if configured).

The report includes:
- 🌙 **Last night**: sleep score, deep/light/awake breakdown
- 📋 **Current status**: activity state, respiration, heart rate
- 🔔 **24-hour alerts**: counts by level and details
- 💡 **Daily advice**: health recommendations

---

## 6. Responding to Alerts

### 🚨 EMERGENCY — Fall Detected
1. **Call the monitored person immediately**
2. If no answer, **go check in person** or call a neighbor
3. If injured, **call emergency services**
4. After the event, have the person see a doctor (internal injuries may not be visible)

### ⚠️ WARNING — Prolonged Inactivity
1. Call to check on them
2. If they are fine (reading, watching TV) — no need to worry
3. If no answer after 3 calls — go check in person

### ℹ️ INFO
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
- No software maintenance needed — the system runs on its own
