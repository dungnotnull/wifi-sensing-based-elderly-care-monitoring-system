# ElderCare Installation Guide

Step-by-step setup for deploying ElderCare on a Raspberry Pi 5 (or Mini PC) with ESP32-S3 nodes.

---

## Prerequisites

### Hardware
| Item | Qty | Notes |
|---|---|---|
| Raspberry Pi 5 (4GB+) or Mini PC | 1 | Runs all server software |
| ESP32-S3 boards | 3-4 | CSI capture nodes (per zone) |
| MicroSD 64GB | 1 | For RPi5 OS |
| 2.4 GHz WiFi Router | 1 | Existing home router works |

### Software
- Ubuntu Server 24.04 LTS or Raspberry Pi OS Lite (64-bit)
- Python 3.9+
- Docker + Docker Compose
- Git

---

## 1. Server Setup

### 1.1 Install system dependencies

```bash
sudo apt update && sudo apt install -y \
  python3 python3-pip python3-venv \
  git docker.io docker-compose-v2 \
  mosquitto mosquitto-clients
```

### 1.2 Clone the repository

```bash
git clone https://github.com/dungnotnull/WiFi-Sensing-Based-Elderly-Care-Monitoring-System.git
cd WiFi-Sensing-Based-Elderly-Care-Monitoring-System
```

### 1.3 Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing
```

### 1.4 Configure environment variables

```bash
cp .env.template .env
nano .env
```

Fill in at minimum:
```
MQTT_BROKER=localhost
MQTT_PORT=1883
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID_PRIMARY=your_chat_id
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
INFERENCE_DEVICE=cpu
```

### 1.5 Configure zones

Edit `configs/zones.yaml` to match your home layout. Each ESP32 needs a unique MAC address and zone ID:

```yaml
zones:
  - zone_id: "zone_bedroom"
    name: "Phòng ngủ"
    mac_address: "AA:BB:CC:DD:EE:01"
    mqtt_topic: "eldercare/csi/zone_bedroom"
    active: true
```

Get the actual MAC addresses from your ESP32 boards after flashing firmware.

### 1.6 Verify installation

```bash
python -m pytest tests/ -v
```

All tests should pass. You should see `51 passed`.

---

## 2. ESP32-S3 Firmware

### 2.1 Prerequisites
- ESP-IDF v5.x installed on your development machine
- USB-C cable for each ESP32-S3 board
- ESP32-S3 boards (LoLin S3 or DevKitC-1 recommended)

### 2.2 Flash firmware

The firmware specification lives at `firmware/esp32_csi/`. It captures CSI at 50 Hz (HT20 mode, 52 subcarriers) and transmits via MQTT.

**Refer to the firmware README** at `firmware/esp32_csi/` for exact flashing commands and esp-idf version pinning.

Basic flashing:
```bash
cd firmware/esp32_csi
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### 2.3 Node placement
- Mount at 1.0-1.5m height (chest level)
- Line-of-sight or one-wall clearance preferred
- Avoid behind large metal objects or appliances
- Minimum 3m between nodes
- Connect to the same WiFi network as the server

---

## 3. Quick Start (Local)

### 3.1 Start MQTT broker

```bash
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

### 3.2 Run inference engine

```bash
python -m pipeline.inference_engine --config configs/zones.yaml
```

The engine starts 12 workers (3 zones x 4 models) and waits for CSI data via MQTT.

### 3.3 Run dashboard

```bash
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://<server-ip>:8000` in a browser.

### 3.4 Simulate CSI data (no hardware)

For testing without ESP32 nodes:

```bash
# Generate mock dataset (fall + sleep)
python -m training.mock_data

# Train models on mock data
python -m training.train_fall_detection --dataset-type mock --epochs 10
python -m training.train_sleep --epochs 10

# Run evaluation
python -m pipeline.evaluate
```

---

## 4. Docker Deployment

### 4.1 Build and start

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This starts:
- `mosquitto` — MQTT broker (port 1883)
- `eldercare-server` — FastAPI dashboard + inference (port 8000)

### 4.2 Verify

```bash
curl http://localhost:8000/api/health
# {"status": "healthy", "version": "0.1.0", ...}
```

---

## 5. Dataset Setup

### 5.1 CSI-Bench (public dataset)

1. Download from [Kaggle — CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) (free account required)
2. Place preprocessed `.npz` files in `data/csibench/`
3. Train:
```bash
python -m training.train_fall_detection --dataset-type csibench --dataset data/csibench/train.npz --epochs 50
```

### 5.2 ElderAL-CSI (elderly-specific dataset)

1. Place CSV files in `data/raw/elderal-csi/` maintaining the directory structure:
   ```
   data/raw/elderal-csi/
     action1_walk/user1_position1_activity1/*.csv
     action2_fall_new/user2_position6_activity2/*.csv
     ...
   ```
2. Train:
```bash
python -m training.train_fall_detection --dataset-type elderal --dataset data/raw/elderal-csi/ --epochs 50
```

### 5.3 In-Situ Data

After deploying hardware, collect your own data:

```bash
python training/annotator.py --output data/annotations/
```

Collect 3-5 hours per zone, then fine-tune:
```bash
python -m training.train_fall_detection \
  --dataset data/processed/in-situ/ \
  --epochs 20 \
  --output models/fall_detection/checkpoints/
```

---

## 6. Telegram Alert Setup

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Add to `.env`:
```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234gh
TELEGRAM_CHAT_ID_PRIMARY=123456789
```
4. Test:
```bash
python -m alerts.alert_manager
```

---

## 7. Troubleshooting

### MQTT connection fails
```bash
sudo systemctl status mosquitto
mosquitto_sub -t "eldercare/csi/#" -v  # verify topics
```

### Workers fail to start
```bash
# Check model checkpoints exist
ls models/fall_detection/checkpoints/
ls models/sleep/checkpoints/

# Set INFERENCE_DEVICE=cpu in .env if no GPU available
```

### Dashboard not loading
```bash
curl http://localhost:8000/api/health
# If connection refused, check uvicorn is running and port is free
```

### ESP32 not sending data
- Verify WiFi credentials in firmware config
- Check MQTT broker IP is reachable from ESP32
- Monitor MQTT traffic: `mosquitto_sub -t "#" -v`

---

## 8. Post-Install Checklist

- [ ] All 51 tests pass (`pytest tests/ -v`)
- [ ] Dashboard loads at `http://<server-ip>:8000`
- [ ] MQTT broker running (`systemctl status mosquitto`)
- [ ] ESP32 nodes connected and publishing CSI
- [ ] Telegram test alert received
- [ ] `.env` configured with real tokens
- [ ] Model checkpoints at expected paths
