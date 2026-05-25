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
    name: "Phong ngu"
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

For recording and replaying real CSI data (after ESP32 nodes are active):

```bash
# Record real CSI data for offline analysis
python -c "from pipeline.record_replay import CSIRecorder; \
  recorder = CSIRecorder(output_dir='data/recorded/'); \
  recorder.start(zone_id='zone_bedroom', duration_seconds=3600)"

# Replay recorded data through pipeline
python -c "from pipeline.record_replay import CSIReplayer; \
  replayer = CSIReplayer(recording_dir='data/recorded/zone_bedroom/'); \
  replayer.play(speed=1.0)"
```

---

## 4. Docker Deployment

### 4.1 Build and start

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This starts:
- `mosquitto` -- MQTT broker (port 1883)
- `eldercare-server` -- FastAPI dashboard + inference (port 8000)

### 4.2 Verify

```bash
curl http://localhost:8000/api/health
# {"status": "healthy", "version": "0.2.0", ...}
```

---

## 5. Dataset Setup

### 5.1 CSI-Bench (public dataset)

1. Download from [Kaggle -- CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) (free account required)
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
- [ ] SQLite database created at `data/eldercare.db`
- [ ] Shadow mode configured for first deployment (see Section 9.1)
- [ ] Locale files present in `configs/locales/`
- [ ] CSI quality monitoring active (check `/api/csi-quality/zone_bedroom`)

---

## 9. Advanced Configuration

### 9.1 Shadow Mode

Shadow Mode runs the full inference pipeline without sending real alerts. Use this during first deployment to verify everything works before going live.

Edit `configs/thresholds.yaml`:

```yaml
shadow_mode:
  enabled: true           # set to false when ready for production
  log_detections: true    # write detections to log file
  report_retention_days: 7
```

While shadow mode is active, all detections are logged but no Telegram messages are sent. Review what would have been alerted:

```bash
curl http://localhost:8000/api/shadow-mode/report
```

Once you are satisfied with the detection accuracy and false positive rate, set `shadow_mode.enabled: false` and restart the pipeline.

### 9.2 Adaptive Thresholds

The system can learn baseline activity and vitals patterns from the first 48 hours of deployment data, then auto-adjust alert thresholds accordingly.

Edit `configs/thresholds.yaml`:

```yaml
adaptive:
  enabled: true
  learning_period_hours: 48
  inactivity_multiplier: 1.5    # alert when 1.5x baseline inactivity
  breathing_low_offset: -4      # alert when BPM drops 4 below baseline
  breathing_high_offset: 4      # alert when BPM rises 4 above baseline
  min_breathing_low: 8          # never alert below this BPM regardless
  max_breathing_high: 25        # never alert above this BPM regardless
```

After the learning period, the system stores baseline values in `data/eldercare.db`. You can review or override learned values via the dashboard.

### 9.3 InfluxDB Setup (Optional)

For long-term historical data storage and trend analysis, you can configure InfluxDB alongside the default SQLite database.

1. Install InfluxDB:
```bash
sudo apt install influxdb2
sudo systemctl enable influxdb2
sudo systemctl start influxdb2
```

2. Add to `.env`:
```
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-influxdb-token
INFLUXDB_ORG=eldercare
INFLUXDB_BUCKET=vitals
```

3. The pipeline automatically writes time-series vitals data to InfluxDB when these variables are set. Historical charts on the dashboard will use InfluxDB when available, falling back to SQLite otherwise.

### 9.4 Home Assistant Integration

ElderCare publishes MQTT auto-discovery messages compatible with Home Assistant. Sensors and binary sensors for each zone appear automatically.

Edit `configs/thresholds.yaml`:

```yaml
home_assistant:
  enabled: true
  discovery_prefix: "homeassistant"    # MQTT discovery topic prefix
  update_interval_seconds: 30         # how often to publish state updates
```

Exposed entities per zone:
- `sensor.eldercare_{zone}_respiration` -- breathing rate (BPM)
- `sensor.eldercare_{zone}_sleep_score` -- nightly sleep score
- `sensor.eldercare_{zone}_activity` -- activity state (active/still/inactive)
- `binary_sensor.eldercare_{zone}_fall` -- fall detection alert
- `sensor.eldercare_{zone}_heart_rate` -- heart rate (experimental)

No additional configuration is needed in Home Assistant. Entities appear under Devices in the MQTT integration within a few minutes.

### 9.5 Model Quantization

For deployment on resource-constrained devices (Raspberry Pi 5), you can quantize models to INT8 to reduce inference time and memory usage.

```bash
python -c "from pipeline.quantization import quantize_from_checkpoint; quantize_from_checkpoint()"
```

This reads model checkpoints from `models/*/checkpoints/`, applies dynamic INT8 quantization, and saves quantized versions alongside the originals. The inference engine automatically loads quantized models when available.

Quantization typically reduces model size by ~4x and improves inference speed by 2-3x on CPU, with minimal accuracy loss.

### 9.6 Locale Configuration

Alert messages and dashboard labels support multiple languages. The default is Vietnamese (`vi`).

Edit `configs/alerts.yaml`:

```yaml
language: vi    # "vi" for Vietnamese, "en" for English

templates:
  vi:
    emergency_title: "KHAN CAP"
    warning_title: "CANH BAO"
    info_title: "THONG TIN"
  en:
    emergency_title: "EMERGENCY"
    warning_title: "WARNING"
    info_title: "INFO"
```

Locale files for dashboard labels are stored in `configs/locales/`. Each locale is a YAML file (e.g., `vi.yaml`, `en.yaml`). To add a new language, create a new file in `configs/locales/` following the same structure.

### 9.7 CSI Recording (for Debugging)

You can record raw CSI data to disk for offline analysis, model debugging, or sharing with developers.

```bash
# Record CSI data from a specific zone
python -c "from pipeline.record_replay import CSIRecorder; \
  recorder = CSIRecorder(output_dir='data/recorded/'); \
  recorder.start(zone_id='zone_bedroom', duration_seconds=3600)"

# Replay recorded data through the pipeline
python -c "from pipeline.record_replay import CSIReplayer; \
  replayer = CSIReplayer(recording_dir='data/recorded/zone_bedroom/'); \
  replayer.play(speed=1.0)"

# List available recordings
python -c "from pipeline.record_replay import CSIRecorder; \
  recorder = CSIRecorder(output_dir='data/recorded/'); \
  print(recorder.list_recordings())"
```

Recordings are stored as compressed numpy arrays in `data/recorded/{zone_id}/`. Each recording includes timestamps, raw CSI matrices, and metadata about the capture session.
