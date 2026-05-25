# 🏥 ElderCare — Hệ thống Giám sát Người cao tuổi qua WiFi

<p align="center">
  <em>Phát hiện té ngã · Theo dõi nhịp thở · Giám sát giấc ngủ · Cảnh báo khẩn cấp</em><br>
  <em>Không camera · Không thiết bị đeo · 100% nội bộ · Chi phí ~2.5 triệu VND</em>
</p>

---

## Tổng quan

**ElderCare** là hệ thống giám sát người cao tuổi sử dụng tín hiệu **WiFi CSI (Channel State Information)** — những biến động trong sóng WiFi do chuyển động cơ thể con người gây ra — để phát hiện té ngã, ước tính dấu hiệu sinh tồn (nhịp thở, nhịp tim), và theo dõi mô hình hoạt động mà không cần camera hay thiết bị đeo.

Hệ thống được phát triển dựa trên mã nguồn mở [RuView](https://github.com/ruvnet/RuView) và tùy chỉnh cho nhu cầu chăm sóc người cao tuổi tại hộ gia đình Việt Nam.

### Tính năng chính

| Tính năng | Mô tả |
|---|---|
| 🚨 **Phát hiện té ngã** | CNN + BiLSTM, xác nhận 2 giai đoạn, F1 > 85% |
| 🫁 **Nhịp thở** | FFT trên biến thiên pha, 6–30 nhịp/phút |
| 😴 **Chất lượng giấc ngủ** | Phân loại awake/light/deep + Điểm ngủ (0–100) |
| ⚠️ **Cảnh báo** | 3 mức: INFO / WARNING / EMERGENCY qua Telegram |
| 📊 **Dashboard** | Giao diện web real-time, tương thích mobile |
| 🏠 **Đa vùng** | Hỗ trợ 3 vùng (phòng ngủ, phòng khách, hành lang) |

### Khác biệt chính

| | ElderCare | Camera | Thiết bị đeo |
|---|---|---|---|
| Bảo vệ riêng tư | ✅ | ❌ | ✅ |
| Không cần đeo thiết bị | ✅ | ✅ | ❌ |
| Hoạt động trong bóng tối | ✅ | ❌ (cần IR) | ✅ |
| Chi phí thấp (~2.5tr VND) | ✅ | ❌ | ❌ |
| 100% nội bộ (không cloud) | ✅ | Tùy | Tùy |
| Phát hiện té ngã | ✅ | ✅ | ✅ |
| Dấu hiệu sinh tồn | ✅ (nhịp thở) | ❌ | ✅ |
| Giám sát giấc ngủ | ✅ | ❌ | ✅ |

---

## Kiến trúc Hệ thống

```
ESP32-S3 (thu CSI 50Hz)
    │  UDP/MQTT
    ▼
Ingestion Layer (MQTT receiver, ring buffer)
    │
    ▼
Preprocessing (Hampel → Bandpass → Phase sanitization → Z-score)
    │
    ▼
Inference Engine (4 model, multiprocessing)
    ├── FallDetector (CNN → BiLSTM → FC)
    ├── VitalSignsEstimator (FFT respiration + HR)
    ├── SleepMonitor (LSTM → sleep score)
    └── ActivityTracker (rule-based)
    │
    ▼
Alert Manager ──► Telegram / Log / InfluxDB
    │
    ▼
Dashboard (FastAPI + React) ──► Browser UI
```

---

## Phần cứng

| Thành phần | Số lượng | Chi phí (VND) |
|---|---|---|
| ESP32-S3 (LoLin S3 hoặc DevKitC) | 4 | ~480,000 |
| Raspberry Pi 5 (4GB) hoặc Mini PC | 1 | ~1,400,000 |
| MicroSD 64GB | 1 | ~200,000 |
| Dây cáp, nguồn, phụ kiện | — | ~150,000 |
| **Tổng** | | **~2,480,000** |

---

## Cài đặt Nhanh

```bash
# Clone repository
git clone https://github.com/dungnotnull/WiFi-Sensing-Based-Elderly-Care-Monitoring-System.git
cd WiFi-Sensing-Based-Elderly-Care-Monitoring-System

# Cài đặt dependencies
pip install -r requirements.txt

# Sao chép file cấu hình
cp .env.template .env
# Sửa .env với Telegram Bot Token và Chat ID

# Chạy full stack với Docker
docker-compose -f docker/docker-compose.yml up --build

# Hoặc chạy inference engine standalone
python -m pipeline.inference_engine --config configs/zones.yaml

# Chạy dashboard
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8000
```

---

## Cấu trúc Dự án

```
eldercare/
├── firmware/esp32_csi/    # ESP32-S3 firmware spec + CSI simulator
├── ingestion/             # MQTT ingestion, ring buffers, validation
├── models/
│   ├── fall_detection/    # CSI-FallNet (CNN→BiLSTM→FC)
│   ├── vital_signs/       # FFT respiration + heart rate
│   ├── sleep/             # SleepLSTM + Sleep Score
│   └── activity/          # Rule-based activity/inactivity
├── pipeline/              # Preprocessing + inference engine
├── alerts/                # Alert manager (Telegram, VN localization)
├── dashboard/
│   ├── backend/           # FastAPI REST API
│   └── frontend/          # React UI
├── configs/               # YAML config (zones, thresholds, alerts, models)
├── training/              # Fine-tuning scripts + data annotator
├── tests/                 # Unit tests (30 tests, all passing)
├── docker/                # Dockerfiles + docker-compose
└── data/                  # CSI datasets + annotations (gitignored)
```

---

## Dataset

Dự án sử dụng dataset công khai miễn phí:

- **[CSI-Bench](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench)** — Dataset CSI quy mô lớn cho pre-training
- **ElderAL-CSI** — Dataset CSI đặc thù người cao tuổi (sẽ được cung cấp dưới dạng local files)
- **In-Situ Data** — Dữ liệu thu thập tại nhà (3–5 giờ mỗi phòng)

---

## Lộ trình Phát triển

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| **Phase 0** | Fork RuView, thiết lập baseline | ✅ Hoàn thành |
| **Phase 1** | Preprocessing + baseline model | 🔜 Tiếp theo |
| **Phase 2** | Fine-tuning + multi-zone | ⏳ Kế hoạch |
| **Phase 3** | Alerting + Dashboard | ⏳ Kế hoạch |
| **Phase 4** | Sleep monitoring + hardening | ⏳ Kế hoạch |
| **Phase 5** | Documentation + handoff | ⏳ Kế hoạch |

---

## Chạy Test

```bash
# Chạy toàn bộ test
pytest tests/ -v

# Chạy test với coverage
pytest tests/ --cov=pipeline --cov=models --cov-report=html
```

---

## Giấy phép

MIT License — xem [LICENSE](LICENSE).

Dự án được fork từ [RuView](https://github.com/ruvnet/RuView) (MIT License).

---

**ElderCare** — Giải pháp giám sát người cao tuổi bảo vệ riêng tư, chi phí thấp cho hộ gia đình Việt Nam.
