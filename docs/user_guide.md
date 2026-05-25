# ElderCare — Hướng Dẫn Sử Dụng / User Guide

Hướng dẫn dành cho người chăm sóc và gia đình.
Guide for caregivers and family members.

Mục lục / Table of Contents:
1. [Tổng quan / Overview](#1-tổng-quan--overview)
2. [Bảng điều khiển / Dashboard](#2-bảng-điều-khiển--dashboard)
3. [Cảnh báo Telegram / Telegram Alerts](#3-cảnh-báo-telegram--telegram-alerts)
4. [Hiểu các chỉ số / Understanding Metrics](#4-hiểu-các-chỉ-số--understanding-metrics)
5. [Báo cáo hàng ngày / Daily Report](#5-báo-cáo-hàng-ngày--daily-report)
6. [Xử lý khi có cảnh báo / Responding to Alerts](#6-xử-lý-khi-có-cảnh-báo--responding-to-alerts)
7. [Câu hỏi thường gặp / FAQ](#7-câu-hỏi-thường-gặp--faq)

---

## 1. Tổng quan / Overview

ElderCare là hệ thống giám sát người cao tuổi **không dùng camera, không cần thiết bị đeo**. Hệ thống sử dụng sóng WiFi để phát hiện:
- **Té ngã** (fall detection)
- **Nhịp thở** (respiration rate)
- **Chất lượng giấc ngủ** (sleep quality)
- **Mức độ hoạt động** (activity level)

Tất cả dữ liệu được xử lý tại nhà, **không gửi lên mạng (cloud)**.

ElderCare is a monitoring system for elderly people that uses **no cameras and no wearables**. It uses WiFi signals to detect:
- **Falls**
- **Breathing rate**
- **Sleep quality**
- **Activity level**

All data is processed locally — **nothing is sent to the cloud**.

---

## 2. Bảng điều khiển / Dashboard

Truy cập / Access: `http://<địa-chỉ-máy-chủ>:8000`

### 2.1 Trang chủ / Home Page

Màn hình chính hiển thị:

| Mục / Item | Ý nghĩa / Meaning |
|---|---|
| 🛏️ Tên khu vực / Zone name | Phòng đang được giám sát / Room being monitored |
| Trạng thái / Status | Hoạt động / Active — Nghỉ ngơi / Still — Không hoạt động / Inactive |
| Nhịp thở / Respiration | Số nhịp thở mỗi phút (bình thường: 12-20) |
| Nhịp tim / Heart rate | Nhịp tim mỗi phút (thử nghiệm) |
| Phát hiện té ngã / Fall detected | Có/Không |
| Điểm giấc ngủ / Sleep score | 0-100 (trên 80 = rất tốt) |

### 2.2 Các trang / Pages

| Đường dẫn / URL | Nội dung / Content |
|---|---|
| `/api/zones` | Trạng thái từng khu vực / Per-zone status |
| `/api/vitals?zone_id=zone_bedroom` | Lịch sử nhịp thở 1 giờ / 1-hour respiration history |
| `/api/alerts` | Nhật ký cảnh báo 50 mục gần nhất / Last 50 alerts |
| `/api/sleep?zone_id=zone_bedroom` | Điểm giấc ngủ 30 ngày / 30-day sleep scores |
| `/api/daily-summary` | Báo cáo hàng ngày / Daily summary |
| `/api/health` | Trạng thái hệ thống / System health |

---

## 3. Cảnh báo Telegram / Telegram Alerts

### 3.1 Cài đặt Telegram / Setup

Sau khi hệ thống được cài đặt, bạn sẽ nhận tin nhắn Telegram khi có sự kiện quan trọng.

After setup, you will receive Telegram messages for important events.

### 3.2 Các mức cảnh báo / Alert Levels

| Mức / Level | Biểu tượng / Icon | Ý nghĩa / Meaning | Ví dụ / Example |
|---|---|---|---|
| ℹ️ THÔNG BÁO / INFO | ℹ️ | Cập nhật thường xuyên / Regular update | Báo cáo buổi sáng, trạng thái hệ thống |
| ⚠️ CẢNH BÁO / WARNING | ⚠️ | Bất thường cần chú ý / Needs attention | Không hoạt động > 2 giờ, nhịp thở bất thường |
| 🚨 KHẨN CẤP / EMERGENCY | 🚨 | Nguy hiểm — hành động ngay / Act immediately | Phát hiện té ngã, không cử động sau té |

### 3.3 Ví dụ tin nhắn / Example Message

```
🚨 KHẨN CẤP
📍 Khu vực: Phòng ngủ
⏰ Thời gian: 14:30:25 25/05/2026
📋 Sự kiện: Phát hiện té ngã! (độ tin cậy: 92%)
```

---

## 4. Hiểu các chỉ số / Understanding Metrics

### 4.1 Nhịp thở / Respiration

| Phạm vi / Range | Đánh giá / Assessment |
|---|---|
| 12-20 nhịp/phút / BPM | ✅ Bình thường / Normal |
| 8-11 hoặc 21-25 | ⚠️ Hơi bất thường / Slightly abnormal |
| < 8 hoặc > 25 | 🚨 Bất thường — kiểm tra ngay / Check immediately |

### 4.2 Điểm giấc ngủ / Sleep Score

| Điểm / Score | Đánh giá / Assessment |
|---|---|
| 80-100 | ✅ Rất tốt / Excellent |
| 60-79 | ✅ Tốt / Good |
| 40-59 | ⚠️ Trung bình / Average |
| < 40 | ❌ Kém — cần cải thiện / Needs improvement |

Điểm giấc ngủ dựa trên: thời gian ngủ sâu, thời gian ngủ nông, thời gian thức, và sự ổn định nhịp thở.

The sleep score considers: deep sleep time, light sleep time, awake time, and breathing regularity.

### 4.3 Mức độ hoạt động / Activity Level

| Trạng thái / State | Ý nghĩa / Meaning |
|---|---|
| Hoạt động / Active | Đang di chuyển / Moving around |
| Nghỉ ngơi / Still | Ngồi hoặc nằm yên, có thở / Sitting or lying still, breathing |
| Không hoạt động / Inactivity | Đứng yên hoàn toàn — kiểm tra nếu > 2 giờ / Completely still — check if > 2 hours |

---

## 5. Báo cáo hàng ngày / Daily Report

Mỗi sáng lúc 7:00, hệ thống gửi báo cáo qua Telegram (nếu đã cấu hình).

Every morning at 7:00 AM, the system sends a summary via Telegram (if configured).

Báo cáo bao gồm / Report includes:
- 🌙 **Đêm qua**: điểm giấc ngủ, tỉ lệ ngủ sâu/nông/thức
- 📋 **Trạng thái hiện tại**: hoạt động, nhịp thở, nhịp tim
- 🔔 **Cảnh báo 24 giờ**: số lượng và chi tiết
- 💡 **Khuyến nghị**: lời khuyên sức khỏe hàng ngày

---

## 6. Xử lý khi có cảnh báo / Responding to Alerts

### 🚨 KHẨN CẤP (EMERGENCY) — Té ngã
1. **Gọi điện ngay** cho người được giám sát
2. Nếu không trả lời, **đến kiểm tra trực tiếp** hoặc gọi hàng xóm
3. Nếu có chấn thương, **gọi cấp cứu 115**
4. Sau sự kiện, đưa người cao tuổi đi khám bác sĩ (có thể có chấn thương bên trong)

### ⚠️ CẢNH BÁO (WARNING) — Không hoạt động
1. Gọi điện kiểm tra
2. Nếu người cao tuổi vẫn bình thường (đang đọc sách, xem TV) — không cần lo
3. Nếu không trả lời sau 3 lần gọi — đến kiểm tra

### ℹ️ THÔNG BÁO (INFO)
- Không cần hành động khẩn cấp
- Đọc báo cáo hàng ngày để nắm tình hình

---

## 7. Câu hỏi thường gặp / FAQ

### Hệ thống có xem được hình ảnh không?
**Không.** ElderCare chỉ đọc sóng WiFi, không có camera. Hệ thống không thể nhìn thấy hình ảnh hay video.

**No.** ElderCare only reads WiFi signals. There are no cameras. The system cannot see images or video.

### Người cao tuổi có cần đeo thiết bị gì không?
**Không.** Hệ thống hoạt động hoàn toàn không cần thiết bị đeo.

**No.** The system requires no wearable devices at all.

### Dữ liệu có bị gửi lên mạng không?
**Không.** Tất cả dữ liệu được xử lý trên máy chủ tại nhà. Không có kết nối đám mây.

**No.** All data is processed on the local home server. There is no cloud connection.

### Nếu WiFi bị mất thì sao?
Hệ thống sẽ không hoạt động cho đến khi WiFi được khôi phục. ESP32 sẽ tự động kết nối lại.

The system will stop working until WiFi is restored. The ESP32 boards will reconnect automatically.

### Có cần thay pin cho ESP32 không?
ESP32 được cấp nguồn qua USB. Cắm vào nguồn điện liên tục.

ESP32 boards are USB-powered. Keep them plugged in continuously.

### Tôi cần bảo trì gì?
- Kiểm tra đèn ESP32 còn sáng (1 lần/tuần)
- Kiểm tra dashboard còn truy cập được
- Không cần bảo trì phần mềm — hệ thống tự chạy
