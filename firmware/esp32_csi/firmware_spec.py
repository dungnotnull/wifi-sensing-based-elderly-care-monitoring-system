"""
ESP32-S3 CSI Capture Firmware — Placeholder

This file documents the expected firmware behavior. The actual firmware is
written in C (ESP-IDF CSI API) and flashed to the ESP32-S3.

Expected behavior:
  - Samples WiFi CSI at ~50 Hz (every 20ms)
  - Extracts CSI amplitude and phase per subcarrier (52 subcarriers for HT20)
  - Packages data: {zone_id, timestamp, sequence_number, csi_amplitude: [52], csi_phase: [52], rssi}
  - Publishes to MQTT topic: eldercare/csi/{zone_id}
  - Auto-reconnects on WiFi / MQTT disconnect
  - Pins esp-idf v5.x for CSI API stability

For testing without hardware, use csi_simulator.py to generate synthetic CSI data.
"""

import json
import struct
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CSIPacket:
    zone_id: str
    timestamp: float
    sequence_number: int
    csi_amplitude: list[float]   # 52 subcarriers
    csi_phase: list[float]       # 52 subcarriers
    rssi: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_mqtt_payload(self) -> bytes:
        """Binary packed format for MQTT transmission (more compact than JSON)."""
        packed = struct.pack(
            f"=dfi{'f'*52}{'f'*52}f",
            self.timestamp,
            float(self.sequence_number),
            self.sequence_number,
            *self.csi_amplitude,
            *self.csi_phase,
            self.rssi,
        )
        return packed

    @classmethod
    def from_json(cls, data: str) -> "CSIPacket":
        d = json.loads(data)
        return cls(**d)


# Firmware build instructions placeholder:
FIRMWARE_BUILD_DOC = """
ESP32-S3 Firmware Build Instructions:
1. Install ESP-IDF v5.x: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/
2. Configure WiFi and MQTT credentials in sdkconfig
3. Build: idf.py build
4. Flash: idf.py -p /dev/ttyUSB0 flash
5. Monitor output: idf.py monitor

CSI configuration:
- HT20 mode (20 MHz bandwidth) → 52 usable subcarriers
- Sample rate: 50 Hz (20ms interval)
- CSI reporting: amplitude + phase per subcarrier
"""
