"""
ESP32-S3 CSI Capture Firmware Specification

The firmware is in firmware/esp32-csi-node/ — a full esp-idf v5.2 project
forked from RuView (version v0.6.5). It captures CSI at ~20 Hz and streams
raw I/Q data over UDP using the ADR-018 binary protocol.

For testing without hardware, use csi_simulator.py to generate synthetic CSI data.

Transport path: ESP32 UDP → udp_mqtt_bridge.py → MQTT → ingestion/receiver.py
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
    csi_amplitude: list[float]   # 52 subcarriers (downsampled from 64)
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


FIRMWARE_BUILD_DOC = """
ESP32-S3 Firmware (RuView esp32-csi-node v0.6.5)

Located at: firmware/esp32-csi-node/

Build (Docker required):
  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd)/firmware/esp32-csi-node:/project" -w /project \
    espressif/idf:v5.2 bash -c \
    "rm -rf build sdkconfig && idf.py set-target esp32s3 && idf.py build"

Flash:
  python -m esptool --chip esp32s3 --port COM7 --baud 460800 \
    write_flash --flash_mode dio --flash_size 8MB \
    0x0     firmware/esp32-csi-node/build/bootloader/bootloader.bin \
    0x8000  firmware/esp32-csi-node/build/partition_table/partition-table.bin \
    0xf000  firmware/esp32-csi-node/build/ota_data_initial.bin \
    0x20000 firmware/esp32-csi-node/build/esp32-csi-node.bin

Provision WiFi:
  python firmware/esp32-csi-node/provision.py --port COM7 \
    --ssid "YourSSID" --password "YourPass" --target-ip <SERVER_IP>

Run the bridge to convert UDP → MQTT:
  python -m ingestion.udp_mqtt_bridge --config configs/zones.yaml

Node IDs are set via NVS on the ESP32 and must match configs/zones.yaml.
"""
