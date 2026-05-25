"""
CSI Data Ingestion Layer

Subscribes to MQTT topics for each configured zone, validates incoming
CSI packets, and maintains per-zone ring buffers for downstream processing.
"""

import json
import logging
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


@dataclass
class ZoneConfig:
    zone_id: str
    name: str
    mac_address: str
    mqtt_topic: str
    active: bool


@dataclass
class CSIRingBuffer:
    """Fixed-size ring buffer for CSI packets per zone.

    Default: 5 seconds at 50 Hz = 250 frames.
    """

    zone_id: str
    max_frames: int = 250
    _buffer: deque[dict[str, Any]] = field(default_factory=deque)

    def push(self, packet: dict[str, Any]) -> None:
        self._buffer.append(packet)
        while len(self._buffer) > self.max_frames:
            self._buffer.popleft()

    def get_recent(self, n_frames: int) -> list[dict[str, Any]]:
        frames = list(self._buffer)[-n_frames:]
        return frames

    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class IngestionStats:
    packets_received: int = 0
    packets_invalid: int = 0
    packets_dropped: int = 0
    last_packet_time: float = 0.0


class CSIIngestion:
    """MQTT-based CSI ingestion manager for multiple zones."""

    def __init__(self, zones: list[ZoneConfig], broker_host: str = "localhost", broker_port: int = 1883) -> None:
        self.zones: dict[str, ZoneConfig] = {z.zone_id: z for z in zones}
        self.buffers: dict[str, CSIRingBuffer] = {z.zone_id: CSIRingBuffer(zone_id=z.zone_id) for z in zones}
        self.stats: dict[str, IngestionStats] = {z.zone_id: IngestionStats() for z in zones}
        self._callbacks: list[callable] = []

        self._client = mqtt.Client(client_id="eldercare-ingestion")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._broker_host = broker_host
        self._broker_port = broker_port

    def start(self) -> None:
        logger.info(f"Connecting to MQTT broker at {self._broker_host}:{self._broker_port}")
        self._client.connect(self._broker_host, self._broker_port, keepalive=60)
        self._client.loop_start()

    def stop(self) -> None:
        logger.info("Stopping MQTT ingestion")
        self._client.loop_stop()
        self._client.disconnect()

    def register_callback(self, callback: callable) -> None:
        """Register a callback to receive validated CSI packets.

        Callback signature: callback(zone_id: str, packet: dict) -> None
        """
        self._callbacks.append(callback)

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: dict, rc: int) -> None:
        if rc == 0:
            logger.info("MQTT connected successfully")
            for zone in self.zones.values():
                if zone.active:
                    client.subscribe(zone.mqtt_topic)
                    logger.info(f"  Subscribed: {zone.mqtt_topic} ({zone.name})")
        else:
            logger.error(f"MQTT connection failed: rc={rc}")

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            topic = msg.topic
            zone = self._find_zone_by_topic(topic)
            if zone is None:
                logger.warning(f"Received message on unknown topic: {topic}")
                return

            stats = self.stats[zone.zone_id]
            stats.packets_received += 1

            packet = self._parse_packet(msg.payload)
            if packet is None:
                stats.packets_invalid += 1
                return

            csipkt = self._validate_packet(packet, zone.zone_id)
            if csipkt is None:
                stats.packets_invalid += 1
                return

            stats.last_packet_time = csipkt["timestamp"]

            self.buffers[zone.zone_id].push(csipkt)

            for cb in self._callbacks:
                try:
                    cb(zone.zone_id, csipkt)
                except Exception:
                    logger.exception(f"Callback error for zone={zone.zone_id}")

        except Exception:
            logger.exception("Error processing MQTT message")
            # Attempt to find any matching zone stats
            try:
                zone = self._find_zone_by_topic(topic)
                if zone:
                    self.stats[zone.zone_id].packets_dropped += 1
            except Exception:
                pass

    def _find_zone_by_topic(self, topic: str) -> Optional[ZoneConfig]:
        for zone in self.zones.values():
            if zone.mqtt_topic == topic:
                return zone
        return None

    def _parse_packet(self, payload: bytes) -> Optional[dict[str, Any]]:
        """Try JSON first, then binary format."""
        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        try:
            n_subcarriers = 52
            fmt = f"<dfi{'f' * n_subcarriers}{'f' * n_subcarriers}f"

            expected_size = struct.calcsize(fmt)
            if len(payload) != expected_size:
                return None

            values = struct.unpack(fmt, payload)
            timestamp, _, seq = values[:3]
            amp = list(values[3:3 + n_subcarriers])
            phase = list(values[3 + n_subcarriers:3 + 2 * n_subcarriers])
            rssi = values[-1]

            return {
                "timestamp": timestamp,
                "sequence_number": seq,
                "csi_amplitude": amp,
                "csi_phase": phase,
                "rssi": rssi,
            }
        except (struct.error, IndexError):
            return None

    def _validate_packet(self, packet: dict[str, Any], zone_id: str) -> Optional[dict[str, Any]]:
        """Validate packet structure. Returns enriched packet or None if invalid."""
        required_keys = {"timestamp", "sequence_number", "csi_amplitude", "csi_phase", "rssi"}
        if not required_keys.issubset(packet.keys()):
            return None

        n_subcarriers = 52
        if len(packet["csi_amplitude"]) != n_subcarriers or len(packet["csi_phase"]) != n_subcarriers:
            return None

        packet["zone_id"] = zone_id
        return packet
