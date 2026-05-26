"""
UDP-to-MQTT Bridge for RuView ESP32-S3 CSI Node Firmware

Receives CSI frames in ADR-018 binary format from ESP32-S3 nodes over UDP,
converts them to ElderCare's MQTT packet format, and publishes to per-zone
MQTT topics.

ADR-018 Binary Frame Format (from csi_collector.c):
  Offset  Size  Field
  0       4     Magic: 0xC5110001
  4       1     Node ID (maps to zone_id)
  5       1     Number of antennas
  6       2     Number of subcarriers (LE u16)
  8       4     Frequency MHz (LE u32)
  12      4     Sequence number (LE u32)
  16      1     RSSI (i8)
  17      1     Noise floor (i8)
  18      2     Reserved / ADR-110 PPDU tags
  20      N*2   I/Q pairs (n_antennas * n_subcarriers * 2 bytes)
                Each pair: I (i8) + Q (i8) = 2 bytes signed

Usage:
    python -m ingestion.udp_mqtt_bridge --config configs/zones.yaml

The bridge maps RuView node_id to ElderCare zone_id via configs/zones.yaml:
    zones:
      - zone_id: zone_bedroom
        node_id: 1
        ...

Node ID = 0 is reserved for "unknown" and publishes to a default topic.
"""

import argparse
import json
import logging
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

CSI_MAGIC = 0xC5110001
CSI_HEADER_SIZE = 20


class NodeConfig:
    """Maps a RuView node_id to an ElderCare zone_id and MQTT topic."""
    def __init__(self, node_id: int, zone_id: str, name: str, topic: str) -> None:
        self.node_id = node_id
        self.zone_id = zone_id
        self.name = name
        self.topic = topic


class UDPBridge:
    """Listens on a UDP port, decodes ADR-018 CSI frames, publishes to MQTT."""

    def __init__(
        self,
        listen_port: int = 5005,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        config_path: str = "configs/zones.yaml",
    ) -> None:
        self.listen_port = listen_port
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.config_path = config_path

        self._nodes: dict[int, NodeConfig] = {}
        self._mqtt_client: Optional[object] = None
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._stats: dict[str, int] = {
            "frames_received": 0,
            "frames_published": 0,
            "frames_dropped": 0,
            "parse_errors": 0,
            "mqtt_errors": 0,
        }

        self._load_node_map()

    def _load_node_map(self) -> None:
        """Load zone config and build node_id → zone mapping."""
        path = Path(self.config_path)
        if not path.exists():
            logger.warning(f"Config not found: {path}. Using default mapping (node_id → zone_N).")
            return

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        for zone in config.get("zones", []):
            zone_id = zone.get("zone_id", "unknown")
            name = zone.get("name", zone_id)
            topic = zone.get("mqtt_topic", f"eldercare/csi/{zone_id}")
            node_id = zone.get("node_id")

            if node_id is not None:
                self._nodes[int(node_id)] = NodeConfig(
                    node_id=int(node_id), zone_id=zone_id, name=name, topic=topic,
                )
                logger.info(f"Mapped node_id={node_id} → zone={zone_id} ({name}), topic={topic}")

    def _decode_frame(self, data: bytes) -> Optional[dict]:
        """Decode an ADR-018 binary frame into ElderCare CSI packet format.

        Returns dict matching ElderCare's expected format:
          {zone_id, timestamp, sequence_number, csi_amplitude[52], csi_phase[52], rssi}
        or None if parse fails.
        """
        if len(data) < CSI_HEADER_SIZE:
            return None

        # Magic
        magic = struct.unpack_from("<I", data, 0)[0]
        if magic != CSI_MAGIC:
            return None

        node_id = data[4]
        n_antennas = data[5]
        n_subcarriers = struct.unpack_from("<H", data, 6)[0]
        freq_mhz = struct.unpack_from("<I", data, 8)[0]
        sequence = struct.unpack_from("<I", data, 12)[0]
        rssi = data[16] if data[16] < 128 else data[16] - 256  # i8 → int
        # noise_floor = data[17]  # currently not passed through

        # Validate payload size
        iq_len = n_antennas * n_subcarriers * 2  # 2 bytes per I/Q pair
        expected_size = CSI_HEADER_SIZE + iq_len
        if len(data) < expected_size:
            logger.debug(f"Frame truncated: got {len(data)}, expected {expected_size}")
            return None

        # Extract I/Q pairs
        iq_offset = CSI_HEADER_SIZE
        i_data = np.zeros(n_subcarriers, dtype=np.float32)
        q_data = np.zeros(n_subcarriers, dtype=np.float32)

        # Unpack all I/Q bytes
        iq_bytes = data[iq_offset:iq_offset + iq_len]
        # Convert signed bytes to numpy
        raw = np.frombuffer(iq_bytes, dtype=np.int8)

        if n_antennas == 1:
            # I and Q interleaved: I0, Q0, I1, Q1, ...
            i_data = raw[0::2].astype(np.float32)
            q_data = raw[1::2].astype(np.float32)
        else:
            # Multiple antennas: per-antenna I/Q blocks
            # Take first antenna's data for simplicity
            per_antenna_samples = n_subcarriers * 2
            raw_a0 = raw[:per_antenna_samples]
            i_data = raw_a0[0::2].astype(np.float32)
            q_data = raw_a0[1::2].astype(np.float32)

        # Compute amplitude and phase from I/Q
        amplitude = np.sqrt(i_data ** 2 + q_data ** 2)
        phase = np.arctan2(q_data, i_data)

        # Normalize amplitude to ElderCare's expected range (~1.0 baseline)
        amp_max = np.max(amplitude)
        if amp_max > 0:
            amplitude = amplitude / amp_max * 1.0

        # If we have exactly 52 subcarriers, use directly.
        # If more (e.g., 64), downsample to 52 to match our model input.
        if n_subcarriers > 52:
            indices = np.linspace(0, n_subcarriers - 1, 52, dtype=int)
            amplitude = amplitude[indices]
            phase = phase[indices]
        elif n_subcarriers < 52:
            # Pad with zeros if fewer subcarriers
            padded_amp = np.zeros(52, dtype=np.float32)
            padded_phase = np.zeros(52, dtype=np.float32)
            padded_amp[:n_subcarriers] = amplitude
            padded_phase[:n_subcarriers] = phase
            amplitude = padded_amp
            phase = padded_phase

        node_cfg = self._nodes.get(node_id)
        zone_id = node_cfg.zone_id if node_cfg else f"node_{node_id}"

        return {
            "zone_id": zone_id,
            "timestamp": time.time(),
            "sequence_number": sequence,
            "csi_amplitude": amplitude.tolist(),
            "csi_phase": phase.tolist(),
            "rssi": float(rssi),
            "_raw": {
                "node_id": node_id,
                "n_subcarriers_original": n_subcarriers,
                "n_antennas": n_antennas,
                "freq_mhz": freq_mhz,
            },
        }

    def _get_mqtt_client(self):
        if self._mqtt_client is not None:
            return self._mqtt_client

        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.error("paho-mqtt not installed. Install with: pip install paho-mqtt")
            raise

        client = mqtt.Client(client_id="eldercare-udp-bridge")
        client.on_connect = self._on_mqtt_connect

        try:
            client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            client.loop_start()
            self._mqtt_client = client
            logger.info(f"Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
            return client
        except Exception as e:
            logger.warning(f"MQTT connection failed: {e}. Frames will be logged only.")
            return None

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("MQTT connected successfully")
        else:
            logger.warning(f"MQTT connect failed with rc={rc}")

    def _publish(self, packet: dict) -> bool:
        zone_id = packet["zone_id"]
        node_cfg = self._nodes.get(packet["_raw"]["node_id"])
        topic = node_cfg.topic if node_cfg else f"eldercare/csi/{zone_id}"

        payload = json.dumps(packet)

        client = self._get_mqtt_client()
        if client is None:
            self._stats["mqtt_errors"] += 1
            return False

        try:
            result = client.publish(topic, payload, qos=1)
            if result.rc == 0:
                self._stats["frames_published"] += 1
                return True
            else:
                logger.debug(f"MQTT publish failed: rc={result.rc}")
                self._stats["mqtt_errors"] += 1
                return False
        except Exception:
            logger.debug("MQTT publish error", exc_info=True)
            self._stats["mqtt_errors"] += 1
            return False

    def _process_frame(self, data: bytes, addr: tuple) -> None:
        self._stats["frames_received"] += 1

        packet = self._decode_frame(data)
        if packet is None:
            self._stats["parse_errors"] += 1
            return

        node_id = packet["_raw"]["node_id"]
        zone_id = packet["zone_id"]

        if self._stats["frames_received"] % 100 == 0:
            logger.info(
                f"Received {self._stats['frames_received']} frames | "
                f"node={node_id} zone={zone_id} seq={packet['sequence_number']} "
            )

        self._publish(packet)

    def start(self) -> None:
        self._running = True
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("0.0.0.0", self.listen_port))
        self._socket.settimeout(1.0)

        logger.info(f"UDP bridge listening on 0.0.0.0:{self.listen_port}")
        logger.info(f"Node mappings: {len(self._nodes)} configured")
        for node in self._nodes.values():
            logger.info(f"  node_id={node.node_id} → {node.zone_id} ({node.name})")

        try:
            while self._running:
                try:
                    data, addr = self._socket.recvfrom(4096)
                    self._process_frame(data, addr)
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        logger.exception("Socket error")
                    break
        finally:
            self._shutdown()

    def stop(self) -> None:
        logger.info("Stopping UDP bridge...")
        self._running = False

    def _shutdown(self) -> None:
        if self._mqtt_client is not None:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
        if self._socket is not None:
            self._socket.close()

        logger.info(f"Bridge stopped. Stats: {json.dumps(self._stats)}")

    def get_stats(self) -> dict:
        return dict(self._stats)


def main():
    parser = argparse.ArgumentParser(
        description="UDP-to-MQTT bridge for RuView ESP32-S3 CSI firmware"
    )
    parser.add_argument("--config", default="configs/zones.yaml", help="Zone configuration file")
    parser.add_argument("--listen-port", type=int, default=5005, help="UDP port to listen on")
    parser.add_argument("--mqtt-broker", default="localhost", help="MQTT broker address")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bridge = UDPBridge(
        listen_port=args.listen_port,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        config_path=args.config,
    )

    def handle_signal(signum, frame):
        bridge.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    bridge.start()


if __name__ == "__main__":
    main()
