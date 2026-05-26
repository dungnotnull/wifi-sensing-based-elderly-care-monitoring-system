"""Tests for the UDP-to-MQTT bridge ADR-018 frame decoder."""
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.udp_mqtt_bridge import CSI_MAGIC, CSI_HEADER_SIZE, UDPBridge


def _build_frame(node_id, n_sub=64, n_ant=1, freq=2412, seq=42, rssi=-40, noise=-76):
    iq_len = n_sub * n_ant * 2
    buf = bytearray(CSI_HEADER_SIZE + iq_len)
    struct.pack_into('<I', buf, 0, CSI_MAGIC)
    buf[4] = node_id
    buf[5] = n_ant
    struct.pack_into('<H', buf, 6, n_sub)
    struct.pack_into('<I', buf, 8, freq)
    struct.pack_into('<I', buf, 12, seq)
    buf[16] = rssi & 0xFF
    buf[17] = noise & 0xFF
    for i in range(n_sub):
        phase = 2 * math.pi * i / 8
        i_val = int(60 * math.cos(phase))
        q_val = int(60 * math.sin(phase))
        offset = CSI_HEADER_SIZE + i * 2
        buf[offset] = i_val & 0xFF
        buf[offset + 1] = q_val & 0xFF
    return bytes(buf)


class TestUDPBridge:
    def setup_method(self):
        self.bridge = UDPBridge(config_path='configs/zones.yaml', listen_port=5555)

    def test_valid_frame_node3_maps_to_hallway(self):
        data = _build_frame(node_id=3)
        pkt = self.bridge._decode_frame(data)
        assert pkt is not None
        assert pkt['zone_id'] == 'zone_hallway'
        assert pkt['sequence_number'] == 42
        assert pkt['rssi'] == -40
        assert len(pkt['csi_amplitude']) == 52
        assert len(pkt['csi_phase']) == 52

    def test_node2_maps_to_living(self):
        data = _build_frame(node_id=2)
        pkt = self.bridge._decode_frame(data)
        assert pkt['zone_id'] == 'zone_living'

    def test_node1_maps_to_bedroom(self):
        data = _build_frame(node_id=1)
        pkt = self.bridge._decode_frame(data)
        assert pkt['zone_id'] == 'zone_bedroom'

    def test_unknown_node_gets_generated_zone(self):
        data = _build_frame(node_id=99)
        pkt = self.bridge._decode_frame(data)
        assert pkt['zone_id'] == 'node_99'

    def test_bad_magic_rejected(self):
        data = bytearray(_build_frame(node_id=1))
        data[0] = 0xFF
        assert self.bridge._decode_frame(bytes(data)) is None

    def test_truncated_frame_rejected(self):
        data = _build_frame(node_id=1)[:10]
        assert self.bridge._decode_frame(data) is None

    def test_downsample_64_to_52(self):
        data = _build_frame(node_id=1, n_sub=64)
        pkt = self.bridge._decode_frame(data)
        assert len(pkt['csi_amplitude']) == 52

    def test_pad_fewer_than_52_subcarriers(self):
        data = _build_frame(node_id=1, n_sub=40)
        pkt = self.bridge._decode_frame(data)
        assert len(pkt['csi_amplitude']) == 52

    def test_raw_metadata_present(self):
        data = _build_frame(node_id=1, n_sub=64, n_ant=1, freq=2412)
        pkt = self.bridge._decode_frame(data)
        raw = pkt['_raw']
        assert raw['node_id'] == 1
        assert raw['n_subcarriers_original'] == 64
        assert raw['n_antennas'] == 1
        assert raw['freq_mhz'] == 2412
