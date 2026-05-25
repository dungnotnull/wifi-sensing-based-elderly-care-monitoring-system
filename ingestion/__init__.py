"""ElderCare CSI data ingestion layer.

Custom Python MQTT ingestion with multi-zone topic routing:
  - Per-zone MQTT topic subscription (eldercare/csi/{zone_id})
  - Ring buffer (5s at 50 Hz = 250 frames)
  - JSON + binary packet parsing
  - Zone validation + statistics

MQTT is chosen over RuView's UDP transport for multi-zone
topic isolation -- each zone has its own MQTT topic, and the
ingestion layer routes packets to per-zone inference workers.
"""
