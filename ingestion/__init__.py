"""ElderCare CSI data ingestion layer.

Two ingestion paths:
  1. MQTT (native): ESP32 publishes directly to eldercare/csi/{zone_id}
     via receiver.py. Each zone has its own MQTT topic for clean routing.
  2. UDP → MQTT (bridge): RuView ESP32-S3 firmware sends ADR-018 binary
     CSI frames over UDP (port 5005). udp_mqtt_bridge.py decodes them and
     republishes to MQTT topics defined in configs/zones.yaml.

Ring buffer (5s at 50 Hz = 250 frames), JSON + binary packet parsing,
zone validation + statistics.
"""
