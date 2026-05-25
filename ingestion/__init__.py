"""ElderCare CSI data ingestion layer.

MQTT ingestion mirrors RuView's wifi-densepose Python client:
  - Per-zone MQTT topic subscription (eldercare/csi/{zone_id})
  - Ring buffer (5s at 50 Hz = 250 frames)
  - JSON + binary packet parsing
  - Zone validation + statistics

RuView source reference:
  python/wifi_densepose/client/mqtt.py
"""

