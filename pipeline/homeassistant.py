"""Home Assistant MQTT Discovery integration.

Publishes MQTT discovery payloads so ElderCare devices auto-appear
as sensors in Home Assistant, enabling HA automations and dashboards.
"""

import json
import logging
import time
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

HA_DISCOVERY_PREFIX = "homeassistant"
HA_DEVICE_ID = "eldercare_server"
HA_MANUFACTURER = "ElderCare"


class HomeAssistantDiscovery:
    """Publishes MQTT auto-discovery messages for Home Assistant."""

    def __init__(
        self,
        mqtt_client: Optional[mqtt.Client] = None,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        discovery_prefix: str = HA_DISCOVERY_PREFIX,
    ) -> None:
        self._client = mqtt_client
        self._host = mqtt_host
        self._port = mqtt_port
        self._prefix = discovery_prefix
        self._owns_client = mqtt_client is None

    def _get_client(self) -> Optional[mqtt.Client]:
        if self._client is not None:
            return self._client
        try:
            client = mqtt.Client(client_id="eldercare_ha_discovery")
            client.connect(self._host, self._port, keepalive=60)
            self._client = client
            return client
        except Exception as e:
            logger.warning(f"Cannot connect to MQTT for HA discovery: {e}")
            return None

    def _device_info(self) -> dict:
        return {
            "identifiers": [HA_DEVICE_ID],
            "name": "ElderCare Monitoring System",
            "manufacturer": HA_MANUFACTURER,
            "model": "WiFi CSI Monitoring",
            "sw_version": "0.1.0",
        }

    def publish_zone_sensors(self, zone_id: str, zone_name: str) -> None:
        """Publish discovery messages for all sensors in a zone."""
        sensors = [
            {
                "component": "binary_sensor",
                "object_id": f"eldercare_{zone_id}_fall",
                "config": {
                    "name": f"Fall Detection - {zone_name}",
                    "state_topic": f"eldercare/ha/{zone_id}/fall",
                    "payload_on": "DETECTED",
                    "payload_off": "CLEAR",
                    "device_class": "safety",
                    "value_template": "{{ value_json.state }}",
                },
            },
            {
                "component": "sensor",
                "object_id": f"eldercare_{zone_id}_respiration",
                "config": {
                    "name": f"Respiration Rate - {zone_name}",
                    "state_topic": f"eldercare/ha/{zone_id}/respiration",
                    "unit_of_measurement": "BPM",
                    "icon": "mdi:lungs",
                    "value_template": "{{ value_json.bpm }}",
                },
            },
            {
                "component": "sensor",
                "object_id": f"eldercare_{zone_id}_activity",
                "config": {
                    "name": f"Activity State - {zone_name}",
                    "state_topic": f"eldercare/ha/{zone_id}/activity",
                    "icon": "mdi:motion-sensor",
                    "value_template": "{{ value_json.state }}",
                },
            },
            {
                "component": "sensor",
                "object_id": f"eldercare_{zone_id}_sleep_stage",
                "config": {
                    "name": f"Sleep Stage - {zone_name}",
                    "state_topic": f"eldercare/ha/{zone_id}/sleep",
                    "icon": "mdi:sleep",
                    "value_template": "{{ value_json.stage }}",
                },
            },
            {
                "component": "sensor",
                "object_id": f"eldercare_{zone_id}_sleep_score",
                "config": {
                    "name": f"Sleep Score - {zone_name}",
                    "state_topic": f"eldercare/ha/{zone_id}/sleep_score",
                    "unit_of_measurement": "%",
                    "icon": "mdi:sleep",
                    "value_template": "{{ value_json.score }}",
                },
            },
            {
                "component": "binary_sensor",
                "object_id": f"eldercare_{zone_id}_online",
                "config": {
                    "name": f"ESP32 Online - {zone_name}",
                    "state_topic": f"eldercare/ha/{zone_id}/online",
                    "payload_on": "true",
                    "payload_off": "false",
                    "device_class": "connectivity",
                },
            },
        ]

        client = self._get_client()
        if client is None:
            return

        for sensor in sensors:
            topic = f"{self._prefix}/{sensor['component']}/{HA_DEVICE_ID}/{sensor['object_id']}/config"
            config = {**sensor["config"], "device": self._device_info(), "unique_id": sensor["object_id"]}
            payload = json.dumps(config)
            client.publish(topic, payload, qos=1, retain=True)

        logger.info(f"Published HA discovery for {len(sensors)} sensors in zone {zone_id}")

    def publish_system_sensors(self) -> None:
        """Publish discovery for system-level sensors."""
        sensors = [
            {
                "component": "sensor",
                "object_id": "eldercare_cpu_usage",
                "config": {
                    "name": "ElderCare CPU Usage",
                    "state_topic": "eldercare/ha/system/cpu",
                    "unit_of_measurement": "%",
                    "icon": "mdi:cpu-64-bit",
                },
            },
            {
                "component": "sensor",
                "object_id": "eldercare_memory_usage",
                "config": {
                    "name": "ElderCare Memory Usage",
                    "state_topic": "eldercare/ha/system/memory",
                    "unit_of_measurement": "%",
                    "icon": "mdi:memory",
                },
            },
            {
                "component": "sensor",
                "object_id": "eldercare_disk_usage",
                "config": {
                    "name": "ElderCare Disk Usage",
                    "state_topic": "eldercare/ha/system/disk",
                    "unit_of_measurement": "%",
                    "icon": "mdi:harddisk",
                },
            },
        ]

        client = self._get_client()
        if client is None:
            return

        for sensor in sensors:
            topic = f"{self._prefix}/{sensor['component']}/{HA_DEVICE_ID}/{sensor['object_id']}/config"
            config = {**sensor["config"], "device": self._device_info(), "unique_id": sensor["object_id"]}
            payload = json.dumps(config)
            client.publish(topic, payload, qos=1, retain=True)

        logger.info(f"Published HA discovery for {len(sensors)} system sensors")

    def update_state(self, zone_id: str, sensor_type: str, payload: dict) -> None:
        """Publish a state update for a zone sensor."""
        client = self._get_client()
        if client is None:
            return
        topic = f"eldercare/ha/{zone_id}/{sensor_type}"
        client.publish(topic, json.dumps(payload), qos=0)

    def remove_zone(self, zone_id: str) -> None:
        """Remove HA discovery for a zone (publish empty config to clear)."""
        client = self._get_client()
        if client is None:
            return

        for component in ["binary_sensor", "sensor"]:
            for suffix in ["fall", "respiration", "activity", "sleep", "sleep_score", "online"]:
                object_id = f"eldercare_{zone_id}_{suffix}"
                topic = f"{self._prefix}/{component}/{HA_DEVICE_ID}/{object_id}/config"
                client.publish(topic, "", qos=1, retain=True)

        logger.info(f"Removed HA discovery for zone {zone_id}")

    def disconnect(self) -> None:
        if self._client and self._owns_client:
            self._client.disconnect()
            self._client = None
