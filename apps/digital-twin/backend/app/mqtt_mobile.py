import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from pydantic import ValidationError

from .models import MobileLocation, MobileTelemetry, Vector3

LOGGER = logging.getLogger(__name__)


def _timestamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
    raise ValueError("timestamp must be ISO 8601 or epoch_ms")


def parse_mobile_payload(payload: bytes | str, topic: str) -> MobileTelemetry:
    data = json.loads(payload)
    worker_id = data.get("worker_id") or data.get("workerID")
    topic_worker = topic.split("/")[-2] if topic.endswith("/mobile") else topic.split("/")[-1]
    if not worker_id:
        worker_id = topic_worker
    accelerometer = data.get("accelerometer") or {
        "x": data.get("ax"), "y": data.get("ay"), "z": data.get("az")
    }
    gyroscope = data.get("gyroscope") or {
        "x": data.get("gx"), "y": data.get("gy"), "z": data.get("gz")
    }
    gps = data.get("gps") or data.get("location") or {}
    if not gps:
        gps = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "accuracy_m": data.get("locationAccuracy"),
            "gps_enabled": data.get("gpsEnabled", False),
            "permission_state": data.get("gps_permission", "unknown"),
            "zone": data.get("zone"),
        }
    return MobileTelemetry(
        worker_id=worker_id,
        device_id=data.get("device_id", "legacy-unknown"),
        mqtt_client_id=data.get("mqtt_client_id"),
        timestamp=_timestamp(data.get("timestamp") or data.get("epoch_ms")),
        received_at=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        accelerometer=Vector3.model_validate(accelerometer),
        gyroscope=Vector3.model_validate(gyroscope),
        location=MobileLocation.model_validate(gps),
        battery_level=data.get("battery_level"),
        association_method=data.get("association_method", "configured"),
        association_confidence=data.get("association_confidence"),
    )


class MobileMqttSubscriber:
    def __init__(self, on_telemetry: Callable[[MobileTelemetry], None]):
        self.on_telemetry = on_telemetry
        self.client = None

    def start(self) -> None:
        if os.getenv("MOBILE_MQTT_ENABLED", "1").lower() not in {"1", "true", "yes"}:
            LOGGER.info("Android MQTT subscriber disabled")
            return
        import paho.mqtt.client as mqtt

        client_id = os.getenv("MOBILE_MQTT_CLIENT_ID", "digital-twin-mac-mobile")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        username = os.getenv("MQTT_USERNAME", "")
        if username:
            self.client.username_pw_set(username, os.getenv("MQTT_PASSWORD", ""))
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

        def on_connect(client, _userdata, _flags, reason_code, _properties):
            if reason_code == 0:
                client.subscribe("digitaltwin/workers/+/mobile", qos=1)
                client.subscribe("factory/workers/+", qos=1)
                LOGGER.info("Subscribed to Android worker telemetry")
            else:
                LOGGER.warning("MQTT connection rejected: %s", reason_code)

        def on_message(_client, _userdata, message):
            try:
                self.on_telemetry(parse_mobile_payload(message.payload, message.topic))
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as error:
                LOGGER.warning("Ignored malformed mobile payload on %s: %s", message.topic, error)

        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect_async(
            os.getenv("MQTT_HOST", "127.0.0.1"), int(os.getenv("MQTT_PORT", "1883")), 20
        )
        self.client.loop_start()

    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
                self.client.loop_stop()
            except Exception as error:
                LOGGER.debug("MQTT shutdown completed with error: %s", error)
            self.client = None
