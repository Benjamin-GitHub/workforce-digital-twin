
#!/usr/bin/env python3

import json
import socket
import time
from pathlib import Path

import psutil
import paho.mqtt.client as mqtt


MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "home/rpi5/status"
PUBLISH_INTERVAL_SECONDS = 5


def get_cpu_temperature() -> float | None:
    """Read the Raspberry Pi CPU temperature in Celsius."""

    thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")

    try:
        raw_value = thermal_file.read_text(encoding="utf-8").strip()
        return round(float(raw_value) / 1000.0, 1)
    except (OSError, ValueError):
        return None


def get_ip_address() -> str:
    """Find the main local IPv4 address without sending internet traffic."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("192.0.2.1", 80))
        return sock.getsockname()[0]
    except OSError:
        return "Unavailable"
    finally:
        sock.close()


def format_uptime() -> str:
    """Return uptime in a short LCD-friendly format."""

    uptime_seconds = max(0, int(time.time() - psutil.boot_time()))

    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60

    if days > 0:
        return f"{days}d {hours}h"

    return f"{hours}h {minutes}m"


def collect_status() -> dict:
    """Collect Raspberry Pi system information."""

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "hostname": socket.gethostname(),
        "cpu_temp": get_cpu_temperature(),
        "cpu_usage": round(psutil.cpu_percent(interval=0.5), 1),
        "ram_usage": round(memory.percent, 1),
        "disk_usage": round(disk.percent, 1),
        "uptime": format_uptime(),
        "ip": get_ip_address(),
        "timestamp": int(time.time()),
    }


def main() -> None:
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="rpi5-status-publisher",
    )
    client.username_pw_set("xiao", "Salam")

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()

    print(f"Publishing to {MQTT_TOPIC}")

    try:
        while True:
            status = collect_status()
            payload = json.dumps(status)

            result = client.publish(
                MQTT_TOPIC,
                payload,
                qos=1,
                retain=True,
            )

            result.wait_for_publish()
            print(payload)

            time.sleep(PUBLISH_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping publisher.")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()


