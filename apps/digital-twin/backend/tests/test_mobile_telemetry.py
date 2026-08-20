import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.models import MobileLocation, MobileTelemetry, Vector3, WorkerState
from app.mqtt_mobile import parse_mobile_payload
from app.state import WorkerStateManager


class MobileTelemetryTests(unittest.TestCase):
    def test_new_payload_preserves_identity_and_optional_gps(self):
        mobile = parse_mobile_payload(
            '{"worker_id":"worker01","device_id":"abc-123","mqtt_client_id":"android-worker01-phone-abc12345",'
            '"source":"android","timestamp":1770000000000,"accelerometer":{"x":1,"y":2,"z":3},'
            '"gyroscope":{"x":4,"y":5,"z":6},"gps":{"gps_enabled":false,"permission_state":"denied"},'
            '"association_method":"manual_pairing"}',
            "digitaltwin/workers/worker01/mobile",
        )
        self.assertEqual(mobile.worker_id, "worker01")
        self.assertEqual(mobile.device_id, "abc-123")
        self.assertIsNone(mobile.location.latitude)

    def test_malformed_missing_gps_does_not_crash(self):
        mobile = parse_mobile_payload(
            '{"worker_id":"worker01","device_id":"abc","timestamp":1770000000000,'
            '"accelerometer":{"x":1,"y":2,"z":3},"gyroscope":{"x":4,"y":5,"z":6}}',
            "digitaltwin/workers/worker01/mobile",
        )
        self.assertFalse(mobile.location.gps_enabled)

    def test_track_can_change_without_changing_mobile_identity(self):
        manager = WorkerStateManager()
        worker = WorkerState(worker_id="worker01")
        worker.tracking.track_id = 1
        manager.set_worker(worker)
        manager.set_mobile("worker01", self.mobile())
        updated = WorkerState(worker_id="worker01", mobile=manager.get_worker("worker01").mobile)
        updated.tracking.track_id = 99
        manager.set_worker(updated)
        self.assertEqual(manager.get_worker("worker01").mobile.device_id, "persistent-device")
        self.assertEqual(manager.get_worker("worker01").tracking.track_id, 99)

    @patch.dict(os.environ, {"MOBILE_STALE_AFTER_S": "5", "MOBILE_DISCONNECTED_AFTER_S": "30"})
    def test_stale_and_recovery(self):
        manager = WorkerStateManager()
        mobile = self.mobile()
        mobile.last_seen = datetime.now(timezone.utc) - timedelta(seconds=10)
        manager.set_mobile("worker01", mobile)
        self.assertEqual(manager.get_worker("worker01").mobile.connection_state, "stale")
        mobile.last_seen = datetime.now(timezone.utc)
        manager.set_mobile("worker01", mobile)
        self.assertEqual(manager.get_worker("worker01").mobile.connection_state, "connected")

    @staticmethod
    def mobile():
        return MobileTelemetry(
            worker_id="worker01", device_id="persistent-device",
            timestamp=datetime.now(timezone.utc), accelerometer=Vector3(x=1, y=2, z=3),
            gyroscope=Vector3(x=4, y=5, z=6), location=MobileLocation(),
        )


if __name__ == "__main__":
    unittest.main()
