import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.database import Base, engine, SessionLocal
from app.db_models import MultimodalSession, SessionMobileSample, SessionVisionSample
from app.models import ActivityState, MobileLocation, MobileTelemetry, TrackingState, Vector3, WorkerState
from app.sessions import SessionRecorder


class MultimodalSessionTests(unittest.TestCase):
    def setUp(self):
        Base.metadata.create_all(bind=engine)
        self.recorder = SessionRecorder()
        self.started_ids = []

    def tearDown(self):
        with SessionLocal() as db:
            for session_id in self.started_ids:
                db.query(SessionVisionSample).filter_by(session_id=session_id).delete()
                db.query(SessionMobileSample).filter_by(session_id=session_id).delete()
                db.query(MultimodalSession).filter_by(session_id=session_id).delete()
            db.commit()

    def start(self, **kwargs):
        result = self.recorder.start(
            worker_id="worker01", source_mode="LIVE", notes="controlled test",
            expected_activity=kwargs.pop("expected_activity", None), cadence_hz=10,
            max_samples=kwargs.pop("max_samples", 100), **kwargs,
        )
        self.started_ids.append(result["session_id"])
        return result["session_id"]

    @staticmethod
    def worker(timestamp):
        return WorkerState(
            worker_id="worker01", timestamp=timestamp,
            tracking=TrackingState(track_id=7, camera_id="esp32_cam_01", online=True),
            activity=ActivityState(
                baseline="walking", baseline_confidence=.71,
                stgcn="standing", stgcn_confidence=.62, display_activity="walking",
            ),
        )

    @staticmethod
    def mobile(timestamp, state="connected", gps=True):
        return MobileTelemetry(
            worker_id="worker01", device_id="persistent-device", timestamp=timestamp,
            received_at=timestamp + timedelta(milliseconds=40),
            connection_state=state,
            accelerometer=Vector3(x=1, y=2, z=3), gyroscope=Vector3(x=4, y=5, z=6),
            location=MobileLocation(
                latitude=51.5 if gps else None, longitude=-.12 if gps else None,
                accuracy_m=8.4 if gps else None, gps_enabled=gps, zone="zone-a" if gps else None,
            ),
        )

    def test_nearest_timestamp_alignment_preserves_sources_and_models(self):
        session_id = self.start()
        base = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        self.recorder.record_mobile(self.mobile(base + timedelta(milliseconds=80)))
        self.recorder.record_vision(self.worker(base), received=base + timedelta(milliseconds=25))

        _, rows = self.recorder._rows(session_id)
        row = rows[0]
        self.assertEqual(row["device_id"], "persistent-device")
        self.assertEqual(row["camera_track_id"], 7)
        self.assertEqual(row["baseline_activity"], "walking")
        self.assertEqual(row["stgcn_activity"], "standing")
        self.assertEqual(row["source_time_delta_ms"], -80.0)
        self.assertEqual(row["vision_age_ms"], 25.0)
        self.assertFalse(row["mobile_missing"])

    @patch.dict(os.environ, {"SESSION_ALIGNMENT_TOLERANCE_MS": "100"})
    def test_missing_and_stale_mobile_are_explicit(self):
        session_id = self.start()
        base = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        self.recorder.record_mobile(self.mobile(base, state="stale", gps=False))
        self.recorder.record_vision(self.worker(base + timedelta(milliseconds=50)), received=base + timedelta(milliseconds=70))
        self.recorder.record_vision(self.worker(base + timedelta(seconds=2)), received=base + timedelta(seconds=2, milliseconds=20))

        _, rows = self.recorder._rows(session_id)
        self.assertTrue(rows[0]["mobile_stale"])
        self.assertTrue(rows[0]["gps_missing"])
        self.assertTrue(rows[1]["mobile_missing"])
        self.assertEqual(rows[1]["connection_state"], "missing")
        summary = self.recorder.summary(session_id)
        self.assertEqual(summary["stale_count"], 2)
        self.assertEqual(summary["missing_mobile_count"], 1)

    def test_start_stop_status_and_csv_json_export(self):
        session_id = self.start(expected_activity="walking")
        self.assertTrue(self.recorder.status()["active"])
        base = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        mobile = self.mobile(base)
        self.recorder.record_mobile(mobile)
        self.assertFalse(self.recorder.record_mobile(mobile))
        self.recorder.record_vision(self.worker(base), received=base + timedelta(milliseconds=10))
        summary = self.recorder.stop()
        self.assertFalse(self.recorder.status()["active"])
        self.assertEqual(summary["duplicate_mobile_samples"], 1)
        csv_path = self.recorder.export(session_id, "csv")
        json_path = self.recorder.export(session_id, "json")
        self.assertIn("baseline_activity", csv_path.read_text())
        payload = json.loads(json_path.read_text())
        self.assertEqual(payload["metadata"]["expected_activity"], "walking")
        self.assertEqual(len(payload["samples"]), 1)
        csv_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
