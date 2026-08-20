import math
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from app.model_input import ModelInputCoordinator
from app.models import PoseKeypoint, PoseState, TrackingState, WorkerState


def worker(
    timestamp: float,
    frame_number: int,
    *,
    worker_id: str = "worker01",
    source: str = "live",
    camera_id: str | None = "camera01",
    track_id: int | None = 1,
) -> WorkerState:
    return WorkerState(
        worker_id=worker_id,
        source=source,
        tracking=TrackingState(camera_id=camera_id, track_id=track_id, online=True),
        pose=PoseState(
            frame_number=frame_number,
            captured_at=datetime.fromtimestamp(timestamp, timezone.utc),
            image_width=640,
            image_height=480,
            keypoints=[PoseKeypoint(x=float(i), y=float(i + 1), confidence=0.9) for i in range(17)],
        ),
    )


class ModelInputCoordinatorTests(unittest.TestCase):
    def coordinator(self, input_hz=None):
        return ModelInputCoordinator(
            input_hz=input_hz,
            reset_gap_seconds=2.0,
            use_environment=False,
        )

    def test_no_cadence_gate_accepts_every_increasing_timestamp(self):
        coordinator = self.coordinator()
        decisions = [coordinator.evaluate(worker(index / 20, index)) for index in range(10)]

        self.assertTrue(all(decision.accepted for decision in decisions))
        diagnostic = coordinator.diagnostic("worker01")
        self.assertIsNone(diagnostic["configured_input_hz"])
        self.assertEqual(diagnostic["accepted_count"], 10)
        self.assertEqual(diagnostic["rejected_count"], 0)

    def test_five_hz_uses_timestamp_buckets_for_5_43_hz_input(self):
        coordinator = self.coordinator(5.0)
        timestamps = [index / 5.43 for index in range(55)]
        decisions = [coordinator.evaluate(worker(timestamp, index)) for index, timestamp in enumerate(timestamps)]
        expected_slots = {math.floor(timestamp * 5.0) for timestamp in timestamps}

        self.assertEqual(sum(decision.accepted for decision in decisions), len(expected_slots))
        self.assertGreaterEqual(sum(decision.accepted for decision in decisions), 49)
        self.assertEqual(coordinator.diagnostic("worker01")["last_cadence_slot"], 49)

    def test_duplicate_timestamp_is_rejected(self):
        coordinator = self.coordinator()
        self.assertTrue(coordinator.evaluate(worker(1.0, 1)).accepted)

        decision = coordinator.evaluate(worker(1.0, 2))

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "duplicate_timestamp")

    def test_identity_changes_reset_before_acceptance(self):
        changes = (
            ({"source": "replay"}, "source"),
            ({"camera_id": "camera02"}, "camera"),
            ({"track_id": 2}, "track"),
        )
        for values, label in changes:
            with self.subTest(change=label):
                coordinator = self.coordinator()
                coordinator.evaluate(worker(1.0, 1))

                decision = coordinator.evaluate(worker(1.1, 2, **values))

                self.assertTrue(decision.accepted)
                self.assertTrue(decision.reset_required)
                self.assertEqual(decision.reason, "identity_change")

    def test_frame_rewind_resets_when_time_advances(self):
        coordinator = self.coordinator()
        coordinator.evaluate(worker(1.0, 10))

        decision = coordinator.evaluate(worker(1.1, 1))

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.reset_required)
        self.assertEqual(decision.reason, "frame_rewind")

    def test_time_gap_over_two_seconds_resets(self):
        coordinator = self.coordinator()
        coordinator.evaluate(worker(1.0, 1))

        decision = coordinator.evaluate(worker(3.01, 2))

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.reset_required)
        self.assertEqual(decision.reason, "time_gap")

    def test_reset_and_worker_state_are_isolated(self):
        coordinator = self.coordinator(5.0)
        coordinator.evaluate(worker(1.0, 1, worker_id="worker01"))
        coordinator.evaluate(worker(1.0, 1, worker_id="worker02"))

        coordinator.reset("worker01", "session_start")

        first = coordinator.diagnostic("worker01")
        second = coordinator.diagnostic("worker02")
        self.assertEqual(first["last_decision_reason"], "session_start")
        self.assertIsNone(first["current_identity"])
        self.assertEqual(first["accepted_count"], 0)
        self.assertEqual(second["accepted_count"], 1)
        self.assertEqual(second["current_identity"]["track_id"], 1)

    def test_environment_configuration(self):
        disabled_values = ("", "0", "-5")
        for value in disabled_values:
            with self.subTest(value=value), patch.dict(os.environ, {"MODEL_INPUT_HZ": value}):
                self.assertIsNone(ModelInputCoordinator().input_hz)
        with patch.dict(os.environ, {"MODEL_INPUT_HZ": "5"}):
            self.assertEqual(ModelInputCoordinator().input_hz, 5.0)
        for value in ("not-a-number", "nan", "inf"):
            with self.subTest(invalid=value), patch.dict(os.environ, {"MODEL_INPUT_HZ": value}):
                with self.assertRaisesRegex(ValueError, "MODEL_INPUT_HZ"):
                    ModelInputCoordinator()


if __name__ == "__main__":
    unittest.main()
