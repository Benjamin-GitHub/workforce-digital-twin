import unittest
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from app.gru import EXPECTED_CLASSES, GRUInferenceService
from app.models import PoseKeypoint, PoseState


REPO_ROOT = Path(__file__).resolve().parents[4]
OLD_CHECKPOINT = REPO_ROOT / "training/gru/runs/cml_plus_local_sqrt/best.pt"


def pose(frame_number: int = 1) -> PoseState:
    return PoseState(
        frame_number=frame_number,
        captured_at=datetime.now(timezone.utc),
        image_width=640,
        image_height=480,
        keypoints=[
            PoseKeypoint(x=float(i), y=float(i + 1), confidence=0.9)
            for i in range(17)
        ],
    )


class FakePredictor:
    classes = EXPECTED_CLASSES
    device = "cpu"
    minimum_observations = 5
    reset_gap_seconds = 2.0
    sequence_length = 16
    source_pose_hz = 5.0
    effective_pose_hz = 5.0
    temporal_stride = 1

    def __init__(self):
        self.received = None
        self.reset_called = False
        self.reset_worker_id = None

    def update(self, worker_id, values, now=None):
        self.received = (worker_id, values.copy(), now)
        return {
            "label": "carrying",
            "confidence": 0.87,
            "ready": True,
            "observations": 5,
            "probabilities": {
                name: 0.87 if name == "carrying" else 0.026
                for name in self.classes
            },
        }

    def reset(self, worker_id=None):
        self.reset_called = True
        self.reset_worker_id = worker_id


class GRUInferenceServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.new_checkpoint = Path(cls._temporary_directory.name) / "gru_5hz_w16.pt"
        checkpoint = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
        checkpoint["source_pose_hz"] = 5.0
        checkpoint["effective_pose_hz"] = 5.0
        checkpoint["temporal_stride"] = 1
        torch.save(checkpoint, cls.new_checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    def test_old_and_new_checkpoint_metadata_is_exposed(self):
        cases = (
            (OLD_CHECKPOINT, 11.0, 5.5, 2),
            (self.new_checkpoint, 5.0, 5.0, 1),
        )
        for checkpoint, source_hz, effective_hz, stride in cases:
            with self.subTest(checkpoint=checkpoint.name, source_hz=source_hz):
                service = GRUInferenceService(checkpoint)
                service.load()
                status = service.status()

                self.assertTrue(status["loaded"], status["error"])
                self.assertEqual(status["checkpoint"], str(checkpoint.resolve()))
                self.assertEqual(status["sequence_length"], 16)
                self.assertEqual(status["minimum_observations"], 16)
                self.assertEqual(status["source_pose_hz"], source_hz)
                self.assertEqual(status["effective_pose_hz"], effective_hz)
                self.assertEqual(status["temporal_stride"], stride)

    def test_environment_checkpoint_override_is_resolved(self):
        with patch.dict(os.environ, {"GRU_CHECKPOINT": str(self.new_checkpoint)}):
            service = GRUInferenceService()

        self.assertEqual(service.checkpoint_path, self.new_checkpoint.resolve())
        self.assertEqual(service.status()["checkpoint"], str(self.new_checkpoint.resolve()))

    def test_new_checkpoint_becomes_ready_at_sequence_length(self):
        service = GRUInferenceService(self.new_checkpoint)
        service.load()
        self.assertIsNotNone(service.predictor, service.error)

        prediction = None
        for frame_number in range(15):
            prediction = service.predict("worker01", pose(frame_number))
        self.assertFalse(prediction.ready)
        self.assertEqual(prediction.observations, 15)

        prediction = service.predict("worker01", pose(15))
        self.assertTrue(prediction.ready)
        self.assertEqual(prediction.observations, 16)

    def test_explicit_minimum_observations_override_is_preserved(self):
        service = GRUInferenceService(self.new_checkpoint)
        service.load()
        predictor_type = type(service.predictor)
        predictor = predictor_type(self.new_checkpoint, minimum_observations=3)

        self.assertEqual(predictor.sequence_length, 16)
        self.assertEqual(predictor.minimum_observations, 3)

    def test_pose_is_forwarded_and_prediction_is_retained(self):
        service = GRUInferenceService()
        service.predictor = FakePredictor()
        sample = pose()

        result = service.predict("worker01", sample)

        self.assertEqual(result.activity, "carrying")
        self.assertAlmostEqual(result.confidence, 0.87)
        self.assertTrue(result.ready)
        self.assertEqual(result.observations, 5)
        self.assertEqual(service.predictor.received[0], "worker01")
        self.assertEqual(service.predictor.received[1].shape, (17, 3))
        self.assertEqual(service.predictor.received[2], sample.captured_at.timestamp())
        self.assertEqual(service.latest("worker01")["activity"], "carrying")

    def test_status_describes_streaming_configuration(self):
        service = GRUInferenceService()
        service.predictor = FakePredictor()

        status = service.status()

        self.assertTrue(status["loaded"])
        self.assertEqual(status["device"], "cpu")
        self.assertEqual(status["minimum_observations"], 5)
        self.assertEqual(status["sequence_length"], 16)
        self.assertEqual(status["source_pose_hz"], 5.0)
        self.assertEqual(status["effective_pose_hz"], 5.0)
        self.assertEqual(status["temporal_stride"], 1)
        self.assertEqual(status["reset_gap_seconds"], 2.0)

    def test_inference_error_is_contained(self):
        class FailingPredictor(FakePredictor):
            def update(self, _worker_id, _values, now=None):
                raise RuntimeError("test failure")

        service = GRUInferenceService()
        service.predictor = FailingPredictor()

        self.assertIsNone(service.predict("worker01", pose()))
        self.assertIn("test failure", service.status()["error"])

    def test_clear_resets_predictions_and_hidden_states(self):
        service = GRUInferenceService()
        service.predictor = FakePredictor()
        service.predict("worker01", pose())

        service.clear_predictions()

        self.assertIsNone(service.latest("worker01"))
        self.assertTrue(service.predictor.reset_called)
        self.assertIsNone(service.predictor.reset_worker_id)

    def test_clear_one_worker_resets_only_that_worker(self):
        service = GRUInferenceService()
        service.predictor = FakePredictor()
        service.predict("worker01", pose())

        service.clear_predictions("worker01")

        self.assertIsNone(service.latest("worker01"))
        self.assertEqual(service.predictor.reset_worker_id, "worker01")


if __name__ == "__main__":
    unittest.main()
