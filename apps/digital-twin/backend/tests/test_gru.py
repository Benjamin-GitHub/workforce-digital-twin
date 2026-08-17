import unittest
from datetime import datetime, timezone

from app.gru import EXPECTED_CLASSES, GRUInferenceService
from app.models import PoseKeypoint, PoseState


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

    def __init__(self):
        self.received = None
        self.reset_called = False

    def update(self, worker_id, values):
        self.received = (worker_id, values.copy())
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

    def reset(self):
        self.reset_called = True


class GRUInferenceServiceTests(unittest.TestCase):
    def test_pose_is_forwarded_and_prediction_is_retained(self):
        service = GRUInferenceService()
        service.predictor = FakePredictor()

        result = service.predict("worker01", pose())

        self.assertEqual(result.activity, "carrying")
        self.assertAlmostEqual(result.confidence, 0.87)
        self.assertTrue(result.ready)
        self.assertEqual(result.observations, 5)
        self.assertEqual(service.predictor.received[0], "worker01")
        self.assertEqual(service.predictor.received[1].shape, (17, 3))
        self.assertEqual(service.latest("worker01")["activity"], "carrying")

    def test_status_describes_streaming_configuration(self):
        service = GRUInferenceService()
        service.predictor = FakePredictor()

        status = service.status()

        self.assertTrue(status["loaded"])
        self.assertEqual(status["device"], "cpu")
        self.assertEqual(status["minimum_observations"], 5)
        self.assertEqual(status["reset_gap_seconds"], 2.0)

    def test_inference_error_is_contained(self):
        class FailingPredictor(FakePredictor):
            def update(self, _worker_id, _values):
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


if __name__ == "__main__":
    unittest.main()
