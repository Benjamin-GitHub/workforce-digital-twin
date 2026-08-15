import unittest
from datetime import datetime, timezone

from app.models import PoseKeypoint, PoseState
from app.stgcn import STGCNInferenceService, TemporalPoseBuffer, normalize_pose


def pose(frame_number: int, offset_x: float = 0.0, scale: float = 1.0) -> PoseState:
    points = [PoseKeypoint(x=offset_x + scale * i, y=scale * (i + 2), confidence=0.8) for i in range(17)]
    points[5] = PoseKeypoint(x=offset_x, y=0.0, confidence=0.9)
    points[6] = PoseKeypoint(x=offset_x + 2.0 * scale, y=0.0, confidence=0.9)
    points[11] = PoseKeypoint(x=offset_x, y=2.0 * scale, confidence=0.9)
    points[12] = PoseKeypoint(x=offset_x + 2.0 * scale, y=2.0 * scale, confidence=0.9)
    return PoseState(
        frame_number=frame_number,
        captured_at=datetime.now(timezone.utc),
        image_width=640,
        image_height=480,
        keypoints=points,
    )


class NormalizationTests(unittest.TestCase):
    def test_body_centre_and_scale_are_translation_and_size_invariant(self):
        first = normalize_pose(pose(1, offset_x=0.0, scale=1.0))
        second = normalize_pose(pose(2, offset_x=100.0, scale=5.0))
        for first_joint, second_joint in zip(first, second):
            self.assertAlmostEqual(first_joint[0], second_joint[0])
            self.assertAlmostEqual(first_joint[1], second_joint[1])
            self.assertEqual(first_joint[2], second_joint[2])
        self.assertEqual(first[11][:2], [-0.5, 0.0])
        self.assertEqual(first[12][:2], [0.5, 0.0])

    def test_missing_joint_is_zeroed_and_confidence_is_retained(self):
        sample = pose(1)
        sample.keypoints[0] = PoseKeypoint(x=999.0, y=999.0, confidence=None)
        normalized = normalize_pose(sample)
        self.assertEqual(normalized[0], [0.0, 0.0, 0.0])
        self.assertEqual(normalized[5][2], 0.9)


class TemporalPoseBufferTests(unittest.TestCase):
    def test_ready_at_fixed_window_and_tensor_has_nctvm_shape(self):
        buffer = TemporalPoseBuffer(window_size=3)
        for frame_number in range(3):
            self.assertTrue(buffer.add("worker01", pose(frame_number)))
        diagnostic = buffer.diagnostic("worker01")
        self.assertTrue(diagnostic.ready)
        self.assertEqual(diagnostic.tensor_shape, [1, 3, 3, 17, 1])
        tensor = buffer.tensor("worker01")
        self.assertEqual(len(tensor), 1)
        self.assertEqual(len(tensor[0]), 3)
        self.assertEqual(len(tensor[0][0]), 3)
        self.assertEqual(len(tensor[0][0][0]), 17)
        self.assertEqual(len(tensor[0][0][0][0]), 1)

    def test_duplicate_or_older_frame_is_ignored(self):
        buffer = TemporalPoseBuffer(window_size=2)
        self.assertTrue(buffer.add("worker01", pose(10)))
        self.assertFalse(buffer.add("worker01", pose(10)))
        self.assertFalse(buffer.add("worker01", pose(9)))
        self.assertEqual(buffer.diagnostic("worker01").frames_collected, 1)


class InferenceServiceTests(unittest.TestCase):
    def test_training_time_face_mask_and_binary_confidence_are_applied(self):
        import torch

        class CapturingModel:
            def __init__(self):
                self.received = None

            def __call__(self, tensor):
                self.received = tensor.detach().cpu()
                return torch.tensor([[0.0, 0.0, 0.0, 4.0, 0.0, 0.0]])

        service = STGCNInferenceService()
        service.model = CapturingModel()
        service.device = torch.device("cpu")
        service.classes = ["walking", "standing", "idle", "bending", "carrying", "material_handling"]
        values = torch.full((1, 3, 32, 17, 1), 0.7).tolist()

        result = service.predict("worker01", values)

        self.assertEqual(result.activity, "bending")
        self.assertGreater(result.confidence, 0.9)
        self.assertTrue(torch.all(service.model.received[:, :, :, 1:5, :] == 0))
        self.assertTrue(torch.all(service.model.received[:, 2, :, [0, *range(5, 17)], :] == 1))

    def test_inference_error_is_contained(self):
        import torch

        class FailingModel:
            def __call__(self, _tensor):
                raise RuntimeError("test failure")

        service = STGCNInferenceService()
        service.model = FailingModel()
        service.device = torch.device("cpu")
        self.assertIsNone(service.predict("worker01", torch.zeros(1, 3, 32, 17, 1).tolist()))
        self.assertIn("test failure", service.status()["error"])


if __name__ == "__main__":
    unittest.main()
