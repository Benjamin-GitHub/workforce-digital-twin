import unittest
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from app.models import PoseKeypoint, PoseState
from app.stgcn import STGCNInferenceService, TemporalPoseBuffer, normalize_pose


REPO_ROOT = Path(__file__).resolve().parents[4]
OLD_CHECKPOINT = REPO_ROOT / "training/stgcn/runs/cml_plus_local_sqrt/best.pt"


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

    def test_missing_joint_is_zeroed_and_present_confidence_is_binary(self):
        sample = pose(1)
        sample.keypoints[0] = PoseKeypoint(x=999.0, y=999.0, confidence=None)
        normalized = normalize_pose(sample)
        self.assertEqual(normalized[0], [0.0, 0.0, 0.0])
        self.assertEqual(normalized[5][2], 1.0)


class TemporalPoseBufferTests(unittest.TestCase):
    def test_ready_at_fixed_window_and_tensor_has_nctvm_shape(self):
        for window_size in (16, 32):
            with self.subTest(window_size=window_size):
                buffer = TemporalPoseBuffer(window_size=window_size)
                for frame_number in range(window_size):
                    self.assertTrue(buffer.add("worker01", pose(frame_number)))
                diagnostic = buffer.diagnostic("worker01")
                self.assertTrue(diagnostic.ready)
                self.assertEqual(diagnostic.tensor_shape, [1, 3, window_size, 17, 1])
                tensor = buffer.tensor("worker01")
                self.assertEqual(len(tensor), 1)
                self.assertEqual(len(tensor[0]), 3)
                self.assertEqual(len(tensor[0][0]), window_size)
                self.assertEqual(len(tensor[0][0][0]), 17)
                self.assertEqual(len(tensor[0][0][0][0]), 1)

    def test_duplicate_or_older_frame_is_ignored(self):
        buffer = TemporalPoseBuffer(window_size=2)
        self.assertTrue(buffer.add("worker01", pose(10)))
        self.assertFalse(buffer.add("worker01", pose(10)))
        self.assertFalse(buffer.add("worker01", pose(9)))
        self.assertEqual(buffer.diagnostic("worker01").frames_collected, 1)

    def test_reconfigure_clears_frames_and_last_frame_tracking(self):
        buffer = TemporalPoseBuffer(window_size=32)
        self.assertTrue(buffer.add("worker01", pose(10)))

        buffer.configure(16)

        self.assertEqual(buffer.window_size, 16)
        self.assertEqual(buffer.diagnostic("worker01").frames_collected, 0)
        self.assertTrue(buffer.add("worker01", pose(1)))

        buffer.configure(16)
        self.assertEqual(buffer.diagnostic("worker01").frames_collected, 0)
        self.assertTrue(buffer.add("worker01", pose(0)))

    def test_reconfigure_requires_positive_window_size(self):
        buffer = TemporalPoseBuffer()
        with self.assertRaisesRegex(ValueError, "window_size must be positive"):
            buffer.configure(0)

    def test_clear_one_worker_preserves_other_workers(self):
        buffer = TemporalPoseBuffer(window_size=2)
        buffer.add("worker01", pose(1))
        buffer.add("worker02", pose(1))

        buffer.clear("worker01")

        self.assertEqual(buffer.diagnostic("worker01").frames_collected, 0)
        self.assertEqual(buffer.diagnostic("worker02").frames_collected, 1)


class InferenceServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.new_checkpoint = Path(cls._temporary_directory.name) / "stgcn_5hz_w16.pt"
        checkpoint = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
        checkpoint["input_shape"] = [3, 16, 17, 1]
        checkpoint["config"] = {**checkpoint["config"], "source_pose_hz": 5.0}
        torch.save(checkpoint, cls.new_checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    def test_old_and_new_checkpoints_define_window_size(self):
        cases = (
            (OLD_CHECKPOINT, 32, None),
            (self.new_checkpoint, 16, 5.0),
        )
        for checkpoint, window_size, source_pose_hz in cases:
            with self.subTest(checkpoint=checkpoint.name, window_size=window_size):
                service = STGCNInferenceService(checkpoint)
                service.load()

                self.assertIsNotNone(service.model, service.error)
                self.assertEqual(service.window_size, window_size)
                self.assertEqual(service.source_pose_hz, source_pose_hz)
                self.assertEqual(service.status()["checkpoint"], str(checkpoint.resolve()))

    def test_environment_checkpoint_override_is_resolved(self):
        with patch.dict(os.environ, {"STGCN_CHECKPOINT": str(self.new_checkpoint)}):
            service = STGCNInferenceService()

        self.assertEqual(service.checkpoint_path, self.new_checkpoint.resolve())
        self.assertEqual(service.status()["checkpoint"], str(self.new_checkpoint.resolve()))

    def test_training_time_face_mask_and_binary_confidence_are_applied(self):
        import torch

        class CapturingModel:
            def __init__(self):
                self.received = None

            def __call__(self, tensor):
                self.received = tensor.detach().cpu()
                return torch.tensor([[0.0, 0.0, 0.0, 4.0, 0.0, 0.0]])

        for window_size in (16, 32):
            with self.subTest(window_size=window_size):
                service = STGCNInferenceService()
                service.model = CapturingModel()
                service.device = torch.device("cpu")
                service.classes = ["walking", "standing", "idle", "bending", "carrying", "material_handling"]
                service.window_size = window_size
                values = torch.full((1, 3, window_size, 17, 1), 0.7).tolist()

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
