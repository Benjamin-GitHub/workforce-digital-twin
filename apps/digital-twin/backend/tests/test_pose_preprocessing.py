import unittest
from datetime import datetime, timezone

import numpy as np

from app.models import PoseKeypoint, PoseState
from app.stgcn import normalize_pose
from training.gru.gru_data import normalize_live_coco17


def empty_pose() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


def joint(raw: np.ndarray, index: int, x: float, y: float, confidence: float = 0.9) -> None:
    raw[index] = (x, y, confidence)


def stgcn_features(raw: np.ndarray) -> np.ndarray:
    pose = PoseState(
        frame_number=1,
        captured_at=datetime.now(timezone.utc),
        image_width=640,
        image_height=480,
        keypoints=[
            PoseKeypoint(x=float(x), y=float(y), confidence=float(confidence))
            for x, y, confidence in raw
        ],
    )
    return np.asarray(normalize_pose(pose), dtype=np.float32)


def gru_features(raw: np.ndarray) -> np.ndarray:
    return normalize_live_coco17(raw).reshape(3, 17).T


def full_body_pose() -> np.ndarray:
    raw = empty_pose()
    joint(raw, 5, -1.0, 0.0)
    joint(raw, 6, 1.0, 0.0)
    joint(raw, 10, 2.0, 2.0)
    joint(raw, 11, -1.0, 2.0)
    joint(raw, 12, 1.0, 2.0)
    joint(raw, 13, -1.0, 4.0)
    joint(raw, 14, 1.0, 4.0)
    joint(raw, 15, -1.0, 6.0)
    joint(raw, 16, 1.0, 6.0)
    return raw


class LivePosePreprocessingTests(unittest.TestCase):
    def assert_pipeline_parity(self, raw: np.ndarray) -> None:
        np.testing.assert_allclose(
            stgcn_features(raw),
            gru_features(raw),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_confidence_threshold_boundary(self):
        raw = full_body_pose()
        joint(raw, 7, 123.0, 456.0, 0.2999)
        joint(raw, 8, 2.0, 2.0, 0.30)

        features = stgcn_features(raw)

        np.testing.assert_array_equal(features[7], np.zeros(3, dtype=np.float32))
        self.assertEqual(features[8, 2], 1.0)
        self.assertFalse(np.allclose(features[8, :2], 0.0))
        self.assert_pipeline_parity(raw)

    def test_low_confidence_knees_and_ankles_are_completely_zeroed(self):
        raw = full_body_pose()
        for index in (13, 14, 15, 16):
            joint(raw, index, 999.0 + index, -999.0 - index, 0.1)

        features = stgcn_features(raw)

        np.testing.assert_array_equal(
            features[[13, 14, 15, 16]], np.zeros((4, 3), dtype=np.float32)
        )
        self.assert_pipeline_parity(raw)

    def test_full_body_uses_hip_centre_and_torso_scale(self):
        raw = full_body_pose()
        features = stgcn_features(raw)

        np.testing.assert_allclose(features[11], [-0.5, 0.0, 1.0])
        np.testing.assert_allclose(features[12], [0.5, 0.0, 1.0])
        np.testing.assert_allclose(features[5], [-0.5, -1.0, 1.0])
        np.testing.assert_allclose(features[10], [1.0, 0.0, 1.0])
        self.assert_pipeline_parity(raw)

    def test_missing_hips_uses_shoulder_centre_and_visible_extent(self):
        raw = empty_pose()
        joint(raw, 5, -1.0, 0.0)
        joint(raw, 6, 1.0, 0.0)
        joint(raw, 10, 1.0, 2.0)

        features = stgcn_features(raw)
        scale = np.sqrt(8.0)

        np.testing.assert_allclose(features[5, :2], [-1.0 / scale, 0.0])
        np.testing.assert_allclose(features[6, :2], [1.0 / scale, 0.0])
        np.testing.assert_allclose(features[10, :2], [1.0 / scale, 2.0 / scale])
        self.assert_pipeline_parity(raw)

    def test_missing_hips_and_shoulders_uses_visible_centroid(self):
        raw = empty_pose()
        joint(raw, 7, 0.0, 0.0)
        joint(raw, 8, 2.0, 0.0)

        features = stgcn_features(raw)

        np.testing.assert_allclose(features[7], [-0.5, 0.0, 1.0])
        np.testing.assert_allclose(features[8], [0.5, 0.0, 1.0])
        self.assert_pipeline_parity(raw)

    def test_degenerate_extent_falls_back_to_one(self):
        raw = empty_pose()
        joint(raw, 7, 0.0, 0.0)
        joint(raw, 8, 5e-7, 0.0)

        features = stgcn_features(raw)

        np.testing.assert_allclose(features[7, :2], [-2.5e-7, 0.0], atol=1e-10)
        np.testing.assert_allclose(features[8, :2], [2.5e-7, 0.0], atol=1e-10)
        self.assert_pipeline_parity(raw)

    def test_all_missing_pose_is_zero(self):
        raw = np.full((17, 3), (100.0, 200.0, 0.2999), dtype=np.float32)

        np.testing.assert_array_equal(stgcn_features(raw), np.zeros((17, 3)))
        np.testing.assert_array_equal(gru_features(raw), np.zeros((17, 3)))

    def test_extreme_facial_coordinates_cannot_affect_centre_or_scale(self):
        baseline = empty_pose()
        joint(baseline, 11, -1.0, 0.0)
        joint(baseline, 12, 1.0, 0.0)
        joint(baseline, 10, 1.0, 2.0)
        extreme = baseline.copy()
        for index in range(1, 5):
            joint(extreme, index, 10000.0 * index, -5000.0 * index)

        np.testing.assert_allclose(stgcn_features(extreme), stgcn_features(baseline))
        np.testing.assert_array_equal(stgcn_features(extreme)[1:5], np.zeros((4, 3)))
        self.assert_pipeline_parity(extreme)

    def test_parity_across_representative_visibility_patterns(self):
        missing_hips = full_body_pose()
        missing_hips[[11, 12]] = 0.0
        centroid = empty_pose()
        joint(centroid, 7, -3.0, 4.0)
        joint(centroid, 9, 5.0, -2.0)
        low_confidence = full_body_pose()
        joint(low_confidence, 15, 800.0, 900.0, 0.01)

        for name, raw in (
            ("full_body", full_body_pose()),
            ("missing_hips", missing_hips),
            ("centroid", centroid),
            ("low_confidence", low_confidence),
        ):
            with self.subTest(name=name):
                self.assert_pipeline_parity(raw)


if __name__ == "__main__":
    unittest.main()
