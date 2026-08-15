import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.main import get_stgcn_sequence_diagnostic, get_worker_pose, update_worker
from app.models import ActivityState, PoseKeypoint, PoseState, WorkerState
from app.state import worker_state_manager
from app.stgcn import STGCNPrediction, stgcn_service, temporal_pose_buffer


class WorkerUpdateTests(unittest.TestCase):
    def setUp(self):
        worker_state_manager._workers.clear()
        temporal_pose_buffer.clear()
        stgcn_service.clear_predictions()

    def test_worker_source_defaults_to_live(self):
        self.assertEqual(self.worker("walking").source, "live")

    @staticmethod
    def worker(activity: str) -> WorkerState:
        return WorkerState(
            worker_id="transition-test",
            activity=ActivityState(
                baseline=activity,
                baseline_confidence=0.8,
                display_activity=activity,
            ),
        )

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    def test_history_is_saved_only_on_activity_transition(self, save_event, _broadcast):
        asyncio.run(update_worker(self.worker("walking")))
        asyncio.run(update_worker(self.worker("walking")))
        asyncio.run(update_worker(self.worker("standing")))
        self.assertEqual(save_event.call_count, 2)

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    def test_pose_updates_live_state_without_extra_history(self, save_event, _broadcast):
        worker = self.worker("walking")
        asyncio.run(update_worker(worker))
        worker.pose = PoseState(
            frame_number=12,
            captured_at=datetime.now(timezone.utc),
            image_width=640,
            image_height=480,
            keypoints=[
                PoseKeypoint(x=i, y=i + 1, confidence=0.9)
                for i in range(17)
            ],
        )
        asyncio.run(update_worker(worker))

        self.assertEqual(save_event.call_count, 1)
        self.assertEqual(get_worker_pose("transition-test").frame_number, 12)
        self.assertEqual(
            temporal_pose_buffer.diagnostic("transition-test").frames_collected, 1
        )

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    def test_last_stgcn_prediction_survives_a_new_edge_payload(self, _save_event, _broadcast):
        worker = self.worker("walking")
        worker.activity.stgcn = "carrying"
        worker.activity.stgcn_confidence = 0.91
        asyncio.run(update_worker(worker))

        next_worker = self.worker("walking")
        saved = asyncio.run(update_worker(next_worker))

        self.assertEqual(saved.activity.stgcn, "carrying")
        self.assertAlmostEqual(saved.activity.stgcn_confidence, 0.91)

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    def test_diagnostic_endpoint_reports_ready_shape(self, _save_event, _broadcast):
        worker = self.worker("walking")
        for frame_number in range(32):
            worker.pose = PoseState(
                frame_number=frame_number,
                captured_at=datetime.now(timezone.utc),
                image_width=640,
                image_height=480,
                keypoints=[
                    PoseKeypoint(x=i, y=i + 1, confidence=0.9)
                    for i in range(17)
                ],
            )
            asyncio.run(update_worker(worker))

        diagnostic = get_stgcn_sequence_diagnostic("transition-test")
        self.assertTrue(diagnostic.ready)
        self.assertEqual(diagnostic.tensor_shape, [1, 3, 32, 17, 1])

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    @patch("app.main.stgcn_service.predict")
    def test_ready_sequence_updates_stgcn_but_not_display_or_history(
        self, predict, save_event, _broadcast
    ):
        predict.return_value = STGCNPrediction(
            worker_id="transition-test", activity="carrying", confidence=0.93,
            probabilities={"carrying": 0.93},
        )
        worker = self.worker("walking")
        for frame_number in range(32):
            worker.pose = PoseState(
                frame_number=frame_number, captured_at=datetime.now(timezone.utc),
                image_width=640, image_height=480,
                keypoints=[PoseKeypoint(x=i, y=i + 1, confidence=0.9) for i in range(17)],
            )
            worker = asyncio.run(update_worker(worker))

        self.assertEqual(worker.activity.stgcn, "carrying")
        self.assertAlmostEqual(worker.activity.stgcn_confidence, 0.93)
        self.assertEqual(worker.activity.display_activity, "walking")
        self.assertEqual(save_event.call_count, 1)
        predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
