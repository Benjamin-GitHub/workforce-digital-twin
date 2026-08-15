import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.main import get_worker_pose, update_worker
from app.models import ActivityState, PoseKeypoint, PoseState, WorkerState
from app.state import worker_state_manager


class WorkerUpdateTests(unittest.TestCase):
    def setUp(self):
        worker_state_manager._workers.clear()

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


if __name__ == "__main__":
    unittest.main()
