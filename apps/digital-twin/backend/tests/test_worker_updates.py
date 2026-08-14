import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.main import update_worker
from app.models import ActivityState, WorkerState
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


if __name__ == "__main__":
    unittest.main()
