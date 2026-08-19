import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from app.main import (
    get_stgcn_sequence_diagnostic,
    get_model_input_diagnostic,
    get_worker_pose,
    get_worker_ppe,
    load_activity_models,
    reset_worker_models,
    start_session,
    update_worker,
)
from app.gru import gru_service
from app.model_input import model_input_coordinator
from app.models import (
    ActivityState,
    PPEObservation,
    PPEState,
    PoseKeypoint,
    PoseState,
    SessionStartRequest,
    TrackingState,
    WorkerState,
)
from app.state import worker_state_manager
from app.stgcn import STGCNPrediction, stgcn_service, temporal_pose_buffer


class WorkerUpdateTests(unittest.TestCase):
    def setUp(self):
        worker_state_manager._workers.clear()
        temporal_pose_buffer.clear()
        stgcn_service.clear_predictions()
        gru_service.clear_predictions()
        model_input_coordinator.reset("transition-test", "test_setup")

    def test_worker_source_defaults_to_live(self):
        self.assertEqual(self.worker("walking").source, "live")

    @patch("app.main.gru_service.load")
    @patch("app.main.temporal_pose_buffer.configure")
    @patch("app.main.stgcn_service.load")
    def test_startup_configures_buffer_from_checkpoint_window(
        self, stgcn_load, configure, gru_load
    ):
        with patch.object(stgcn_service, "window_size", 16):
            load_activity_models()

        stgcn_load.assert_called_once_with()
        configure.assert_called_once_with(16)
        gru_load.assert_called_once_with()

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

    @staticmethod
    def posed_worker(
        timestamp: datetime,
        frame_number: int,
        *,
        track_id: int = 1,
        camera_id: str = "camera01",
        source: str = "live",
    ) -> WorkerState:
        worker = WorkerUpdateTests.worker("walking")
        worker.source = source
        worker.tracking = TrackingState(
            track_id=track_id, camera_id=camera_id, online=True
        )
        worker.pose = PoseState(
            frame_number=frame_number,
            captured_at=timestamp,
            image_width=640,
            image_height=480,
            keypoints=[
                PoseKeypoint(x=i, y=i + 1, confidence=0.9)
                for i in range(17)
            ],
        )
        return worker

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
    def test_real_ppe_is_available_without_creating_history(self, save_event, _broadcast):
        worker = self.worker("standing")
        worker.ppe = PPEState(
            helmet=PPEObservation(detected=True, confidence=0.84),
            vest=PPEObservation(detected=True, confidence=0.72),
            observed_at=datetime.now(timezone.utc),
            association_method="ppe_box_within_pose_person",
        )
        asyncio.run(update_worker(worker))
        ppe = get_worker_ppe("transition-test")
        self.assertTrue(ppe.helmet.detected)
        self.assertAlmostEqual(ppe.vest.confidence, 0.72)
        self.assertEqual(save_event.call_count, 1)

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

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    @patch("app.main.stgcn_service.predict")
    @patch("app.main.gru_service.predict", return_value=None)
    def test_gru_executes_when_stgcn_buffer_rejects(
        self, gru_predict, stgcn_predict, _save_event, _broadcast
    ):
        timestamp = datetime.now(timezone.utc)
        first = self.posed_worker(timestamp, 1)
        asyncio.run(update_worker(first))
        gru_predict.reset_mock()

        duplicate_frame = self.posed_worker(
            timestamp + timedelta(milliseconds=100), 1
        )
        asyncio.run(update_worker(duplicate_frame))

        gru_predict.assert_called_once_with("transition-test", duplicate_frame.pose)
        stgcn_predict.assert_not_called()
        self.assertEqual(
            temporal_pose_buffer.diagnostic("transition-test").frames_collected, 1
        )

    @patch("app.main.websocket_manager.broadcast", new_callable=AsyncMock)
    @patch("app.main.save_worker_event")
    def test_identity_reset_clears_predictions_during_warmup(
        self, _save_event, _broadcast
    ):
        timestamp = datetime.now(timezone.utc)
        first = self.posed_worker(timestamp, 1, track_id=1)
        first.activity.stgcn = "carrying"
        first.activity.stgcn_confidence = 0.9
        first.activity.gru = "standing"
        first.activity.gru_confidence = 0.8
        asyncio.run(update_worker(first))

        second = self.posed_worker(timestamp + timedelta(milliseconds=100), 2, track_id=2)
        saved = asyncio.run(update_worker(second))

        self.assertEqual(saved.activity.stgcn, "unknown")
        self.assertEqual(saved.activity.stgcn_confidence, 0.0)
        self.assertEqual(saved.activity.gru, "unknown")
        self.assertEqual(saved.activity.gru_confidence, 0.0)
        self.assertEqual(
            get_model_input_diagnostic("transition-test")["last_decision_reason"],
            "identity_change",
        )

    @patch("app.main.session_recorder.start", return_value={"session_id": "test"})
    def test_session_start_resets_worker_models(self, _start):
        timestamp = datetime.now(timezone.utc)
        worker = self.posed_worker(timestamp, 1)
        worker.activity.stgcn = "carrying"
        worker.activity.stgcn_confidence = 0.9
        worker.activity.gru = "standing"
        worker.activity.gru_confidence = 0.8
        worker_state_manager.set_worker(worker)
        temporal_pose_buffer.add(worker.worker_id, worker.pose)
        model_input_coordinator.evaluate(worker)

        result = start_session(SessionStartRequest(worker_id=worker.worker_id))

        saved = worker_state_manager.get_worker(worker.worker_id)
        self.assertEqual(result, {"session_id": "test"})
        self.assertEqual(saved.activity.stgcn, "unknown")
        self.assertEqual(saved.activity.stgcn_confidence, 0.0)
        self.assertEqual(saved.activity.gru, "unknown")
        self.assertEqual(saved.activity.gru_confidence, 0.0)
        self.assertEqual(temporal_pose_buffer.diagnostic(worker.worker_id).frames_collected, 0)
        diagnostic = get_model_input_diagnostic(worker.worker_id)
        self.assertEqual(diagnostic["last_decision_reason"], "session_start")
        self.assertIsNone(diagnostic["current_identity"])

    def test_coordinated_reset_isolated_to_one_worker(self):
        timestamp = datetime.now(timezone.utc)
        first = self.posed_worker(timestamp, 1)
        second = self.posed_worker(timestamp, 1)
        second.worker_id = "other-worker"
        for worker in (first, second):
            worker.activity.stgcn = "carrying"
            worker.activity.gru = "standing"
            worker_state_manager.set_worker(worker)
            temporal_pose_buffer.add(worker.worker_id, worker.pose)
            model_input_coordinator.evaluate(worker)

        reset_worker_models(first.worker_id, "session_start")

        self.assertEqual(
            temporal_pose_buffer.diagnostic(first.worker_id).frames_collected, 0
        )
        self.assertEqual(
            temporal_pose_buffer.diagnostic(second.worker_id).frames_collected, 1
        )
        self.assertEqual(
            worker_state_manager.get_worker(first.worker_id).activity.stgcn, "unknown"
        )
        self.assertEqual(
            worker_state_manager.get_worker(second.worker_id).activity.stgcn, "carrying"
        )
        self.assertEqual(
            model_input_coordinator.diagnostic(second.worker_id)["accepted_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
