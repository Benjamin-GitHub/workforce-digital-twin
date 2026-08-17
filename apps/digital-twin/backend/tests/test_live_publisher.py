import threading
import time
import unittest

from src.digital_twin.publisher import WorkerStatePublisher, build_worker_state


class LivePublisherTests(unittest.TestCase):
    def test_payload_matches_worker_state_fields_without_unavailable_ppe(self):
        payload = build_worker_state(
            worker_id="worker01",
            track_id=7,
            camera_id="pi_cam_01",
            activity="walking",
            confidence=0.82,
            frame_number=42,
            image_width=640,
            image_height=480,
            keypoints=[[float(i), float(i + 1)] for i in range(17)],
            keypoint_confidences=[0.9] * 17,
            fps=10.5,
        )

        self.assertEqual(payload["tracking"]["track_id"], 7)
        self.assertEqual(payload["activity"]["baseline"], "walking")
        self.assertEqual(payload["activity"]["baseline_confidence"], 0.82)
        self.assertEqual(payload["edge"]["fps"], 10.5)
        self.assertNotIn("ppe", payload)
        self.assertEqual(payload["pose"]["frame_number"], 42)
        self.assertEqual(payload["pose"]["coordinate_space"], "image_pixels")
        self.assertEqual(payload["pose"]["layout"], "coco_17")
        self.assertEqual(len(payload["pose"]["keypoints"]), 17)
        self.assertEqual(payload["pose"]["keypoints"][5], {
            "x": 5.0, "y": 6.0, "confidence": 0.9
        })

    def test_payload_includes_model_ppe_when_supplied(self):
        ppe = {
            "helmet": {"detected": True, "confidence": 0.84},
            "vest": {"detected": None, "confidence": None},
            "gloves": {"detected": None, "confidence": None},
            "boots": {"detected": None, "confidence": None},
            "observed_at": "2026-08-16T12:00:00+00:00",
            "association_method": "ppe_box_within_pose_person",
        }
        payload = build_worker_state(
            worker_id="worker01", track_id=7, camera_id="pi_cam_01",
            activity="standing", confidence=0.7, frame_number=1,
            image_width=640, image_height=480,
            keypoints=[[0.0, 0.0]] * 17,
            keypoint_confidences=[0.9] * 17, ppe=ppe,
        )
        self.assertEqual(payload["ppe"], ppe)

    def test_transport_runs_off_the_calling_thread_and_errors_are_contained(self):
        called = threading.Event()
        transport_thread = []

        def failing_transport(_url, _payload, _timeout):
            transport_thread.append(threading.current_thread().name)
            called.set()
            raise RuntimeError("backend unavailable")

        publisher = WorkerStatePublisher(
            "http://127.0.0.1:8000",
            interval=0.01,
            timeout=0.01,
            transport=failing_transport,
        )
        started = time.monotonic()
        accepted = publisher.submit({"worker_id": "worker01"})
        submit_duration = time.monotonic() - started

        self.assertTrue(accepted)
        self.assertLess(submit_duration, 0.05)
        self.assertTrue(called.wait(0.5))
        self.assertEqual(transport_thread, ["digital-twin-publisher"])
        publisher.close()


if __name__ == "__main__":
    unittest.main()
