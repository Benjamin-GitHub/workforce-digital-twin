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
            fps=10.5,
        )

        self.assertEqual(payload["tracking"]["track_id"], 7)
        self.assertEqual(payload["activity"]["baseline"], "walking")
        self.assertEqual(payload["activity"]["baseline_confidence"], 0.82)
        self.assertEqual(payload["edge"]["fps"], 10.5)
        self.assertNotIn("ppe", payload)

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
