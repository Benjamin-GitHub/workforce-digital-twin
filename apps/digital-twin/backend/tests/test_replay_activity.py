import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "replay_activity.py"
SPEC = importlib.util.spec_from_file_location("replay_activity", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)


class ReplayActivityTests(unittest.TestCase):
    def test_real_csv_headers_and_mapping(self):
        repo_root = MODULE_PATH.parents[3]
        source = replay.find_default_source(repo_root)
        rows = replay.read_rows(source)
        fps = replay.infer_fps(rows)
        payload = replay.map_worker_state(rows[1], "worker-test", fps)

        self.assertEqual(source.name, "activity_final.csv")
        self.assertEqual(payload["worker_id"], "worker-test")
        self.assertEqual(payload["activity"]["display_activity"], "carrying")
        self.assertEqual(payload["tracking"]["track_id"], 1)
        self.assertAlmostEqual(fps, 11.0, places=1)

    def test_repeated_activities_are_preserved_for_live_updates(self):
        rows = [
            self.row("1", "0.0", "walking"),
            self.row("2", "0.1", "walking"),
            self.row("3", "0.2", "standing"),
            self.row("4", "0.3", "standing"),
            self.row("5", "0.4", "walking"),
        ]
        selected = list(replay.selected_rows(rows, 10.0, None, None))
        self.assertEqual(
            [row["frame"] for _, row in selected], ["1", "2", "3", "4", "5"]
        )

    def test_start_and_end_use_video_seconds(self):
        rows = [
            self.row("1", "0.0", "idle"),
            self.row("2", "1.0", "walking"),
            self.row("3", "2.0", "standing"),
        ]
        selected = list(replay.selected_rows(rows, 1.0, 1.0, 1.5))
        self.assertEqual([row["frame"] for _, row in selected], ["2"])

    def test_rejects_missing_activity_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.csv"
            source.write_text("frame,track_id,confidence,timestamp\n1,1,0.9,1\n")
            with self.assertRaisesRegex(ValueError, "activity column"):
                replay.read_rows(source)

    @staticmethod
    def row(frame: str, video_time: str, activity: str) -> dict[str, str]:
        return {
            "frame": frame,
            "video_time_sec": video_time,
            "timestamp": str(1_700_000_000 + float(video_time)),
            "track_id": "1",
            "semantic_activity": activity,
            "confidence": "0.8",
        }


if __name__ == "__main__":
    unittest.main()
