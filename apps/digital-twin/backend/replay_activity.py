#!/usr/bin/env python3
"""Replay Raspberry Pi activity CSV output into the Digital Twin API."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REQUIRED_COLUMNS = {"frame", "track_id", "confidence"}
ACTIVITY_COLUMNS = ("semantic_activity", "smoothed_activity", "raw_activity")
CADENCE_COLUMNS = ("video_time_sec", "timestamp", "frame")
DEFAULT_SOURCE_CANDIDATES = (
    Path("results/activity/activity_final.csv"),
    Path("results/activity/controlled_5min/activity_final.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Raspberry Pi activity CSV frames into POST /workers at their "
            "recorded cadence."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Activity CSV (defaults to the first known activity_final.csv that exists)",
    )
    parser.add_argument("--worker-id", default="worker01")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="Replay multiplier; 10 replays ten times faster (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=nonnegative_float,
        help="Optional inclusive start time in source video seconds",
    )
    parser.add_argument(
        "--end",
        type=nonnegative_float,
        help="Optional inclusive end time in source video seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mapped frames without posting or waiting",
    )
    args = parser.parse_args()
    if args.start is not None and args.end is not None and args.end < args.start:
        parser.error("--end must be greater than or equal to --start")
    return args


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return number


def find_default_source(repo_root: Path) -> Path:
    for relative_path in DEFAULT_SOURCE_CANDIDATES:
        candidate = repo_root / relative_path
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(repo_root / path) for path in DEFAULT_SOURCE_CANDIDATES)
    raise FileNotFoundError(f"No default activity CSV found; searched: {searched}")


def read_rows(source: Path) -> list[dict[str, str]]:
    with source.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        if not any(column in headers for column in ACTIVITY_COLUMNS):
            raise ValueError(
                "CSV needs one activity column: " + ", ".join(ACTIVITY_COLUMNS)
            )
        if not any(column in headers for column in CADENCE_COLUMNS):
            raise ValueError(
                "CSV needs one cadence column: " + ", ".join(CADENCE_COLUMNS)
            )
        return list(reader)


def source_time(row: dict[str, str], fps: float) -> float:
    if row.get("video_time_sec"):
        return float(row["video_time_sec"])
    if row.get("timestamp"):
        return float(row["timestamp"])
    return float(row["frame"]) / fps


def activity_for(row: dict[str, str]) -> str:
    for column in ACTIVITY_COLUMNS:
        value = row.get(column, "").strip()
        if value:
            return value
    return "unknown"


def infer_fps(rows: list[dict[str, str]]) -> float:
    samples: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        try:
            frame_delta = float(current["frame"]) - float(previous["frame"])
            if previous.get("video_time_sec") and current.get("video_time_sec"):
                time_delta = float(current["video_time_sec"]) - float(
                    previous["video_time_sec"]
                )
                if frame_delta > 0 and time_delta > 0:
                    samples.append(frame_delta / time_delta)
        except (KeyError, TypeError, ValueError):
            continue
    return sum(samples) / len(samples) if samples else 11.0


def selected_rows(
    rows: Iterable[dict[str, str]], fps: float, start: float | None, end: float | None
) -> Iterator[tuple[float, dict[str, str]]]:
    origin: float | None = None
    for row in rows:
        cadence = source_time(row, fps)
        if origin is None:
            origin = cadence
        video_seconds = (
            float(row["video_time_sec"])
            if row.get("video_time_sec")
            else cadence - origin
        )
        if start is not None and video_seconds < start:
            continue
        if end is not None and video_seconds > end:
            break
        yield cadence, row


def optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def map_worker_state(
    row: dict[str, str], worker_id: str, fps: float
) -> dict[str, object]:
    activity = activity_for(row)
    confidence = float(row.get("confidence") or 0.0)
    timestamp = row.get("timestamp", "")
    try:
        timestamp_iso = datetime.fromtimestamp(float(timestamp)).astimezone().isoformat()
    except (OSError, OverflowError, ValueError):
        timestamp_iso = datetime.now().astimezone().isoformat()

    return {
        "worker_id": worker_id,
        "timestamp": timestamp_iso,
        "tracking": {
            "track_id": optional_int(row.get("track_id")),
            "camera_id": "raspberry_pi_replay",
            "online": True,
        },
        "ppe": {"helmet": None, "vest": None, "gloves": None, "boots": None},
        "activity": {
            "baseline": activity,
            "baseline_confidence": confidence,
            "stgcn": "unknown",
            "stgcn_confidence": 0.0,
            "display_activity": activity,
        },
        "edge": {"fps": fps, "cpu_temperature": None, "throttled": False},
    }


def post_worker(api_url: str, payload: dict[str, object]) -> None:
    endpoint = api_url.rstrip("/") + "/workers"
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"POST {endpoint} returned HTTP {response.status}")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {endpoint} returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {endpoint}: {error.reason}") from error


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    source = args.source or find_default_source(repo_root)
    rows = read_rows(source)
    if not rows:
        raise ValueError(f"CSV has no data rows: {source}")
    fps = infer_fps(rows)
    selected = list(selected_rows(rows, fps, args.start, args.end))
    print(f"Source: {source}")
    print(f"Rows: {len(rows)}; frames selected: {len(selected)}; FPS: {fps:.2f}")

    previous_cadence: float | None = None
    for index, (cadence, row) in enumerate(selected, start=1):
        if not args.dry_run and previous_cadence is not None:
            time.sleep(max(0.0, cadence - previous_cadence) / args.speed)
        payload = map_worker_state(row, args.worker_id, fps)
        if args.dry_run:
            print(json.dumps(payload, separators=(",", ":")))
        else:
            post_worker(args.api_url, payload)
            print(
                f"[{index}/{len(selected)}] frame={row['frame']} "
                f"activity={payload['activity']['display_activity']}"
            )
        previous_cadence = cadence
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
