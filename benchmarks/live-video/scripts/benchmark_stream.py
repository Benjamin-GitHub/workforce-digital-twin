#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import psutil
from ultralytics import YOLO


def pi_status(command: str) -> str | None:
    try:
        result = subprocess.run(
            ["vcgencmd", command],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    process = psutil.Process()

    class_counts: Counter[str] = Counter()

    frame_intervals_ms: list[float] = []
    inference_times_ms: list[float] = []
    preprocess_times_ms: list[float] = []
    postprocess_times_ms: list[float] = []

    peak_rss = process.memory_info().rss
    processed_frames = 0

    start_temp = pi_status("measure_temp")
    start_throttled = pi_status("get_throttled")

    started = time.perf_counter()
    previous = started

    results = model.predict(
        source=args.source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device="cpu",
        stream_buffer=False,
        save=False,
        verbose=False,
    )

    for result in results:
        now = time.perf_counter()

        if now - started >= args.duration:
            break

        processed_frames += 1
        peak_rss = max(peak_rss, process.memory_info().rss)

        if processed_frames > args.warmup_frames:
            frame_intervals_ms.append((now - previous) * 1000.0)

            speed = result.speed or {}
            preprocess_times_ms.append(float(speed.get("preprocess", 0.0)))
            inference_times_ms.append(float(speed.get("inference", 0.0)))
            postprocess_times_ms.append(float(speed.get("postprocess", 0.0)))

        previous = now

        if result.boxes is not None:
            for class_id in result.boxes.cls.cpu().tolist():
                class_counts[result.names[int(class_id)]] += 1

    total_seconds = time.perf_counter() - started

    end_temp = pi_status("measure_temp")
    end_throttled = pi_status("get_throttled")

    summary = {
        "name": args.name,
        "model": args.model,
        "source": args.source,
        "requested_duration_seconds": args.duration,
        "actual_duration_seconds": total_seconds,
        "processed_frames": processed_frames,
        "processed_fps": (
            processed_frames / total_seconds if total_seconds > 0 else None
        ),
        "warmup_frames_excluded": args.warmup_frames,
        "mean_frame_interval_ms": (
            mean(frame_intervals_ms) if frame_intervals_ms else None
        ),
        "median_frame_interval_ms": (
            median(frame_intervals_ms) if frame_intervals_ms else None
        ),
        "p95_frame_interval_ms": percentile(frame_intervals_ms, 0.95),
        "mean_preprocess_ms": (
            mean(preprocess_times_ms) if preprocess_times_ms else None
        ),
        "mean_inference_ms": (
            mean(inference_times_ms) if inference_times_ms else None
        ),
        "median_inference_ms": (
            median(inference_times_ms) if inference_times_ms else None
        ),
        "p95_inference_ms": percentile(inference_times_ms, 0.95),
        "inference_sd_ms": (
            pstdev(inference_times_ms)
            if len(inference_times_ms) > 1
            else 0.0
        ),
        "mean_postprocess_ms": (
            mean(postprocess_times_ms) if postprocess_times_ms else None
        ),
        "peak_rss_mb": peak_rss / (1024 * 1024),
        "start_temperature": start_temp,
        "end_temperature": end_temp,
        "start_throttled": start_throttled,
        "end_throttled": end_throttled,
        "class_counts": dict(class_counts),
    }

    summary_path = output_dir / f"{args.name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    frame_csv = output_dir / f"{args.name}_frames.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "frame_interval_ms",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
            ]
        )

        for index, row in enumerate(
            zip(
                frame_intervals_ms,
                preprocess_times_ms,
                inference_times_ms,
                postprocess_times_ms,
            ),
            start=1,
        ):
            writer.writerow([index, *row])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
