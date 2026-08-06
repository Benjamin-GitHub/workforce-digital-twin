#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import psutil
from ultralytics import YOLO


def read_pi_command(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a YOLO model on a recorded video."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_duration = (
        source_frames / source_fps if source_fps and source_fps > 0 else None
    )
    capture.release()

    start_temp = read_pi_command("vcgencmd", "measure_temp")
    start_throttled = read_pi_command("vcgencmd", "get_throttled")

    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss

    model = YOLO(str(model_path))

    wall_times_ms: list[float] = []
    preprocess_times_ms: list[float] = []
    inference_times_ms: list[float] = []
    postprocess_times_ms: list[float] = []

    class_counts: Counter[str] = Counter()
    confidences: defaultdict[str, list[float]] = defaultdict(list)

    processed_frames = 0
    measured_frames = 0

    overall_start = time.perf_counter()

    results = model.predict(
        source=str(source_path),
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device="cpu",
        batch=1,
        vid_stride=args.vid_stride,
        stream_buffer=False,
        save=False,
        verbose=False,
    )

    previous_result_time = time.perf_counter()

    for result in results:
        now = time.perf_counter()
        frame_wall_ms = (now - previous_result_time) * 1000.0
        previous_result_time = now

        processed_frames += 1
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

        if processed_frames > args.warmup_frames:
            measured_frames += 1
            wall_times_ms.append(frame_wall_ms)

            speeds = result.speed or {}
            preprocess_times_ms.append(float(speeds.get("preprocess", 0.0)))
            inference_times_ms.append(float(speeds.get("inference", 0.0)))
            postprocess_times_ms.append(float(speeds.get("postprocess", 0.0)))

        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().tolist()
            box_confidences = result.boxes.conf.cpu().tolist()

            for class_id, confidence in zip(class_ids, box_confidences):
                class_name = result.names[int(class_id)]
                class_counts[class_name] += 1
                confidences[class_name].append(float(confidence))

    overall_seconds = time.perf_counter() - overall_start

    end_temp = read_pi_command("vcgencmd", "measure_temp")
    end_throttled = read_pi_command("vcgencmd", "get_throttled")

    processing_fps = (
        processed_frames / overall_seconds if overall_seconds > 0 else None
    )

    real_time_factor = (
        overall_seconds / source_duration
        if source_duration and source_duration > 0
        else None
    )

    summary = {
        "name": args.name,
        "model": str(model_path),
        "source": str(source_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "vid_stride": args.vid_stride,
        "warmup_frames_excluded": args.warmup_frames,
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": source_fps,
        "source_frames": source_frames,
        "source_duration_seconds": source_duration,
        "processed_frames": processed_frames,
        "measured_frames": measured_frames,
        "total_processing_seconds": overall_seconds,
        "processing_fps": processing_fps,
        "real_time_factor": real_time_factor,
        "mean_frame_wall_ms": mean(wall_times_ms) if wall_times_ms else None,
        "median_frame_wall_ms": median(wall_times_ms) if wall_times_ms else None,
        "p95_frame_wall_ms": percentile(wall_times_ms, 0.95),
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
        "peak_rss_mb": peak_rss_bytes / (1024 * 1024),
        "start_temperature": start_temp,
        "end_temperature": end_temp,
        "start_throttled": start_throttled,
        "end_throttled": end_throttled,
        "python_version": platform.python_version(),
        "class_counts": dict(class_counts),
        "mean_confidence_by_class": {
            class_name: mean(values)
            for class_name, values in confidences.items()
        },
    }

    json_file = output_dir / f"{args.name}_summary.json"
    csv_file = output_dir / f"{args.name}_frames.csv"

    json_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "wall_ms",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
            ]
        )

        for index, values in enumerate(
            zip(
                wall_times_ms,
                preprocess_times_ms,
                inference_times_ms,
                postprocess_times_ms,
            ),
            start=args.warmup_frames + 1,
        ):
            writer.writerow([index, *values])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
