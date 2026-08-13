#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PPE_CLASSES = [
    "helmet",
    "vest",
    "gloves",
    "boots",
    "no_helmet",
    "no_gloves",
    "no_goggle",
    "none",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export frame-level PPE predictions from a video."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model, task="detect")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps:.3f}")
    print(f"Frames: {total_frames}")
    print("Starting PPE prediction export...")

    fieldnames = [
        "frame",
        "time_sec",
        "person_count",

        "helmet_count",
        "helmet_conf",
        "helmet_present",

        "vest_count",
        "vest_conf",
        "vest_present",

        "gloves_count",
        "gloves_conf",
        "gloves_present",

        "boots_count",
        "boots_conf",
        "boots_present",

        "no_helmet_count",
        "no_helmet_conf",

        "no_gloves_count",
        "no_gloves_conf",

        "no_goggle_count",
        "no_goggle_conf",

        "none_count",
        "none_conf",
    ]

    start_time = time.perf_counter()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        frame_idx = 0

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )

            result = results[0]

            counts = {}
            max_conf = {}

            for cls_name in ["Person"] + PPE_CLASSES:
                counts[cls_name] = 0
                max_conf[cls_name] = 0.0

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = result.names[class_id]

                    if class_name not in counts:
                        counts[class_name] = 0
                        max_conf[class_name] = 0.0

                    counts[class_name] += 1

                    if confidence > max_conf[class_name]:
                        max_conf[class_name] = confidence

            timestamp = frame_idx / fps

            writer.writerow(
                {
                    "frame": frame_idx,
                    "time_sec": f"{timestamp:.6f}",

                    "person_count": counts.get("Person", 0),

                    "helmet_count": counts.get("helmet", 0),
                    "helmet_conf": f"{max_conf.get('helmet', 0.0):.6f}",
                    "helmet_present": int(counts.get("helmet", 0) > 0),

                    "vest_count": counts.get("vest", 0),
                    "vest_conf": f"{max_conf.get('vest', 0.0):.6f}",
                    "vest_present": int(counts.get("vest", 0) > 0),

                    "gloves_count": counts.get("gloves", 0),
                    "gloves_conf": f"{max_conf.get('gloves', 0.0):.6f}",
                    "gloves_present": int(counts.get("gloves", 0) > 0),

                    "boots_count": counts.get("boots", 0),
                    "boots_conf": f"{max_conf.get('boots', 0.0):.6f}",
                    "boots_present": int(counts.get("boots", 0) > 0),

                    "no_helmet_count": counts.get("no_helmet", 0),
                    "no_helmet_conf": f"{max_conf.get('no_helmet', 0.0):.6f}",

                    "no_gloves_count": counts.get("no_gloves", 0),
                    "no_gloves_conf": f"{max_conf.get('no_gloves', 0.0):.6f}",

                    "no_goggle_count": counts.get("no_goggle", 0),
                    "no_goggle_conf": f"{max_conf.get('no_goggle', 0.0):.6f}",

                    "none_count": counts.get("none", 0),
                    "none_conf": f"{max_conf.get('none', 0.0):.6f}",
                }
            )

            frame_idx += 1

            # Lightweight progress only every 100 frames
            if frame_idx % 100 == 0 or frame_idx == total_frames:

                elapsed = time.perf_counter() - start_time
                processing_fps = frame_idx / elapsed

                print(
                    f"Frame {frame_idx}/{total_frames} "
                    f"| {timestamp:.1f}s "
                    f"| processing {processing_fps:.2f} FPS"
                )

    cap.release()

    elapsed = time.perf_counter() - start_time

    print()
    print("Finished.")
    print(f"Processed frames: {frame_idx}")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
