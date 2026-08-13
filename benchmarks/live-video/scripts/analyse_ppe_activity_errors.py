#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path


GROUND_TRUTH = "benchmarks/live-video/ground-truth/esp32_controlled_5min_ground_truth.csv"
ERRORS = "results/controlled-video/ppe_ground_truth_errors.csv"
OUTPUT = "results/controlled-video/ppe_activity_error_rates.csv"

PPE_ITEMS = ["helmet", "vest", "gloves", "boots"]


def load_ground_truth():
    rows = []

    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            duration_sec = float(row["end_sec"]) - float(row["start_sec"])

            # Video is standardised to 11 FPS
            labelled_frames = round(duration_sec * 11)

            rows.append(
                {
                    "activity": row["activity"],
                    "labelled_frames": labelled_frames,
                }
            )

    return rows


def main():
    gt_rows = load_ground_truth()

    # Total available labelled frames per activity
    activity_frames = defaultdict(int)

    for row in gt_rows:
        activity_frames[row["activity"]] += row["labelled_frames"]

    # Count errors by backend / PPE / activity / error type
    errors = defaultdict(int)

    with open(ERRORS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                row["backend"],
                row["ppe_item"],
                row["activity"],
                row["error"],
            )

            errors[key] += 1

    output_rows = []

    for backend in ["PyTorch", "NCNN"]:

        for ppe_item in PPE_ITEMS:

            for activity in sorted(activity_frames):

                total_frames = activity_frames[activity]

                fn = errors.get(
                    (backend, ppe_item, activity, "FN"),
                    0,
                )

                fp = errors.get(
                    (backend, ppe_item, activity, "FP"),
                    0,
                )

                fn_rate = fn / total_frames if total_frames else 0
                fp_rate = fp / total_frames if total_frames else 0

                output_rows.append(
                    {
                        "backend": backend,
                        "ppe_item": ppe_item,
                        "activity": activity,
                        "labelled_frames": total_frames,
                        "false_negatives": fn,
                        "false_negative_rate": fn_rate,
                        "false_positives": fp,
                        "false_positive_rate": fp_rate,
                    }
                )

    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:

        fieldnames = [
            "backend",
            "ppe_item",
            "activity",
            "labelled_frames",
            "false_negatives",
            "false_negative_rate",
            "false_positives",
            "false_positive_rate",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("PPE ACTIVITY ERROR RATE ANALYSIS")
    print("=" * 110)

    for backend in ["PyTorch", "NCNN"]:

        print()
        print(backend)
        print("-" * 110)

        rows = [
            row
            for row in output_rows
            if row["backend"] == backend
        ]

        # Only show combinations where at least one error occurred
        rows = [
            row
            for row in rows
            if row["false_negatives"] > 0
            or row["false_positives"] > 0
        ]

        rows.sort(
            key=lambda x: (
                x["ppe_item"],
                -x["false_negative_rate"],
                -x["false_positive_rate"],
            )
        )

        for row in rows:

            print(
                f"{row['ppe_item']:7s} | "
                f"{row['activity'][:32]:32s} | "
                f"frames={row['labelled_frames']:4d} | "
                f"FN={row['false_negatives']:4d} "
                f"({row['false_negative_rate'] * 100:6.1f}%) | "
                f"FP={row['false_positives']:4d} "
                f"({row['false_positive_rate'] * 100:6.1f}%)"
            )

    print()
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
