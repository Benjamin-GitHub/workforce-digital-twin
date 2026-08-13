#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path


GROUND_TRUTH = (
    "benchmarks/live-video/ground-truth/"
    "esp32_controlled_5min_ground_truth.csv"
)

PREDICTIONS = {
    "PyTorch": (
        "results/controlled-video/"
        "controlled_ppe_pt_predictions.csv"
    ),
    "NCNN": (
        "results/controlled-video/"
        "controlled_ppe_ncnn_predictions.csv"
    ),
}

OUTPUT = (
    "results/controlled-video/"
    "ppe_activity_error_rates_exact.csv"
)

PPE_ITEMS = ["helmet", "vest", "gloves", "boots"]


def load_ground_truth():
    rows = []

    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "activity": row["activity"],
                    "helmet": int(row["helmet"]),
                    "vest": int(row["vest"]),
                    "gloves": int(row["gloves"]),
                    "boots": int(row["boots"]),
                }
            )

    return rows


def find_ground_truth(gt_rows, timestamp):
    for row in gt_rows:
        if row["start_sec"] <= timestamp < row["end_sec"]:
            return row

    return None


def main():
    gt_rows = load_ground_truth()

    output_rows = []

    for backend, prediction_file in PREDICTIONS.items():

        # Exact number of evaluated frames per activity
        activity_frames = defaultdict(int)

        # Confusion counts per PPE item + activity
        stats = defaultdict(
            lambda: {
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            }
        )

        with open(
            prediction_file,
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                timestamp = float(row["time_sec"])

                gt = find_ground_truth(
                    gt_rows,
                    timestamp,
                )

                # Exclude transition/unlabelled periods
                if gt is None:
                    continue

                activity = gt["activity"]

                activity_frames[activity] += 1

                for ppe_item in PPE_ITEMS:

                    actual = gt[ppe_item]
                    predicted = int(
                        row[f"{ppe_item}_present"]
                    )

                    key = (
                        backend,
                        ppe_item,
                        activity,
                    )

                    if actual == 1 and predicted == 1:
                        stats[key]["tp"] += 1

                    elif actual == 0 and predicted == 0:
                        stats[key]["tn"] += 1

                    elif actual == 0 and predicted == 1:
                        stats[key]["fp"] += 1

                    elif actual == 1 and predicted == 0:
                        stats[key]["fn"] += 1

        for ppe_item in PPE_ITEMS:

            for activity in sorted(activity_frames):

                key = (
                    backend,
                    ppe_item,
                    activity,
                )

                s = stats[key]

                total_frames = activity_frames[activity]

                positive_frames = s["tp"] + s["fn"]
                negative_frames = s["tn"] + s["fp"]

                fn_rate = (
                    s["fn"] / positive_frames
                    if positive_frames
                    else 0.0
                )

                fp_rate = (
                    s["fp"] / negative_frames
                    if negative_frames
                    else 0.0
                )

                recall = (
                    s["tp"] / positive_frames
                    if positive_frames
                    else 0.0
                )

                specificity = (
                    s["tn"] / negative_frames
                    if negative_frames
                    else 0.0
                )

                output_rows.append(
                    {
                        "backend": backend,
                        "ppe_item": ppe_item,
                        "activity": activity,
                        "evaluated_frames": total_frames,
                        "positive_gt_frames": positive_frames,
                        "negative_gt_frames": negative_frames,
                        "tp": s["tp"],
                        "tn": s["tn"],
                        "fp": s["fp"],
                        "fn": s["fn"],
                        "false_negative_rate": fn_rate,
                        "false_positive_rate": fp_rate,
                        "recall": recall,
                        "specificity": specificity,
                    }
                )

    output_path = Path(OUTPUT)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "backend",
        "ppe_item",
        "activity",
        "evaluated_frames",
        "positive_gt_frames",
        "negative_gt_frames",
        "tp",
        "tn",
        "fp",
        "fn",
        "false_negative_rate",
        "false_positive_rate",
        "recall",
        "specificity",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print(
        "EXACT FRAME-LEVEL PPE ACTIVITY ERROR ANALYSIS"
    )
    print("=" * 118)

    for backend in ["PyTorch", "NCNN"]:

        print()
        print(backend)
        print("-" * 118)

        rows = [
            row
            for row in output_rows
            if row["backend"] == backend
        ]

        rows = [
            row
            for row in rows
            if row["fp"] > 0 or row["fn"] > 0
        ]

        rows.sort(
            key=lambda x: (
                x["ppe_item"],
                -x["false_negative_rate"],
                -x["false_positive_rate"],
            )
        )

        for row in rows:

            fn_display = (
                f"{row['false_negative_rate'] * 100:6.1f}%"
                if row["positive_gt_frames"] > 0
                else "  N/A "
            )

            fp_display = (
                f"{row['false_positive_rate'] * 100:6.1f}%"
                if row["negative_gt_frames"] > 0
                else "  N/A "
            )

            print(
                f"{row['ppe_item']:7s} | "
                f"{row['activity'][:32]:32s} | "
                f"frames={row['evaluated_frames']:4d} | "
                f"TP={row['tp']:4d} "
                f"FN={row['fn']:4d} "
                f"FNR={fn_display} | "
                f"TN={row['tn']:4d} "
                f"FP={row['fp']:4d} "
                f"FPR={fp_display}"
            )

    print()
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
