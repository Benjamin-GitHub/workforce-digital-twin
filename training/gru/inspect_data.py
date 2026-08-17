from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gru_data import group_safe_split, load_pose_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an existing ST-GCN NPZ for GRU training")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--temporal-stride", type=int, default=2)
    args = parser.parse_args()
    data = load_pose_archive(args.data, args.temporal_stride)
    train, val, test = group_safe_split(data.y, data.groups, [0.70, 0.15, 0.15], 42)
    print(f"input={args.data}")
    print(f"gru_shape={data.x.shape}; dtype={data.x.dtype}; finite={np.isfinite(data.x).all()}")
    for name, index in (("all", np.arange(len(data.y))), ("train", train), ("val", val), ("test", test)):
        counts = np.bincount(data.y[index], minlength=len(data.classes))
        print(f"{name}: samples={len(index)} groups={len(set(data.groups[index]))} counts={dict(zip(data.classes, counts.tolist()))}")


if __name__ == "__main__":
    main()

