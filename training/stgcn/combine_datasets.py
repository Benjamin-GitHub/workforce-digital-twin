from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine compatible ST-GCN NPZ datasets")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives = [np.load(path) for path in args.inputs]
    classes = archives[0]["classes"]
    for path, archive in zip(args.inputs[1:], archives[1:]):
        if not np.array_equal(classes, archive["classes"]):
            raise ValueError(f"Class order differs in {path}")
        if archive["x"].shape[1:] != archives[0]["x"].shape[1:]:
            raise ValueError(f"Tensor shape differs in {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=np.concatenate([archive["x"] for archive in archives]),
        y=np.concatenate([archive["y"] for archive in archives]),
        groups=np.concatenate([archive["groups"] for archive in archives]),
        sample_ids=np.concatenate([archive["sample_ids"] for archive in archives]),
        classes=classes,
    )
    combined = np.load(args.output)
    counts = np.bincount(combined["y"], minlength=len(classes))
    print(f"Wrote {len(combined['y'])} windows to {args.output}")
    print(dict(zip(classes.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
