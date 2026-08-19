"""Shared deterministic dataset splitting for ST-GCN and GRU training."""

from __future__ import annotations

import numpy as np


def _validate_fractions(fractions) -> None:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must contain train/val/test values summing to 1")


def group_safe_split(y, groups, fractions, seed, attempts=500):
    """Preserve the existing attempted group-shuffle behavior."""
    from sklearn.model_selection import GroupShuffleSplit

    _validate_fractions(fractions)
    labels = set(np.unique(y).tolist())
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three independent groups are required")

    for attempt in range(attempts):
        current_seed = seed + attempt
        first = GroupShuffleSplit(1, train_size=fractions[0], random_state=current_seed)
        train, remainder = next(first.split(y, y, groups))
        relative_val = fractions[1] / (fractions[1] + fractions[2])
        second = GroupShuffleSplit(1, train_size=relative_val, random_state=current_seed + 10_000)
        val_rel, test_rel = next(
            second.split(y[remainder], y[remainder], groups[remainder])
        )
        val, test = remainder[val_rel], remainder[test_rel]
        if all(set(np.unique(y[index]).tolist()) == labels for index in (train, val, test)):
            return train, val, test
    raise ValueError(
        "Could not create grouped train/val/test splits containing every class. "
        "Add independent groups for rare classes or supply a pre-defined split."
    )


def _allocation_counts(group_count: int, fractions) -> np.ndarray:
    exact = np.asarray(fractions, dtype=np.float64) * group_count
    counts = np.floor(exact).astype(np.int64)
    remaining = group_count - int(counts.sum())
    if remaining:
        remainders = exact - counts
        order = np.argsort(-remainders, kind="stable")
        counts[order[:remaining]] += 1
    return counts


def stratified_group_split(y, groups, fractions, seed):
    """Shuffle and allocate intact single-class groups independently per class."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    _validate_fractions(fractions)
    if len(y) != len(groups):
        raise ValueError("y and groups lengths differ")

    labels = np.unique(y)
    groups_by_label = {}
    for group in np.unique(groups):
        group_labels = np.unique(y[groups == group])
        if len(group_labels) != 1:
            raise ValueError(
                f"Group {group!r} contains multiple labels: {group_labels.tolist()}; "
                "stratified_group requires exactly one class per group"
            )
        groups_by_label.setdefault(group_labels[0], []).append(group)

    split_groups = [set(), set(), set()]
    class_seeds = np.random.SeedSequence(seed).spawn(len(labels))
    for label, class_seed in zip(labels, class_seeds):
        class_groups = np.asarray(sorted(groups_by_label.get(label, [])), dtype=groups.dtype)
        np.random.default_rng(class_seed).shuffle(class_groups)
        train_count, val_count, _ = _allocation_counts(len(class_groups), fractions)
        val_end = int(train_count + val_count)
        split_groups[0].update(class_groups[:train_count].tolist())
        split_groups[1].update(class_groups[train_count:val_end].tolist())
        split_groups[2].update(class_groups[val_end:].tolist())

    return tuple(
        np.flatnonzero(np.isin(groups, list(selected)))
        for selected in split_groups
    )


def split_dataset(y, groups, fractions, seed, strategy=None, attempts=500):
    if strategy in (None, "group_safe"):
        return group_safe_split(y, groups, fractions, seed, attempts=attempts)
    if strategy == "stratified_group":
        return stratified_group_split(y, groups, fractions, seed)
    raise ValueError(
        f"Unknown split_strategy {strategy!r}; expected group_safe or stratified_group"
    )
