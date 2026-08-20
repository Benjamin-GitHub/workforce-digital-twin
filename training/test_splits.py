#!/usr/bin/env python3
"""Focused tests for shared ST-GCN and GRU dataset splitting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
STGCN_DIR = ROOT / "training" / "stgcn"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STGCN_DIR) not in sys.path:
    sys.path.insert(0, str(STGCN_DIR))

from training.gru import gru_data  # noqa: E402
from training.split_utils import group_safe_split, split_dataset  # noqa: E402
from training.stgcn import train as stgcn_train  # noqa: E402


FRACTIONS = (0.60, 0.20, 0.20)


class StratifiedGroupSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Reproduce the private archive's balanced grouping without requiring
        # ignored training data: 6 classes, 15 source groups per class, and
        # 11 windows per group.
        cls.classes = np.asarray(
            ["walking", "standing", "idle", "bending", "carrying", "material_handling"]
        )
        labels: list[int] = []
        groups: list[str] = []
        for class_index, class_name in enumerate(cls.classes):
            for group_index in range(15):
                labels.extend([class_index] * 11)
                groups.extend([f"{class_name}-{group_index:02d}"] * 11)
        cls.y = np.asarray(labels)
        cls.groups = np.asarray(groups)

    def split(self, seed: int = 42):
        return split_dataset(
            self.y,
            self.groups,
            FRACTIONS,
            seed,
            strategy="stratified_group",
        )

    def test_balanced_archive_has_no_leakage_and_exact_counts(self) -> None:
        train_idx, val_idx, test_idx = self.split()
        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        all_indices = np.concatenate(tuple(split_indices.values()))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(self.y)))
        self.assertEqual(len(np.unique(all_indices)), len(self.y))

        split_groups = {
            name: set(self.groups[indices].tolist())
            for name, indices in split_indices.items()
        }
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["val"]))
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["test"]))
        self.assertTrue(split_groups["val"].isdisjoint(split_groups["test"]))

        expected_groups = {"train": 9, "val": 3, "test": 3}
        expected_windows = {"train": 99, "val": 33, "test": 33}
        for class_index, class_name in enumerate(self.classes):
            for split_name, indices in split_indices.items():
                class_indices = indices[self.y[indices] == class_index]
                self.assertEqual(
                    len(np.unique(self.groups[class_indices])),
                    expected_groups[split_name],
                    f"{class_name} {split_name} group count",
                )
                self.assertEqual(
                    len(class_indices),
                    expected_windows[split_name],
                    f"{class_name} {split_name} window count",
                )

    def test_seed_42_is_deterministic(self) -> None:
        first = self.split(seed=42)
        second = self.split(seed=42)
        for first_indices, second_indices in zip(first, second):
            np.testing.assert_array_equal(first_indices, second_indices)

    def test_different_seeds_can_change_group_assignments(self) -> None:
        seed_42 = self.split(seed=42)
        seed_43 = self.split(seed=43)
        self.assertTrue(
            any(
                not np.array_equal(first_indices, second_indices)
                for first_indices, second_indices in zip(seed_42, seed_43)
            )
        )

    def test_stgcn_and_gru_receive_identical_indices(self) -> None:
        stgcn_indices = stgcn_train.split_groups(
            self.y,
            self.groups,
            FRACTIONS,
            seed=42,
            strategy="stratified_group",
        )
        gru_indices = gru_data.split_dataset(
            self.y,
            self.groups,
            FRACTIONS,
            seed=42,
            strategy="stratified_group",
        )
        for stgcn_split, gru_split in zip(stgcn_indices, gru_indices):
            np.testing.assert_array_equal(stgcn_split, gru_split)

    def test_group_with_multiple_labels_is_rejected(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        groups = np.asarray(["mixed", "mixed", "class-0", "class-1"])
        with self.assertRaisesRegex(
            ValueError,
            "contains multiple labels.*exactly one class per group",
        ):
            split_dataset(
                labels,
                groups,
                FRACTIONS,
                seed=42,
                strategy="stratified_group",
            )

    def test_existing_non_stratified_behavior_remains_the_default(self) -> None:
        expected = group_safe_split(
            self.y,
            self.groups,
            fractions=(0.70, 0.15, 0.15),
            seed=42,
        )
        actual = split_dataset(
            self.y,
            self.groups,
            fractions=(0.70, 0.15, 0.15),
            seed=42,
        )
        for expected_split, actual_split in zip(expected, actual):
            np.testing.assert_array_equal(expected_split, actual_split)


class LocalConfigTests(unittest.TestCase):
    def test_local_configs_only_change_split_settings(self) -> None:
        config_pairs = (
            (
                ROOT / "training" / "stgcn" / "configs" / "train_5hz_w16.yaml",
                ROOT / "training" / "stgcn" / "configs" / "train_local_5hz_w16.yaml",
            ),
            (
                ROOT / "training" / "gru" / "configs" / "train_5hz_w16.yaml",
                ROOT / "training" / "gru" / "configs" / "train_local_5hz_w16.yaml",
            ),
        )
        for base_path, local_path in config_pairs:
            with self.subTest(config=str(local_path)):
                with base_path.open("r", encoding="utf-8") as handle:
                    base = yaml.safe_load(handle)
                with local_path.open("r", encoding="utf-8") as handle:
                    local = yaml.safe_load(handle)

                self.assertEqual(local["split_strategy"], "stratified_group")
                self.assertEqual(local["split"], [0.60, 0.20, 0.20])
                self.assertEqual(
                    {key: value for key, value in local.items() if key not in {"split_strategy", "split"}},
                    {key: value for key, value in base.items() if key != "split"},
                )


if __name__ == "__main__":
    unittest.main()
