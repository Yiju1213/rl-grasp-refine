from __future__ import annotations

import unittest

from src.runtime.object_splits import resolve_object_split


class TestObjectSplits(unittest.TestCase):
    def test_resolve_object_split_uses_count_over_ratio_and_is_deterministic(self):
        cfg = {
            "seed": 7,
            "train_object_id_range": [0, 74],
            "holdout_object_id_range": [75, 87],
            "val_object_count": 4,
            "val_object_ratio": 0.5,
        }

        split_a = resolve_object_split(cfg)
        split_b = resolve_object_split(cfg)

        self.assertEqual(split_a, split_b)
        self.assertEqual(split_a.split_seed, 7)
        self.assertEqual(len(split_a.train_ids), 75)
        self.assertEqual(len(split_a.val_ids), 4)
        self.assertEqual(len(split_a.test_ids), 9)
        self.assertFalse(set(split_a.train_ids) & set(split_a.val_ids))
        self.assertFalse(set(split_a.train_ids) & set(split_a.test_ids))
        self.assertFalse(set(split_a.val_ids) & set(split_a.test_ids))
        self.assertEqual(
            sorted(split_a.val_ids + split_a.test_ids),
            list(range(75, 88)),
        )

    def test_resolve_object_split_supports_ratio_when_count_missing(self):
        cfg = {
            "seed": 3,
            "train_object_id_range": [0, 2],
            "holdout_object_id_range": [3, 7],
            "val_object_ratio": 0.4,
        }

        split = resolve_object_split(cfg)

        self.assertEqual(len(split.val_ids), 2)
        self.assertEqual(len(split.test_ids), 3)
        self.assertEqual(sorted(split.val_ids + split.test_ids), [3, 4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
