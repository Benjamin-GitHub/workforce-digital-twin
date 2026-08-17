import unittest

from src.vision.ppe import associate_ppe


class PPEAssociationTests(unittest.TestCase):
    def test_associates_positive_and_explicit_negative_to_one_person(self):
        state = associate_ppe(
            [[0, 0, 100, 200]],
            [
                {"class_name": "helmet", "confidence": 0.8, "box": [30, 5, 70, 40]},
                {"class_name": "no_gloves", "confidence": 0.6, "box": [10, 80, 90, 140]},
            ],
            observed_at="2026-08-16T12:00:00+00:00",
        )[0]
        self.assertTrue(state["helmet"]["detected"])
        self.assertFalse(state["gloves"]["detected"])
        self.assertIsNone(state["vest"]["detected"])

    def test_discards_detection_ambiguous_between_people(self):
        states = associate_ppe(
            [[0, 0, 100, 200], [50, 0, 150, 200]],
            [{"class_name": "helmet", "confidence": 0.9, "box": [60, 10, 90, 40]}],
        )
        self.assertIsNone(states[0]["helmet"]["detected"])
        self.assertIsNone(states[1]["helmet"]["detected"])

    def test_missing_detection_is_unknown_not_noncompliant(self):
        state = associate_ppe([[0, 0, 100, 200]], [])[0]
        self.assertIsNone(state["helmet"]["detected"])
        self.assertIsNone(state["vest"]["detected"])


if __name__ == "__main__":
    unittest.main()
