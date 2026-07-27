"""Tests for robustness-aware model selection."""

import unittest

import pandas as pd

from src.train import select_robust_model


class RobustModelSelectionTests(unittest.TestCase):
    def test_group_robustness_breaks_statistical_tie(self):
        repeated = pd.DataFrame(
            {
                "model": ["Random forest", "Extra trees", "Clearly worse"],
                "accuracy_mean": [0.830, 0.829, 0.810],
                "accuracy_std": [0.020, 0.018, 0.020],
                "roc_auc_mean": [0.88, 0.87, 0.85],
            }
        )
        grouped = pd.DataFrame(
            {
                "model": ["Random forest", "Extra trees", "Clearly worse"],
                "accuracy_mean": [0.800, 0.820, 0.825],
            }
        )

        selected, rationale = select_robust_model(
            repeated, grouped, evaluation_count=25
        )

        self.assertEqual(selected, "Extra trees")
        self.assertNotIn("Clearly worse", rationale["shortlisted_models"])


if __name__ == "__main__":
    unittest.main()

