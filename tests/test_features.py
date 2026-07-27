"""Unit tests for Titanic feature engineering."""

import unittest

import numpy as np
import pandas as pd

from src.features import TitanicFeatureEngineer


class TitanicFeatureEngineerTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Name": [
                    "Example, Mlle. Alice",
                    "Example, Mr. Bob",
                    "Example, Dona. Carla",
                ],
                "Ticket": ["PC 123", "PC 123", "456"],
                "SibSp": [0, 1, 0],
                "Parch": [0, 1, 0],
                "Fare": [20.0, 20.0, np.nan],
                "Cabin": ["C85", np.nan, "E10"],
                "Age": [15.0, 35.0, 30.0],
                "Sex": ["female", "male", "female"],
            }
        )

    def test_expected_features_are_created(self):
        transformed = TitanicFeatureEngineer().fit_transform(self.frame)

        self.assertEqual(transformed["Title"].tolist(), ["Miss", "Mr", "Rare"])
        self.assertEqual(transformed["FamilySize"].tolist(), [1, 3, 1])
        self.assertEqual(transformed["IsAlone"].tolist(), [1, 0, 1])
        self.assertEqual(transformed["TicketGroupSize"].tolist(), [2, 2, 1])
        self.assertEqual(transformed["CabinDeck"].tolist(), ["C", "Unknown", "E"])
        self.assertEqual(
            transformed["FamilySizeCategory"].tolist(), ["Alone", "Small", "Alone"]
        )
        self.assertEqual(transformed["IsChild"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(transformed["CabinKnown"].tolist(), [1, 0, 1])

    def test_unknown_ticket_defaults_to_one(self):
        engineer = TitanicFeatureEngineer().fit(self.frame.iloc[:2])
        transformed = engineer.transform(self.frame.iloc[2:])

        self.assertEqual(transformed.iloc[0]["TicketGroupSize"], 1)


if __name__ == "__main__":
    unittest.main()
