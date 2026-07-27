"""Tests for group-aware validation helpers."""

import unittest

import pandas as pd

from src.validation import passenger_groups


class PassengerGroupTests(unittest.TestCase):
    def test_ticket_and_family_connections_are_transitive(self):
        frame = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4],
                "Name": [
                    "Smith, Mr. A",
                    "Smith, Mrs. B",
                    "Jones, Mr. C",
                    "Solo, Mr. D",
                ],
                "Ticket": ["A", "B", "B", "C"],
                "SibSp": [1, 1, 0, 0],
                "Parch": [0, 0, 0, 0],
            }
        )

        groups = passenger_groups(frame)

        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertEqual(groups.iloc[1], groups.iloc[2])
        self.assertNotEqual(groups.iloc[2], groups.iloc[3])


if __name__ == "__main__":
    unittest.main()

