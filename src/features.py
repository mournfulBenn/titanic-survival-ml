"""Leakage-safe feature engineering for the Titanic dataset."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create row-level and training-fitted features from raw Titanic columns.

    Ticket group sizes are learned only from the data passed to ``fit``. During
    cross-validation this prevents information from the validation fold from
    leaking into training-time feature engineering.
    """

    _TITLE_REPLACEMENTS = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "the Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    _COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master", "Rare"}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self._validate_input(X)
        tickets = X["Ticket"].fillna("UNKNOWN").astype(str)
        self.ticket_counts_ = tickets.value_counts().to_dict()
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "ticket_counts_")
        self._validate_input(X)

        result = X.copy()
        tickets = result["Ticket"].fillna("UNKNOWN").astype(str)

        result["Title"] = self._extract_title(result["Name"])
        result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
        result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
        result["FamilySizeCategory"] = pd.cut(
            result["FamilySize"],
            bins=[0, 1, 4, float("inf")],
            labels=["Alone", "Small", "Large"],
        ).astype("object")
        result["TicketGroupSize"] = tickets.map(self.ticket_counts_).fillna(1)
        result["FarePerPerson"] = result["Fare"] / result["TicketGroupSize"]
        result["FarePerFamilyMember"] = result["Fare"] / result["FamilySize"]
        result["CabinDeck"] = (
            result["Cabin"].fillna("Unknown").astype(str).str[0].replace("U", "Unknown")
        )
        result["CabinKnown"] = result["Cabin"].notna().astype(int)
        result["TicketPrefix"] = tickets.map(self._ticket_prefix)
        result["NameLength"] = result["Name"].fillna("").astype(str).str.len()
        age_known = result["Age"].notna()
        result["IsChild"] = ((result["Age"] < 16) & age_known).astype(float)
        result.loc[~age_known, "IsChild"] = float("nan")
        result["IsMother"] = (
            result["Sex"].eq("female")
            & result["Age"].gt(18)
            & result["Parch"].gt(0)
            & result["Title"].ne("Miss")
        ).astype(int)
        return result

    @classmethod
    def _extract_title(cls, names: pd.Series) -> pd.Series:
        titles = (
            names.fillna("")
            .astype(str)
            .str.extract(r",\s*([^.]*)\.", expand=False)
            .fillna("Rare")
            .str.strip()
            .replace(cls._TITLE_REPLACEMENTS)
        )
        return titles.where(titles.isin(cls._COMMON_TITLES), "Rare")

    @staticmethod
    def _ticket_prefix(ticket: str) -> str:
        prefix = re.sub(r"[^A-Za-z]+", "", ticket).upper()
        return prefix or "NONE"

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        required = {
            "Name",
            "Ticket",
            "SibSp",
            "Parch",
            "Fare",
            "Cabin",
        }
        missing = required.difference(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
