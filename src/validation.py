"""Validation helpers for passenger-group-aware model assessment."""

from __future__ import annotations

import pandas as pd


def passenger_groups(frame: pd.DataFrame) -> pd.Series:
    """Build connected household groups from shared families and tickets.

    Two passengers belong to the same validation group when they share a
    non-singleton ticket or the same surname and family size. Connections are
    transitive, so overlapping ticket and family relationships stay in one fold.
    """

    required = {"PassengerId", "Name", "Ticket", "SibSp", "Parch"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Cannot build passenger groups; missing: {sorted(missing)}")

    working = frame.reset_index(drop=True)
    size = len(working)
    parent = list(range(size))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    family_size = working["SibSp"] + working["Parch"] + 1
    surname = (
        working["Name"]
        .fillna("")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
        .str.upper()
    )
    family_key = surname + "|" + family_size.astype(str)
    family_key = family_key.where(family_size > 1)

    ticket = working["Ticket"].fillna("").astype(str).str.strip().str.upper()
    repeated_tickets = set(ticket[ticket.map(ticket.value_counts()) > 1])

    for keys in (family_key, ticket.where(ticket.isin(repeated_tickets))):
        for _, indices in keys.dropna().groupby(keys.dropna()).groups.items():
            members = list(indices)
            for member in members[1:]:
                union(members[0], member)

    roots = [find(index) for index in range(size)]
    root_to_group = {root: group for group, root in enumerate(dict.fromkeys(roots))}
    values = [f"group_{root_to_group[root]}" for root in roots]
    return pd.Series(values, index=frame.index, name="PassengerGroup")

