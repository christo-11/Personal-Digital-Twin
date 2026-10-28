"""Feature selection and preparation."""

from __future__ import annotations

import pandas as pd


FEATURES = [
    "sleep_hours",
    "exercise_minutes",
    "focus_hours",
    "screen_hours",
    "social_minutes",
]
TARGET = "wellbeing"


def make_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target, validating expected columns."""
    required = set(FEATURES + [TARGET])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return data.loc[:, FEATURES].copy(), data[TARGET].copy()
