"""Prediction helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.base import RegressorMixin

from .features import FEATURES


def predict_wellbeing(model: RegressorMixin, observation: dict[str, float]) -> float:
    """Predict wellbeing for one observation."""
    row = pd.DataFrame([{feature: observation[feature] for feature in FEATURES}])
    return float(model.predict(row)[0])
