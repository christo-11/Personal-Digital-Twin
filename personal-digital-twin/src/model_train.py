"""Model training utilities."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import make_features


def train_model(data, random_state: int = 42):
    """Train a wellbeing regressor and return it with a holdout score."""
    features, target = make_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=random_state
    )
    model = Pipeline([("scale", StandardScaler()), ("regressor", RandomForestRegressor(
        n_estimators=150, random_state=random_state
    ))])
    model.fit(x_train, y_train)
    return model, model.score(x_test, y_test)
