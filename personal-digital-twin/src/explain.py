"""Model explanation helpers."""

from __future__ import annotations

from .features import FEATURES


def feature_importance(model) -> dict[str, float]:
    """Return feature importances from the fitted random forest pipeline."""
    importances = model.named_steps["regressor"].feature_importances_
    return dict(zip(FEATURES, importances))
