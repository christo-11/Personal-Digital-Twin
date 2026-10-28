"""Synthetic personal behavior data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Create reproducible daily behavior and wellbeing observations."""
    rng = np.random.default_rng(random_state)
    sleep_hours = np.clip(rng.normal(7.0, 1.0, n_samples), 4.0, 10.0)
    exercise_minutes = np.clip(rng.gamma(2.0, 20.0, n_samples), 0.0, 150.0)
    focus_hours = np.clip(rng.normal(5.0, 1.5, n_samples), 0.0, 10.0)
    screen_hours = np.clip(rng.normal(5.0, 1.8, n_samples), 0.0, 12.0)
    social_minutes = np.clip(rng.normal(60.0, 30.0, n_samples), 0.0, 240.0)
    treatment = (exercise_minutes >= 45).astype(int)
    wellbeing = (
        45 + 4.5 * sleep_hours + 0.08 * exercise_minutes
        + 2.0 * focus_hours + 0.03 * social_minutes
        - 1.8 * screen_hours + rng.normal(0, 4, n_samples)
    )
    return pd.DataFrame({
        "sleep_hours": sleep_hours,
        "exercise_minutes": exercise_minutes,
        "focus_hours": focus_hours,
        "screen_hours": screen_hours,
        "social_minutes": social_minutes,
        "treatment": treatment,
        "wellbeing": wellbeing,
    })
