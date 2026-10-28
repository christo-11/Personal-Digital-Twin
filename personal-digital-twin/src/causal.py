"""Small causal-analysis helpers for the synthetic example."""

from __future__ import annotations

import pandas as pd


def estimate_treatment_effect(data: pd.DataFrame) -> float:
    """Estimate the raw wellbeing difference for treated vs untreated days."""
    groups = data.groupby("treatment")["wellbeing"].mean()
    if not {0, 1}.issubset(groups.index):
        raise ValueError("Both treatment groups are required")
    return float(groups.loc[1] - groups.loc[0])
