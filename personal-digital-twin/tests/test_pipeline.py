import pytest

from src.causal import estimate_treatment_effect
from src.data_sim import simulate_data
from src.explain import feature_importance
from src.model_train import train_model
from src.predict import predict_wellbeing


def test_simulation_is_reproducible():
    assert simulate_data(10).equals(simulate_data(10))


def test_training_and_prediction_pipeline():
    data = simulate_data(120)
    model, score = train_model(data)
    assert -1.0 <= score <= 1.0
    value = predict_wellbeing(model, data.iloc[0].to_dict())
    assert isinstance(value, float)
    assert set(feature_importance(model)) == {
        "sleep_hours",
        "exercise_minutes",
        "focus_hours",
        "screen_hours",
        "social_minutes",
    }


def test_causal_effect_requires_two_groups():
    data = simulate_data(20)
    assert isinstance(estimate_treatment_effect(data), float)
    with pytest.raises(ValueError):
        estimate_treatment_effect(data.assign(treatment=1))
