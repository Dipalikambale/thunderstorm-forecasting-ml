import pytest

from thunderstorm_forecasting.demo import generate_demo_sources
from thunderstorm_forecasting.data import merge_weather_frames
from thunderstorm_forecasting.modeling import train_and_evaluate


def test_random_forest_training_returns_holdout_metrics():
    frame = merge_weather_frames(list(generate_demo_sources(rows=180)))
    result = train_and_evaluate(frame, "thunderstorm")

    assert result.train_rows + result.test_rows == len(frame)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= result.metrics.keys()
    assert len(result.confusion) == 2
    assert result.model.predict(frame.drop(columns=["thunderstorm", "timestamp"]).head(3)).shape == (3,)


def test_training_rejects_target_with_only_one_class():
    frame = merge_weather_frames(list(generate_demo_sources(rows=100)))
    frame["thunderstorm"] = 0

    with pytest.raises(ValueError, match="exactly two classes"):
        train_and_evaluate(frame, "thunderstorm")
