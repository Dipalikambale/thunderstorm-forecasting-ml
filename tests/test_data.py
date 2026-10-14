from io import BytesIO

import pandas as pd
import pytest

from thunderstorm_forecasting.data import load_weather_csvs, merge_weather_frames


def test_load_and_merge_csvs_normalizes_timestamps_and_keeps_overlapping_fields():
    observations = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Temperature C": [20, 21],
            "humidity": [70, 80],
        }
    )
    atmosphere = pd.DataFrame(
        {
            "datetime": ["2024-01-02", "2024-01-03"],
            "temperature c": [22, 23],
            "pressure": [1010, 1008],
        }
    )

    frames = load_weather_csvs(
        [
            ("observations.csv", BytesIO(observations.to_csv(index=False).encode())),
            ("atmosphere.csv", BytesIO(atmosphere.to_csv(index=False).encode())),
        ]
    )
    merged = merge_weather_frames(frames)

    assert len(merged) == 3
    assert "temperature_c_source_2" in merged.columns
    assert merged["timestamp"].is_monotonic_increasing
    assert merged.loc[1, "temperature_c"] == 21
    assert merged.loc[1, "temperature_c_source_2"] == 22


def test_load_csvs_rejects_missing_timestamp():
    contents = BytesIO(pd.DataFrame({"temperature": [20]}).to_csv(index=False).encode())
    with pytest.raises(ValueError, match="must contain"):
        load_weather_csvs([("missing-time.csv", contents)])


def test_merge_rejects_invalid_join_type():
    with pytest.raises(ValueError, match="Merge type"):
        merge_weather_frames([pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]})], "left")
