import pandas as pd
import pytest

from thunderstorm_forecasting import live_weather


def test_open_meteo_forecast_normalizes_feature_names_and_units(monkeypatch):
    timestamp = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)).floor("h")
    payload = {
        "hourly": {
            "time": [timestamp.isoformat()],
            "temperature_2m": [20],
            "relative_humidity_2m": [60],
            "precipitation": [0.5],
            "pressure_msl": [1010],
            "wind_speed_10m": [12],
        }
    }
    monkeypatch.setattr(live_weather, "_get_json", lambda *args, **kwargs: payload)

    frame = live_weather.fetch_open_meteo_forecast(18.5, 73.8)

    assert set(live_weather.FEATURE_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "temperature_c"] == 20
    assert frame.loc[0, "wind_speed_kmh"] == 12


def test_met_norway_forecast_converts_wind_speed_to_kilometers_per_hour(monkeypatch):
    payload = {
        "properties": {
            "timeseries": [
                {
                    "time": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).isoformat(),
                    "data": {
                        "instant": {
                            "details": {
                                "air_temperature": 20,
                                "relative_humidity": 75,
                                "air_pressure_at_sea_level": 1008,
                                "wind_speed": 5,
                            }
                        },
                        "next_1_hours": {"details": {"precipitation_amount": 0.5}},
                    },
                }
            ]
        }
    }
    monkeypatch.setattr(live_weather, "_get_json", lambda *args, **kwargs: payload)

    frame = live_weather.fetch_met_norway_forecast(18.5, 73.8)

    assert frame.loc[0, "wind_speed_kmh"] == pytest.approx(18)
    assert frame.loc[0, "precipitation_mm"] == 0.5
    assert set(live_weather.FEATURE_COLUMNS).issubset(frame.columns)


def test_live_weather_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="Latitude"):
        live_weather.fetch_open_meteo_forecast(91, 73.8)
