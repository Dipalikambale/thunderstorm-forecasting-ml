"""Reproducible synthetic weather data for demonstrating the application."""

import numpy as np
import pandas as pd


def generate_demo_sources(
    rows: int = 720, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two timestamp-aligned synthetic CSV-like data frames."""
    if rows < 20:
        raise ValueError("The demo needs at least 20 rows.")

    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    humidity = np.clip(rng.normal(67, 17, rows), 20, 100)
    pressure = np.clip(rng.normal(1012, 8, rows), 975, 1040)
    wind_speed = np.clip(rng.gamma(2.2, 2.2, rows), 0, 30)
    temperature = rng.normal(22, 8, rows)
    precipitation = np.clip(rng.exponential(0.5, rows), 0, 30)
    dew_point = temperature - (100 - humidity) / 5
    event_score = (
        (humidity - 75) / 10
        + (1007 - pressure) / 7
        + (wind_speed - 5) / 6
        + precipitation / 3
        + rng.normal(0, 0.8, rows)
    )
    thunderstorm = (event_score > 0.65).astype(int)

    observations = pd.DataFrame(
        {
            "timestamp": timestamp,
            "temperature_c": temperature.round(2),
            "humidity_pct": humidity.round(2),
            "wind_speed_mps": wind_speed.round(2),
            "precipitation_mm": precipitation.round(2),
            "thunderstorm": thunderstorm,
        }
    )
    atmosphere = pd.DataFrame(
        {
            "timestamp": timestamp,
            "pressure_hpa": pressure.round(2),
            "dew_point_c": dew_point.round(2),
            "visibility_km": np.clip(rng.normal(12, 4, rows), 0.2, 25).round(2),
        }
    )
    return observations, atmosphere
