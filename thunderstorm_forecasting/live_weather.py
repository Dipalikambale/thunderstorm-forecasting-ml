"""Fetch and normalize live forecasts from public weather APIs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "pressure_hpa",
    "wind_speed_kmh",
]

INDIAN_LOCATIONS = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Kochi": (9.9312, 76.2673),
    "Guwahati": (26.1445, 91.7362),
}

_OPEN_METEO_VARIABLES = (
    "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m"
)
_MET_NORWAY_USER_AGENT = (
    "ThunderstormForecasting/1.0 "
    "(https://github.com/Dipalikambale/thunderstorm-forecasting-ml)"
)


def _get_json(url: str, source_name: str, user_agent: str | None = None) -> dict[str, Any]:
    headers = {"User-Agent": user_agent} if user_agent else {}
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source_name} request failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{source_name} returned an unexpected response.")
    if "error" in payload:
        reason = payload.get("reason", "The provider rejected the request.")
        raise ValueError(f"{source_name}: {reason}")
    return payload


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")


def _open_meteo_hourly_frame(
    payload: dict[str, Any], source_name: str
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError(f"{source_name} returned no hourly weather data.")

    columns = {
        "temperature_c": "temperature_2m",
        "humidity_pct": "relative_humidity_2m",
        "precipitation_mm": "precipitation",
        "pressure_hpa": "pressure_msl",
        "wind_speed_kmh": "wind_speed_10m",
    }
    missing = [name for name in columns.values() if name not in hourly]
    if missing:
        raise ValueError(f"{source_name} response is missing fields: {', '.join(missing)}.")

    lengths = {len(hourly["time"]), *(len(hourly[name]) for name in columns.values())}
    if len(lengths) != 1:
        raise ValueError(f"{source_name} returned inconsistent hourly data lengths.")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"], errors="coerce", utc=True),
            **{
                normalized: pd.to_numeric(hourly[provided], errors="coerce")
                for normalized, provided in columns.items()
            },
        }
    )
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        raise ValueError(f"{source_name} returned no valid hourly timestamps.")
    return frame


def fetch_open_meteo_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetch a three-day hourly forecast from Open-Meteo."""
    _validate_coordinates(latitude, longitude)
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": _OPEN_METEO_VARIABLES,
            "forecast_days": 3,
            "timezone": "UTC",
        }
    )
    payload = _get_json(
        f"https://api.open-meteo.com/v1/forecast?{query}", "Open-Meteo forecast"
    )
    frame = _open_meteo_hourly_frame(payload, "Open-Meteo forecast")
    now = pd.Timestamp(datetime.now(timezone.utc))
    return frame.loc[frame["timestamp"] >= now].reset_index(drop=True)


def fetch_met_norway_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetch and normalize MET Norway's location forecast for the same features."""
    _validate_coordinates(latitude, longitude)
    query = urlencode({"lat": latitude, "lon": longitude})
    payload = _get_json(
        f"https://api.met.no/weatherapi/locationforecast/2.0/compact?{query}",
        "MET Norway forecast",
        user_agent=_MET_NORWAY_USER_AGENT,
    )
    properties = payload.get("properties")
    timeseries = properties.get("timeseries") if isinstance(properties, dict) else None
    if not isinstance(timeseries, list):
        raise ValueError("MET Norway returned no forecast time series.")

    rows = []
    for period in timeseries:
        if not isinstance(period, dict):
            continue
        data = period.get("data")
        instant = data.get("instant") if isinstance(data, dict) else None
        details = instant.get("details") if isinstance(instant, dict) else None
        if not isinstance(details, dict):
            continue
        next_hour = data.get("next_1_hours") if isinstance(data, dict) else None
        next_details = next_hour.get("details") if isinstance(next_hour, dict) else {}
        rows.append(
            {
                "timestamp": period.get("time"),
                "temperature_c": details.get("air_temperature"),
                "humidity_pct": details.get("relative_humidity"),
                "precipitation_mm": (
                    next_details.get("precipitation_amount")
                    if isinstance(next_details, dict)
                    else None
                ),
                "pressure_hpa": details.get("air_pressure_at_sea_level"),
                "wind_speed_kmh": (
                    float(details["wind_speed"]) * 3.6
                    if details.get("wind_speed") is not None
                    else None
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("MET Norway returned no usable forecast periods.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    return frame.loc[frame["timestamp"] >= now].reset_index(drop=True)
