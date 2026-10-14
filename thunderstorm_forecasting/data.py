"""Weather CSV loading, timestamp normalization, and data merging."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import IO

import pandas as pd


def _normalize_name(name: object) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def _with_timestamp(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{source_name} contains no rows.")

    normalized = frame.copy()
    normalized.columns = [_normalize_name(column) for column in normalized.columns]
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError(f"{source_name} has duplicate column names after normalization.")

    timestamp_column = next(
        (
            name
            for name in ("timestamp", "datetime", "date_time", "valid_time")
            if name in normalized.columns
        ),
        None,
    )
    if timestamp_column is not None:
        timestamps = pd.to_datetime(normalized[timestamp_column], errors="coerce", utc=True)
    elif "date" in normalized.columns and "time" in normalized.columns:
        timestamps = pd.to_datetime(
            normalized["date"].astype(str) + " " + normalized["time"].astype(str),
            errors="coerce",
            utc=True,
        )
    elif "date" in normalized.columns:
        timestamps = pd.to_datetime(normalized["date"], errors="coerce", utc=True)
    else:
        raise ValueError(
            f"{source_name} must contain a timestamp, datetime, date_time, valid_time, "
            "or date column (optionally with a separate time column)."
        )

    normalized = normalized.drop(columns=["timestamp", "datetime", "date_time", "valid_time", "date", "time"], errors="ignore")
    normalized.insert(0, "timestamp", timestamps)
    normalized = normalized.dropna(subset=["timestamp"])
    if normalized.empty:
        raise ValueError(f"{source_name} has no rows with a valid timestamp.")

    return normalized.drop_duplicates(subset=["timestamp"], keep="last")


def load_weather_csvs(
    sources: Iterable[tuple[str, str | Path | IO[bytes] | IO[str]]],
) -> list[pd.DataFrame]:
    """Read CSV sources and normalize their timestamp columns."""
    frames: list[pd.DataFrame] = []
    for source_name, source in sources:
        try:
            frame = pd.read_csv(source)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ValueError(f"Could not read {source_name}: {exc}") from exc
        frames.append(_with_timestamp(frame, source_name))

    if not frames:
        raise ValueError("Upload at least one weather CSV file.")
    return frames


def merge_weather_frames(
    frames: Iterable[pd.DataFrame], how: str = "outer"
) -> pd.DataFrame:
    """Merge normalized weather frames on timestamp while retaining duplicate fields."""
    if how not in {"outer", "inner"}:
        raise ValueError("Merge type must be 'outer' or 'inner'.")

    iterator = iter(frames)
    try:
        merged = next(iterator).copy()
    except StopIteration as exc:
        raise ValueError("At least one weather data frame is required.") from exc

    if "timestamp" not in merged.columns:
        raise ValueError("Weather frames must have a normalized 'timestamp' column.")

    for source_number, frame in enumerate(iterator, start=2):
        if "timestamp" not in frame.columns:
            raise ValueError(f"Weather frame {source_number} has no 'timestamp' column.")
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"Weather frame {source_number} has duplicate timestamps.")
        if merged["timestamp"].duplicated().any():
            raise ValueError("The merged weather data has duplicate timestamps.")

        overlap = (set(merged.columns) & set(frame.columns)) - {"timestamp"}
        renamed = frame.rename(
            columns={column: f"{column}_source_{source_number}" for column in overlap}
        )
        merged = merged.merge(renamed, on="timestamp", how=how, validate="one_to_one")

    return merged.sort_values("timestamp").reset_index(drop=True)
