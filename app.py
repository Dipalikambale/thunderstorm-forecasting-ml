"""Streamlit interface for the thunderstorm forecasting workflow."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from thunderstorm_forecasting.data import load_weather_csvs, merge_weather_frames
from thunderstorm_forecasting.demo import generate_demo_sources
from thunderstorm_forecasting.live_weather import (
    FEATURE_COLUMNS,
    INDIAN_LOCATIONS,
    fetch_met_norway_forecast,
    fetch_open_meteo_forecast,
)
from thunderstorm_forecasting.modeling import TrainingResult, train_and_evaluate


st.set_page_config(page_title="Thunderstorm Forecasting", page_icon="⛈️", layout="wide")
st.title("⛈️ Thunderstorm Forecasting")
st.caption(
    "Merge weather observations, tune a classifier, evaluate held-out predictions, "
    "and optionally track experiments with MLflow."
)

with st.sidebar:
    st.header("Weather data")
    workflow = st.radio(
        "Data workflow",
        ["Live forecasts + historical training", "Upload CSVs / demo"],
    )
    if workflow == "Live forecasts + historical training":
        training_csv = st.file_uploader(
            "Historical CSV with observed thunderstorm labels",
            type=["csv"],
            help=(
                "Needs timestamp, temperature_c, humidity_pct, precipitation_mm, "
                "pressure_hpa, wind_speed_kmh, and a binary target column."
            ),
        )
        uploaded_files = None
        model_name = None
    else:
        uploaded_files = st.file_uploader(
            "Upload one or more weather CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Each file needs a timestamp/datetime/date column, or a date and time pair.",
        )
        training_csv = None
        merge_type = st.selectbox("Timestamp merge", ["outer", "inner"])
        model_name = st.selectbox("Model to tune", ["Random Forest", "Logistic Regression"])
    st.divider()
    log_to_mlflow = st.checkbox("Log this run to MLflow")
    tracking_uri = st.text_input(
        "MLflow tracking URI",
        value=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
        disabled=not log_to_mlflow,
    )
    experiment_name = st.text_input(
        "MLflow experiment", value="thunderstorm-forecasting", disabled=not log_to_mlflow
    )

if workflow == "Live forecasts + historical training":
    st.subheader("Live forecast from two weather providers")
    st.caption(
        "Train on historical observations labeled with actual thunderstorm outcomes, "
        "then compare forecasts for an Indian location from Open-Meteo and MET Norway."
    )
    location_column, target_column_area = st.columns(2)
    with location_column:
        city = st.selectbox(
            "Indian city", [*INDIAN_LOCATIONS, "Custom coordinates (India)"]
        )
        if city == "Custom coordinates (India)":
            latitude = st.number_input(
                "Latitude", min_value=6.0, max_value=38.0, value=18.5204, format="%.4f"
            )
            longitude = st.number_input(
                "Longitude", min_value=68.0, max_value=98.0, value=73.8567, format="%.4f"
            )
        else:
            latitude, longitude = INDIAN_LOCATIONS[city]
            st.caption(f"Coordinates: {latitude:.4f}, {longitude:.4f}")
    if training_csv is None:
        st.info(
            "Upload historical observations with a timestamp, the five listed weather "
            "features, and a thunderstorm/no-thunderstorm target to train and evaluate."
        )
        st.stop()

    try:
        history = load_weather_csvs(
            [(training_csv.name, BytesIO(training_csv.getvalue()))]
        )[0]
    except (OSError, ValueError) as error:
        st.error(f"Could not load the historical training CSV: {error}")
        st.stop()

    missing_features = [column for column in FEATURE_COLUMNS if column not in history]
    if missing_features:
        st.error(
            "The historical CSV is missing required model features: "
            f"{', '.join(missing_features)}."
        )
        st.stop()
    for column in FEATURE_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce")
        if history[column].notna().sum() == 0:
            st.error(f"The historical CSV feature '{column}' has no numeric values.")
            st.stop()

    target_options = [
        column
        for column in history.columns
        if column != "timestamp" and column not in FEATURE_COLUMNS
    ]
    if not target_options:
        st.error("The historical CSV needs a target column for observed storm outcomes.")
        st.stop()
    default_target = (
        target_options.index("thunderstorm")
        if "thunderstorm" in target_options
        else 0
    )
    with target_column_area:
        target_column = st.selectbox(
            "Observed thunderstorm target", target_options, index=default_target
        )
    live_model_name = st.selectbox(
        "Model to tune", ["Random Forest", "Logistic Regression"], key="live_model_name"
    )
    training_frame = history[["timestamp", *FEATURE_COLUMNS, target_column]].copy()
    fingerprint = hashlib.sha256(training_csv.getvalue()).hexdigest()
    config = (
        latitude,
        longitude,
        target_column,
        live_model_name,
        fingerprint,
        log_to_mlflow,
        tracking_uri,
        experiment_name,
    )

    if st.button("Fetch data, train, and compare forecasts", type="primary"):
        st.session_state.pop("live_result", None)
        try:
            with st.spinner(
                "Training on labeled history and fetching two live forecasts..."
            ):
                result = train_and_evaluate(
                    training_frame, target_column, live_model_name, time_aware=True
                )
                forecasts = {
                    "Open-Meteo": fetch_open_meteo_forecast(latitude, longitude),
                    "MET Norway": fetch_met_norway_forecast(latitude, longitude),
                }
                positive_class = list(result.model.classes_).index(1)
                forecast_rows = []
                for source_name, forecast in forecasts.items():
                    if forecast.empty:
                        raise ValueError(f"{source_name} returned no future forecast rows.")
                    predictions = forecast[FEATURE_COLUMNS].copy()
                    forecast_with_predictions = forecast.copy()
                    forecast_with_predictions["thunderstorm_probability_pct"] = (
                        result.model.predict_proba(predictions)[:, positive_class] * 100
                    ).round(1)
                    forecast_with_predictions.insert(0, "source", source_name)
                    forecast_rows.append(forecast_with_predictions)

                run_id = None
                if log_to_mlflow:
                    if not tracking_uri.strip():
                        raise ValueError(
                            "Provide an MLflow tracking URI or disable MLflow logging."
                        )
                    import mlflow

                    mlflow.set_tracking_uri(tracking_uri.strip())
                    from thunderstorm_forecasting.mlflow_tracking import log_training_run

                    run_id = log_training_run(
                        result, live_model_name, experiment_name.strip()
                    )

            st.session_state["live_result"] = {
                "config": config,
                "history": training_frame,
                "training": result,
                "forecasts": pd.concat(forecast_rows, ignore_index=True),
                "mlflow_run_id": run_id,
            }
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Live data or training failed: {error}")

    live_result = st.session_state.get("live_result")
    if isinstance(live_result, dict) and live_result.get("config") == config:
        history = live_result["history"]
        result = live_result["training"]
        forecasts = live_result["forecasts"]
        st.subheader("Historical training data")
        st.caption(
            "These uploaded labels must represent observed thunderstorm outcomes; the "
            "app does not infer them from the forecast providers."
        )
        st.dataframe(history.head(20), width="stretch")
        st.write(
            f"{len(history):,} timestamped rows · target: {target_column}"
        )

        st.subheader("Chronological holdout evaluation")
        metric_columns = st.columns(len(result.metrics))
        for column, (metric, value) in zip(metric_columns, result.metrics.items()):
            column.metric(metric.replace("_", " ").title(), f"{value:.3f}")
        st.caption(
            f"Trained on the earlier 80% of the selected period and evaluated on the "
            f"later 20% ({result.test_rows:,} rows). This is an experimental estimate, "
            "not an official warning or a validated operational forecast."
        )

        st.subheader("Model estimates for provider forecasts")
        st.dataframe(forecasts.head(48), width="stretch")
        st.caption(
            "Probability is produced by the model trained on your labeled historical data. "
            "Provider forecast variables are normalized to common units; MET Norway "
            "forecasts may have missing values that the model imputes."
        )
        st.caption(
            "These are global weather-model forecasts for India, not official IMD "
            "station observations or warnings. Check the "
            "[India Meteorological Department](https://mausam.imd.gov.in/) for official alerts."
        )
        run_id = live_result["mlflow_run_id"]
        if log_to_mlflow and run_id:
            st.success(f"Logged to MLflow run `{run_id}`.")
    st.stop()

try:
    if uploaded_files:
        sources = [(uploaded.name, uploaded) for uploaded in uploaded_files]
        data = merge_weather_frames(load_weather_csvs(sources), how=merge_type)
        data_origin = "Uploaded CSV data"
    else:
        demo_sources = generate_demo_sources()
        data = merge_weather_frames(list(demo_sources), how=merge_type)
        data_origin = "Synthetic demo data"
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

st.subheader("Merged weather data")
if data_origin == "Synthetic demo data":
    st.info(
        "Showing reproducible synthetic demo observations, not real measurements or "
        "validated forecasts. Upload your own CSV files to train on real data."
    )
else:
    st.success(f"Loaded and merged {len(uploaded_files)} CSV file(s).")

left, right = st.columns([3, 1])
with left:
    st.dataframe(data.head(20), width="stretch")
with right:
    st.metric("Merged rows", f"{len(data):,}")
    st.metric("Weather fields", len(data.columns) - 1)
    st.metric("Time range", f"{data['timestamp'].min():%Y-%m-%d} to {data['timestamp'].max():%Y-%m-%d}")

target_options = [column for column in data.columns if column != "timestamp"]
default_target = target_options.index("thunderstorm") if "thunderstorm" in target_options else 0
target_column = st.selectbox("Thunderstorm target column", target_options, index=default_target)

fingerprint = hashlib.sha256(
    data.to_csv(index=False).encode("utf-8")
).hexdigest()
if st.button("Tune model and evaluate", type="primary"):
    try:
        with st.spinner("Tuning hyperparameters and evaluating on held-out data..."):
            result = train_and_evaluate(data, target_column, model_name)
        st.session_state["training_result"] = result
        st.session_state["training_fingerprint"] = fingerprint
        st.session_state["training_target"] = target_column
        st.session_state["training_model_name"] = model_name
        if log_to_mlflow:
            if not tracking_uri.strip():
                raise ValueError("Provide an MLflow tracking URI or disable MLflow logging.")
            import mlflow

            mlflow.set_tracking_uri(tracking_uri.strip())
            from thunderstorm_forecasting.mlflow_tracking import log_training_run

            run_id = log_training_run(result, model_name, experiment_name.strip())
            st.session_state["mlflow_run_id"] = run_id
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Training or MLflow logging failed: {error}")
        st.exception(error)

result = st.session_state.get("training_result")
if (
    isinstance(result, TrainingResult)
    and st.session_state.get("training_fingerprint") == fingerprint
    and st.session_state.get("training_target") == target_column
    and st.session_state.get("training_model_name") == model_name
):
    st.subheader("Holdout evaluation")
    metric_columns = st.columns(len(result.metrics))
    for column, (metric, value) in zip(metric_columns, result.metrics.items()):
        column.metric(metric.replace("_", " ").title(), f"{value:.3f}")

    chart_column, params_column = st.columns(2)
    with chart_column:
        st.markdown("**Confusion matrix** (rows: actual, columns: predicted)")
        figure, axis = plt.subplots()
        image = axis.imshow(result.confusion, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No thunderstorm", "Thunderstorm"],
            yticklabels=["No thunderstorm", "Thunderstorm"],
            xlabel="Predicted",
            ylabel="Actual",
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, result.confusion[row][column], ha="center", va="center")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with params_column:
        st.markdown("**Best cross-validation parameters**")
        st.json(result.best_params)
        st.caption(
            f"Trained on {result.train_rows:,} rows; evaluated on "
            f"{result.test_rows:,} held-out rows. Hyperparameters were selected "
            "using stratified cross-validation and F1 scoring."
        )
        run_id = st.session_state.get("mlflow_run_id")
        if log_to_mlflow and run_id:
            st.success(f"Logged to MLflow run `{run_id}`.")
