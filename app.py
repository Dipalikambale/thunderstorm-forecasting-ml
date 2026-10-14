"""Streamlit interface for the thunderstorm forecasting workflow."""

from __future__ import annotations

import hashlib
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from thunderstorm_forecasting.data import load_weather_csvs, merge_weather_frames
from thunderstorm_forecasting.demo import generate_demo_sources
from thunderstorm_forecasting.modeling import TrainingResult, train_and_evaluate


st.set_page_config(page_title="Thunderstorm Forecasting", page_icon="⛈️", layout="wide")
st.title("⛈️ Thunderstorm Forecasting")
st.caption(
    "Merge weather observations, tune a classifier, evaluate held-out predictions, "
    "and optionally track experiments with MLflow."
)

with st.sidebar:
    st.header("Weather data")
    uploaded_files = st.file_uploader(
        "Upload one or more weather CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each file needs a timestamp/datetime/date column, or a date and time pair.",
    )
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
