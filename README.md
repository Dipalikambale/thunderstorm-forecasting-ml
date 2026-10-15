# Thunderstorm Forecasting with MLflow

A modular Streamlit project for merging weather observations, preparing a binary thunderstorm target, tuning classification models, evaluating predictions, and optionally logging runs and model artifacts to MLflow.

> The built-in demo data is synthetic and intended only to demonstrate the workflow. It is not a source of real weather forecasts or a measure of real-world model performance.

## Project layout

```text
.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── tests/
│   ├── test_data.py
│   └── test_model.py
└── thunderstorm_forecasting/
    ├── __init__.py
    ├── data.py
    ├── demo.py
    ├── mlflow_tracking.py
    └── modeling.py
```

## Run locally

Python 3.10 or newer is recommended.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with synthetic demo data so you can try the full workflow without weather files. Upload one or more CSV files to use your own observations. Each file needs a timestamp column (`timestamp`, `datetime`, `date_time`, or `date`); a separate `date` and `time` pair is also supported. Files are merged on their parsed timestamps. Overlapping non-timestamp columns receive source suffixes so values from different files are retained.

Select a binary target column containing `0`/`1`, `yes`/`no`, or recognized thunderstorm/event labels. Numeric and categorical weather features are imputed and transformed as part of the model pipeline. Rows without a valid timestamp are excluded; rows without a target are excluded during training.

## Live forecasts and historical training

Choose **Live forecasts + historical training** in the sidebar, upload an hourly historical CSV with **observed thunderstorm labels**, select an Indian city (or enter custom Indian coordinates), and click **Fetch data, train, and compare forecasts**. The city list includes Pune, Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Ahmedabad, Jaipur, Kochi, and Guwahati. The required CSV columns are:

```text
timestamp,temperature_c,humidity_pct,precipitation_mm,pressure_hpa,wind_speed_kmh,thunderstorm
```

The five feature columns must be numeric and use the units shown in their names. `timestamp` can also be named `datetime`, `date_time`, `valid_time`, or `date`. The target can be named `thunderstorm` or another column selected in the app, with values encoded as `0`/`1`, `yes`/`no`, or recognized storm/no-storm labels. Use labels from actual observations; do not create them from one of the input weather fields.

The selected period must include both target classes, with at least three labeled rows per class for the train/test split. A CSV containing only rain labels or a dataset without timestamps and the five required features is not suitable for thunderstorm training.

The app trains and evaluates the model using the earlier 80% of the uploaded time series and holds out the later 20%. It then fetches weather forecasts for the selected Indian location from both the [Open-Meteo Forecast API](https://open-meteo.com/en/docs) and [MET Norway Locationforecast](https://api.met.no/weatherapi/locationforecast/2.0/documentation), normalizes the weather variables, and displays the trained model's estimate for each provider's forecast periods. These providers supply global weather-model forecasts; they are not official IMD station data. Check the [India Meteorological Department](https://mausam.imd.gov.in/) for official Indian weather warnings. The live mode requires a separately labeled historical dataset because the forecast APIs do not provide reliable observed thunderstorm training labels for every Indian location.

This mode needs an internet connection. The model output is an experimental estimate—not an official alert, calibrated probability, or validated operational forecast. Verify hazardous-weather conditions with your local meteorological authority. Forecast providers can differ, and some MET Norway forecast fields may be unavailable and imputed by the model.

## MLflow tracking

Enable **Log this run to MLflow** in the app to record parameters, evaluation metrics, and the fitted scikit-learn pipeline. By default, MLflow writes to the local `mlruns` directory. To use a persistent or remote tracking server, set `MLFLOW_TRACKING_URI` before launching the app, for example:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow server --host 0.0.0.0 --port 5000
streamlit run app.py
```

For a remote tracking server, configure its artifact store and credentials separately; do not put secrets in this repository. The Streamlit Cloud filesystem is ephemeral, so configure a remote tracking server if experiment history must persist between deployments.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app for that repository and set **Main file path** to `app.py`.
3. Add any remote `MLFLOW_TRACKING_URI` through the app's secrets/environment configuration if persistent tracking is needed.
4. Deploy. Dependencies are installed from `requirements.txt`.

## Evaluate and test

The app reports accuracy, precision, recall, F1, ROC AUC when the test split contains both classes, and a confusion matrix. A stratified holdout set is kept out of the grid search. Hyperparameters are selected using stratified cross-validation on the training split and F1 scoring.

```powershell
python -m pytest
```

Model scores from the synthetic demo are not indicative of performance on real weather data. For real forecasting, use representative, time-ordered historical observations and evaluate with a time-aware holdout to avoid temporal leakage.
