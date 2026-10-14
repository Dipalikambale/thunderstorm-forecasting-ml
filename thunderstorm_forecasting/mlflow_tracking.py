"""MLflow logging for trained thunderstorm classifiers."""

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

from thunderstorm_forecasting.modeling import TrainingResult


def log_training_run(
    result: TrainingResult, model_name: str, experiment_name: str
) -> str:
    """Log model parameters, holdout metrics, and the fitted pipeline to MLflow."""
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model", model_name)
            mlflow.log_params(result.best_params)
            mlflow.log_param("train_rows", result.train_rows)
            mlflow.log_param("test_rows", result.test_rows)
            mlflow.log_metrics(result.metrics)
            mlflow.sklearn.log_model(result.model, artifact_path="model")
            return run.info.run_id
    except MlflowException as exc:
        raise RuntimeError(f"MLflow tracking failed: {exc}") from exc
