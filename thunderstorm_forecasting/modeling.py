"""Feature preprocessing, hyperparameter search, and holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, float]
    best_params: dict[str, Any]
    confusion: list[list[int]]
    train_rows: int
    test_rows: int


def _binary_target(values: pd.Series, target_name: str) -> pd.Series:
    present = values.dropna()
    if present.empty:
        raise ValueError(f"Target '{target_name}' has no non-empty values.")

    numeric = pd.to_numeric(present, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.unique())
        if not unique.issubset({0, 1}) or len(unique) != 2:
            raise ValueError(
                f"Target '{target_name}' must contain exactly two classes encoded as 0 and 1."
            )
        return pd.to_numeric(values, errors="coerce").astype("Int64")

    positive_labels = {
        "1", "yes", "true", "positive", "event", "occurred", "thunderstorm",
        "thunder", "lightning", "storm",
    }
    negative_labels = {
        "0", "no", "false", "negative", "none", "clear", "absent",
        "no storm", "no-storm", "no thunderstorm", "non-thunderstorm",
    }
    normalized = present.astype(str).str.strip().str.lower()
    unique_labels = set(normalized.unique())
    positive = unique_labels & positive_labels
    negative = unique_labels & negative_labels
    if len(unique_labels) != 2 or not positive or not negative:
        raise ValueError(
            f"Target '{target_name}' must contain two classes recognizable as "
            "thunderstorm/no-thunderstorm, yes/no, true/false, positive/negative, or 0/1."
        )

    encoded = values.astype("string").str.strip().str.lower().map(
        lambda label: 1 if label in positive_labels else 0 if label in negative_labels else pd.NA
    )
    return encoded.astype("Int64")


def train_and_evaluate(
    frame: pd.DataFrame,
    target_column: str,
    model_name: str = "Random Forest",
    time_aware: bool = False,
) -> TrainingResult:
    """Tune a classifier on a training split and evaluate it on held-out rows."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    if model_name not in {"Random Forest", "Logistic Regression"}:
        raise ValueError(f"Unsupported model: {model_name}")

    labeled = frame.loc[frame[target_column].notna()].copy()
    target = _binary_target(labeled[target_column], target_column)
    labeled = labeled.loc[target.notna()].copy()
    target = target.loc[target.notna()].astype(int)
    if labeled.empty:
        raise ValueError("No rows remain after removing rows with missing targets.")

    if time_aware:
        if "timestamp" not in labeled.columns:
            raise ValueError("Time-aware evaluation requires a 'timestamp' column.")
        order = labeled["timestamp"].sort_values(kind="stable").index
        labeled = labeled.loc[order]
        target = target.loc[order]

    features = labeled.drop(columns=[target_column, "timestamp"], errors="ignore")
    features = features.loc[:, features.notna().any()]
    if features.shape[1] == 0:
        raise ValueError("No usable weather feature columns remain.")
    if features.shape[0] < 10:
        raise ValueError("At least 10 labeled rows are required for a train/test split.")

    class_counts = target.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 3:
        raise ValueError("Each target class must have at least 3 labeled rows.")

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_columns)
        )
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessing = ColumnTransformer(transformers=transformers)
    if model_name == "Random Forest":
        classifier = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=1
        )
        parameter_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
        }
    else:
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        parameter_grid = {"classifier__C": [0.1, 1.0, 10.0]}

    pipeline = Pipeline(
        [("preprocessing", preprocessing), ("classifier", classifier)]
    )
    if time_aware:
        split_at = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_at], features.iloc[split_at:]
        y_train, y_test = target.iloc[:split_at], target.iloc[split_at:]
        train_class_counts = y_train.value_counts()
        if len(train_class_counts) != 2 or train_class_counts.min() < 2:
            raise ValueError(
                "The historical training period needs at least 2 examples of each "
                "target class before the holdout period."
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target,
        )
    folds = min(3, int(y_train.value_counts().min()))
    search = GridSearchCV(
        pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    predictions = search.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if y_test.nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
        )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return TrainingResult(
        model=search.best_estimator_,
        metrics=metrics,
        best_params=search.best_params_,
        confusion=matrix,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )
