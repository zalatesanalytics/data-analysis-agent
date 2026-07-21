from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass
class AnalysisConfiguration:
    """Store the main settings selected by the client."""

    target_column: str
    positive_class: Any = 1
    test_size: float = 0.20
    business_priority: str = "Balanced performance"
    age_column: str | None = None
    default_months_column: str | None = None
    excluded_columns: list[str] | None = None


def standardize_column_name(column: str) -> str:
    """Convert a column name to lowercase snake_case."""

    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)

    return column.strip("_")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize all column names while preserving the original data."""

    cleaned = df.copy()

    new_columns = [
        standardize_column_name(column)
        for column in cleaned.columns
    ]

    if len(new_columns) != len(set(new_columns)):
        raise ValueError(
            "Some columns became duplicates after standardization. "
            "Rename the duplicate columns and upload the dataset again."
        )

    cleaned.columns = new_columns

    return cleaned


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel files uploaded through Streamlit."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raise ValueError("Only CSV and XLSX files are supported.")


def profile_dataset(df: pd.DataFrame) -> dict:
    """Create a collection of data-quality tables."""

    missing_values = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (
                df.isna().mean().mul(100).round(2).values
            ),
        }
    ).sort_values(
        by="missing_percent",
        ascending=False,
    )

    column_summary = pd.DataFrame(
        {
            "column": df.columns,
            "data_type": df.dtypes.astype(str).values,
            "non_missing_count": df.notna().sum().values,
            "unique_values": df.nunique(dropna=False).values,
        }
    )

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    possible_identifier_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) == len(df)
    ]

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_summary": column_summary,
        "missing_values": missing_values,
        "duplicate_count": int(df.duplicated().sum()),
        "constant_columns": constant_columns,
        "possible_identifier_columns": possible_identifier_columns,
        "numeric_summary": df.describe(
            include=[np.number]
        ).transpose(),
    }


def create_engineered_features(
    df: pd.DataFrame,
    age_column: str | None = None,
    default_months_column: str | None = None,
) -> pd.DataFrame:
    """Create age and default-duration categories."""

    engineered = df.copy()

    if age_column and age_column in engineered.columns:
        age = pd.to_numeric(
            engineered[age_column],
            errors="coerce",
        )

        engineered["age_category"] = pd.cut(
            age,
            bins=[0, 24, 34, 44, 54, 64, np.inf],
            labels=[
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65+",
            ],
            include_lowest=True,
        )

    if (
        default_months_column
        and default_months_column in engineered.columns
    ):
        default_months = pd.to_numeric(
            engineered[default_months_column],
            errors="coerce",
        )

        engineered["default_duration_category"] = pd.cut(
            default_months,
            bins=[-np.inf, 2, 6, 8, np.inf],
            labels=[
                "Less than 3 months",
                "3 to 6 months",
                "7 to 8 months",
                "More than 8 months",
            ],
        )

    return engineered


def calculate_default_rates(
    df: pd.DataFrame,
    target_column: str,
    grouping_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Calculate outcome rates for selected demographic groups."""

    tables = {}

    for column in grouping_columns:
        if column not in df.columns:
            continue

        table = (
            df.groupby(
                column,
                dropna=False,
                observed=False,
            )[target_column]
            .agg(
                observations="size",
                positive_cases="sum",
                default_rate="mean",
            )
            .reset_index()
        )

        table["default_rate_percent"] = (
            table["default_rate"] * 100
        ).round(2)

        tables[column] = table

    return tables


def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical variables."""

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_models(
    preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    """Create the three classification models."""

    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_specificity(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """Calculate the true-negative rate."""

    matrix = confusion_matrix(y_true, y_pred)

    if matrix.shape != (2, 2):
        return np.nan

    true_negative, false_positive, _, _ = matrix.ravel()

    denominator = true_negative + false_positive

    if denominator == 0:
        return np.nan

    return true_negative / denominator


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    """Calculate performance metrics for a fitted model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "specificity": calculate_specificity(
            y_test,
            predictions,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "log_loss": log_loss(
            y_test,
            probabilities,
        ),
    }

    return metrics, confusion_matrix(y_test, predictions)


def select_recommended_model(
    comparison: pd.DataFrame,
    priority: str,
) -> str:
    """Recommend a model according to the client's priority."""

    if priority == "Precision":
        ranked = comparison.sort_values(
            by=["test_precision", "test_pr_auc"],
            ascending=False,
        )

    elif priority == "Recall":
        ranked = comparison.sort_values(
            by=["test_recall", "test_pr_auc"],
            ascending=False,
        )

    else:
        ranked = comparison.sort_values(
            by=[
                "test_f1_score",
                "test_pr_auc",
                "test_roc_auc",
            ],
            ascending=False,
        )

    return str(ranked.iloc[0]["model"])


def train_and_compare_models(
    df: pd.DataFrame,
    configuration: AnalysisConfiguration,
) -> tuple[pd.DataFrame, dict, dict]:
    """Train models and return their performance results."""

    target_column = configuration.target_column
    excluded_columns = configuration.excluded_columns or []

    if target_column not in df.columns:
        raise KeyError(
            f"The target variable '{target_column}' was not found."
        )

    modelling_data = df.dropna(
        subset=[target_column]
    ).copy()

    unique_target_values = modelling_data[
        target_column
    ].dropna().unique()

    if len(unique_target_values) != 2:
        raise ValueError(
            "This version supports binary classification only. "
            f"The selected target has {len(unique_target_values)} classes."
        )

    target_mapping = {
        unique_target_values[0]: 0,
        unique_target_values[1]: 1,
    }

    y = modelling_data[target_column].map(target_mapping)

    X = modelling_data.drop(
        columns=[target_column] + excluded_columns,
        errors="ignore",
    )

    if X.shape[1] == 0:
        raise ValueError(
            "No predictor variables remain after exclusions."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=configuration.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = create_preprocessor(X_train)
    models = build_models(preprocessor)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    comparison_rows = []
    fitted_models = {}
    confusion_matrices = {}

    for model_name, model in models.items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        model.fit(X_train, y_train)

        metrics, matrix = evaluate_model(
            model,
            X_test,
            y_test,
        )

        fitted_models[model_name] = model
        confusion_matrices[model_name] = matrix

        comparison_rows.append(
            {
                "model": model_name,
                "cv_precision": np.mean(
                    cv_results["test_precision"]
                ),
                "cv_recall": np.mean(
                    cv_results["test_recall"]
                ),
                "cv_f1": np.mean(
                    cv_results["test_f1"]
                ),
                "cv_roc_auc": np.mean(
                    cv_results["test_roc_auc"]
                ),
                "cv_pr_auc": np.mean(
                    cv_results["test_pr_auc"]
                ),
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_specificity": metrics["specificity"],
                "test_f1_score": metrics["f1_score"],
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
                "test_log_loss": metrics["log_loss"],
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    return comparison, fitted_models, {
        "confusion_matrices": confusion_matrices,
        "target_mapping": target_mapping,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }