"""Model training utilities for sleep stage classification."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def create_model(parallel_jobs: int = 1) -> Pipeline:
    """Create the default Random Forest pipeline."""
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=parallel_jobs,
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def load_saved_features(features_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load saved feature and label arrays from disk."""
    X = np.load(features_path)
    y = np.load(labels_path)
    return X, y


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    output_dir: str = "outputs",
    parallel_jobs: int = 1,
) -> dict:
    """Train the model, run cross-validation, and save outputs."""
    os.makedirs(output_dir, exist_ok=True)

    model = create_model(parallel_jobs=parallel_jobs)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\n{'=' * 60}")
    print(f"Running {n_splits}-fold stratified cross-validation")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'=' * 60}\n")

    y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=parallel_jobs)

    acc = accuracy_score(y, y_pred_cv)
    kappa = cohen_kappa_score(y, y_pred_cv)
    target_names = [SLEEP_STAGE_NAMES.get(i, f"Stage-{i}") for i in sorted(np.unique(y))]
    report = classification_report(y, y_pred_cv, target_names=target_names, output_dict=True)
    report_str = classification_report(y, y_pred_cv, target_names=target_names)
    cm = confusion_matrix(y, y_pred_cv)

    print("Classification report:")
    print(report_str)
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")

    print("\nTraining final model on the full dataset...")
    model.fit(X, y)

    model_path = os.path.join(output_dir, "sleep_stage_model.joblib")
    joblib.dump(model, model_path)

    feature_importances = model.named_steps["classifier"].feature_importances_

    metrics = {
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            SLEEP_STAGE_NAMES.get(k, str(k)): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        "classification_report": report,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

    return {
        "model": model,
        "y_pred_cv": y_pred_cv,
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
        "report": report_str,
        "metrics": metrics,
    }


def predict_sleep_stage(model: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict class labels and probabilities."""
    return model.predict(X), model.predict_proba(X)


def load_model(model_path: str) -> Pipeline:
    """Load a saved model."""
    return joblib.load(model_path)
