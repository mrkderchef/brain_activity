"""
train_model.py – Training und Evaluation des Schlafphasen-Klassifikators.

Modell: Random Forest + Gradient Boosting Ensemble
Features: EEG-Frequenzband-Leistungen (Delta, Theta, Alpha, Sigma, Beta)
Target: Schlafstadium (W=0, N1=1, N2=2, N3=3, REM=4)
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
)
import joblib


SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ---------------------------------------------------------------------------
# Modell erstellen
# ---------------------------------------------------------------------------

def create_model() -> Pipeline:
    """Erstellt eine sklearn-Pipeline mit Scaler und Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", rf),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    output_dir: str = "outputs",
) -> dict:
    """
    Trainiert das Modell mit stratifizierter K-Fold Cross-Validation
    und gibt Metriken + trainiertes Modell zurück.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = create_model()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-Validation Predictions für vollständige Confusion Matrix
    print(f"\n{'='*60}")
    print(f"Starte {n_splits}-Fold Stratified Cross-Validation ...")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Klassen-Verteilung: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'='*60}\n")

    y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    # Metriken
    acc = accuracy_score(y, y_pred_cv)
    kappa = cohen_kappa_score(y, y_pred_cv)
    target_names = [SLEEP_STAGE_NAMES.get(i, f"Stage-{i}") for i in sorted(np.unique(y))]
    report = classification_report(y, y_pred_cv, target_names=target_names, output_dict=True)
    report_str = classification_report(y, y_pred_cv, target_names=target_names)
    cm = confusion_matrix(y, y_pred_cv)

    print("Klassifikationsbericht:")
    print(report_str)
    print(f"Gesamt-Accuracy:    {acc:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}")

    # Finales Modell auf allen Daten trainieren
    print("\nTrainiere finales Modell auf allen Daten ...")
    model.fit(X, y)

    # Speichern
    model_path = os.path.join(output_dir, "sleep_stage_model.joblib")
    joblib.dump(model, model_path)
    print(f"Modell gespeichert: {model_path}")

    # Feature Importances (vom Random Forest)
    rf_model = model.named_steps["classifier"]
    feature_importances = rf_model.feature_importances_

    # Metriken als JSON speichern
    metrics = {
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {SLEEP_STAGE_NAMES.get(k, str(k)): int(v)
                                for k, v in zip(*np.unique(y, return_counts=True))},
        "classification_report": report,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriken gespeichert: {metrics_path}")

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


def predict_sleep_stage(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vorhersage der Schlafphasen und Wahrscheinlichkeiten."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba


def load_model(model_path: str):
    """Lädt ein gespeichertes Modell."""
    return joblib.load(model_path)
