"""Evaluate N1-focused strategies under group-aware cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


N1_LABEL = 1


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_model(n1_weight: float, parallel_jobs: int) -> Pipeline:
    class_weight = {label: 1.0 for label in SLEEP_STAGE_NAMES}
    class_weight[N1_LABEL] = float(n1_weight)
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=3,
        class_weight=class_weight,
        random_state=42,
        n_jobs=parallel_jobs,
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def predict_with_n1_threshold(
    classes: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    n1_position = int(np.where(classes == N1_LABEL)[0][0])
    predictions = classes[np.argmax(probabilities, axis=1)]
    n1_mask = probabilities[:, n1_position] >= threshold
    predictions[n1_mask] = N1_LABEL
    return predictions.astype(int)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "n1_precision": float(report["N1"]["precision"]),
        "n1_recall": float(report["N1"]["recall"]),
        "n1_f1": float(report["N1"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N1-focused group-CV experiments")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--n1-weights", default="1,2,3,4,6")
    parser.add_argument("--n1-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    labels = sorted(int(label) for label in np.unique(y))
    if N1_LABEL not in labels:
        raise RuntimeError("N1 is not present in the supplied labels")

    groups = derive_group_ids(metadata, args.group_column).astype(str)
    if groups.nunique() < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} groups, found {groups.nunique()}")

    weights = parse_float_list(args.n1_weights)
    thresholds = parse_float_list(args.n1_thresholds)
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    experiments = {}
    for weight in weights:
        y_pred = np.empty_like(y)
        y_proba = np.empty((len(y), len(labels)), dtype=float)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups.to_numpy()), start=1):
            model = make_model(weight, args.parallel_jobs)
            model.fit(X[train_idx], y[train_idx])
            fold_classes = model.named_steps["classifier"].classes_.astype(int)
            if list(fold_classes) != labels:
                raise RuntimeError(f"Fold {fold} is missing one or more classes: {fold_classes}")
            y_pred[test_idx] = model.predict(X[test_idx])
            y_proba[test_idx] = model.predict_proba(X[test_idx])

        base_summary = summarize_predictions(y, y_pred, labels)
        experiment_key = f"n1_weight_{weight:g}"
        experiments[experiment_key] = {
            "n1_weight": float(weight),
            "argmax": base_summary,
            "thresholds": {},
        }
        rows.append(
            {
                "strategy": "argmax",
                "n1_weight": weight,
                "n1_threshold": np.nan,
                **{key: base_summary[key] for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "cohen_kappa",
                    "macro_f1",
                    "n1_precision",
                    "n1_recall",
                    "n1_f1",
                ]},
            }
        )

        for threshold in thresholds:
            threshold_pred = predict_with_n1_threshold(np.array(labels), y_proba, threshold)
            threshold_summary = summarize_predictions(y, threshold_pred, labels)
            experiments[experiment_key]["thresholds"][f"{threshold:g}"] = threshold_summary
            rows.append(
                {
                    "strategy": "n1_threshold",
                    "n1_weight": weight,
                    "n1_threshold": threshold,
                    **{key: threshold_summary[key] for key in [
                        "accuracy",
                        "balanced_accuracy",
                        "cohen_kappa",
                        "macro_f1",
                        "n1_precision",
                        "n1_recall",
                        "n1_f1",
                    ]},
                }
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["n1_f1", "balanced_accuracy"], ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "n1_focus_summary.csv", index=False)

    result = {
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "group_column": args.group_column,
        "n_groups": int(groups.nunique()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            SLEEP_STAGE_NAMES.get(int(label), str(int(label))): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
        "n1_weights": weights,
        "n1_thresholds": thresholds,
        "best_by_n1_f1": summary.iloc[0].to_dict(),
        "experiments": experiments,
    }
    with open(output_dir / "n1_focus_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved N1-focused evaluation to {output_dir}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
