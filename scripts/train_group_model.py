"""Train and evaluate with subject/recording-aware cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES, create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with group-aware cross-validation")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    groups = derive_group_ids(metadata, args.group_column).astype(str)
    n_groups = groups.nunique()
    if n_groups < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} groups, found {n_groups}")

    model = create_model(parallel_jobs=args.parallel_jobs)
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    print(f"\n{'=' * 60}")
    print(f"Running {args.n_splits}-fold stratified group CV")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Groups: {n_groups}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'=' * 60}\n")

    y_pred_cv = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        groups=groups.to_numpy(),
        n_jobs=args.parallel_jobs,
    )

    labels = sorted(int(label) for label in np.unique(y))
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y,
        y_pred_cv,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y,
        y_pred_cv,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y, y_pred_cv, labels=labels)

    accuracy = accuracy_score(y, y_pred_cv)
    balanced_accuracy = balanced_accuracy_score(y, y_pred_cv)
    kappa = cohen_kappa_score(y, y_pred_cv)

    print("Classification report:")
    print(report_str)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")

    print("\nTraining final model on the full dataset...")
    model.fit(X, y)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "sleep_stage_model.joblib")
    metadata_with_groups = metadata.copy()
    metadata_with_groups["cv_group"] = groups
    metadata_with_groups.to_csv(output_dir / "epoch_metadata.csv", index=False)

    metrics = {
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "group_column": args.group_column,
        "n_groups": int(n_groups),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "cohen_kappa": float(kappa),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            SLEEP_STAGE_NAMES.get(int(k), str(int(k))): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
