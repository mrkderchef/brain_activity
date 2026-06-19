"""Evaluate cross-dataset transfer from combined feature outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.training import SLEEP_STAGE_NAMES, create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on one dataset and test on another")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--shared-labels-only", action="store_true")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if len(metadata) != len(y):
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match labels ({len(y)})")

    train_mask = metadata["dataset_id"].astype(str).eq(args.train_dataset).to_numpy().copy()
    test_mask = metadata["dataset_id"].astype(str).eq(args.test_dataset).to_numpy().copy()
    if args.shared_labels_only:
        shared_labels = np.intersect1d(np.unique(y[train_mask]), np.unique(y[test_mask]))
        train_mask &= np.isin(y, shared_labels)
        test_mask &= np.isin(y, shared_labels)
    else:
        shared_labels = np.unique(y[test_mask])

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    if len(y_train) == 0 or len(y_test) == 0:
        raise RuntimeError("Empty train or test split after filtering")

    model = create_model(parallel_jobs=args.parallel_jobs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(int(label) for label in np.unique(np.concatenate([y_test, y_pred])))
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    results = {
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "shared_labels_only": bool(args.shared_labels_only),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "labels": [SLEEP_STAGE_NAMES.get(int(label), str(int(label))) for label in shared_labels],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "classification_report": report,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(report_str)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Cohen's kappa: {results['cohen_kappa']:.4f}")
    print(f"Saved transfer metrics to {output_path}")


if __name__ == "__main__":
    main()
