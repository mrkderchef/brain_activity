"""Train the local model from saved .npy feature arrays."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.training import load_saved_features, predict_sleep_stage, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sleep stage classifier from .npy features")
    parser.add_argument("--features-path", default=os.path.join("outputs", "X_features.npy"))
    parser.add_argument("--labels-path", default=os.path.join("outputs", "y_labels.npy"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    X, y = load_saved_features(args.features_path, args.labels_path)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    results = train_and_evaluate(
        X,
        y,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
        parallel_jobs=args.parallel_jobs,
    )

    if not args.skip_plots:
        from sleep_stage_prediction.visualization import create_all_visualizations

        y_pred, y_proba = predict_sleep_stage(results["model"], X)
        tvb_dir = create_all_visualizations(
            X,
            y,
            y_pred,
            y_proba,
            results["feature_importances"],
            args.output_dir,
        )
        print(f"Saved visualizations and TVB export to {tvb_dir}")


if __name__ == "__main__":
    main()
