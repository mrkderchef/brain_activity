"""Run the local end-to-end pipeline."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.data_loader import (
    check_data_availability,
    find_eeg_files,
    get_labels_for_session,
    load_eeg_raw,
    load_sleep_stages,
)
from sleep_stage_prediction.feature_extraction import extract_features_from_raw, generate_synthetic_features
from sleep_stage_prediction.training import predict_sleep_stage, train_and_evaluate
from sleep_stage_prediction.visualization import create_all_visualizations


def get_default_bids_root(project_dir: str) -> str | None:
    candidates = [
        os.path.join(project_dir, "..", "ds003768-master"),
        os.path.join(project_dir, "..", "ds003768"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return None


def run_demo_mode(bids_root: str) -> tuple[np.ndarray, np.ndarray]:
    stages_df = load_sleep_stages(os.path.join(bids_root, "sourcedata"))
    y = stages_df["stage_int"].to_numpy()
    X = generate_synthetic_features(y)
    return X, y


def run_real_mode(bids_root: str) -> tuple[np.ndarray, np.ndarray]:
    stages_df = load_sleep_stages(os.path.join(bids_root, "sourcedata"))
    eeg_files = find_eeg_files(bids_root)

    all_X = []
    all_y = []
    for record in eeg_files:
        labels = get_labels_for_session(stages_df, record["subject"], record["session"])
        if len(labels) == 0:
            continue

        print(f"Loading {record['subject_str']} / {record['session']} ...")
        raw = load_eeg_raw(record["vhdr_path"])
        X_record = extract_features_from_raw(raw)
        n_epochs = min(X_record.shape[0], len(labels))
        all_X.append(X_record[:n_epochs])
        all_y.append(labels[:n_epochs])
        print(f"  Extracted {n_epochs} epochs")

    if not all_X:
        raise RuntimeError("No EEG features extracted from real data.")

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local sleep stage prediction pipeline")
    parser.add_argument("--bids-root", default=None, help="Path to ds003768")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--demo", action="store_true", help="Force demo mode with synthetic features")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    args = parser.parse_args()

    bids_root = args.bids_root or get_default_bids_root(PROJECT_DIR)
    if bids_root is None:
        raise FileNotFoundError("Could not find ds003768. Pass --bids-root explicitly.")

    availability = check_data_availability(bids_root)
    use_demo = args.demo or len(availability["available"]) == 0
    X, y = run_demo_mode(bids_root) if use_demo else run_real_mode(bids_root)

    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "X_features.npy"), X)
    np.save(os.path.join(args.output_dir, "y_labels.npy"), y)

    results = train_and_evaluate(
        X,
        y,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
        parallel_jobs=args.parallel_jobs,
    )
    y_pred, y_proba = predict_sleep_stage(results["model"], X)
    tvb_dir = create_all_visualizations(X, y, y_pred, y_proba, results["feature_importances"], args.output_dir)

    print("\nSummary")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Cohen's kappa: {results['kappa']:.4f}")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"  TVB export: {tvb_dir}")


if __name__ == "__main__":
    main()
