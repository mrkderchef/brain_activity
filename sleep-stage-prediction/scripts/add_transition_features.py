"""Add label-free rolling transition features within each recording/subject."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids


def parse_int_list(value: str) -> list[int]:
    radii = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not radii or any(radius < 1 for radius in radii):
        raise ValueError("--radii must contain positive integers")
    return radii


def sort_group_indices(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    sort_frame = metadata.iloc[indices].copy()
    sort_frame["_row_index"] = indices
    sort_columns = []
    for column in ["epoch_start_time_sec", "epoch_index"]:
        if column in sort_frame.columns:
            sort_frame[column] = pd.to_numeric(sort_frame[column], errors="coerce")
            sort_columns.append(column)
    if sort_columns:
        sort_frame = sort_frame.sort_values(sort_columns + ["_row_index"], kind="mergesort")
    return sort_frame["_row_index"].to_numpy(dtype=int)


def window_slope(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(values.shape[1], dtype=values.dtype)
    positions = np.arange(len(values), dtype=np.float64)
    positions = positions - positions.mean()
    denominator = np.sum(positions**2)
    centered_values = values - values.mean(axis=0)
    return (positions[:, None] * centered_values).sum(axis=0) / denominator


def add_group_transition_features(group_X: np.ndarray, radii: list[int]) -> np.ndarray:
    blocks = [group_X]
    for radius in radii:
        padded = np.pad(group_X, ((radius, radius), (0, 0)), mode="edge")
        means = np.empty_like(group_X)
        stds = np.empty_like(group_X)
        slopes = np.empty_like(group_X)
        deviations = np.empty_like(group_X)
        for row_idx in range(len(group_X)):
            window = padded[row_idx : row_idx + 2 * radius + 1]
            means[row_idx] = window.mean(axis=0)
            stds[row_idx] = window.std(axis=0)
            slopes[row_idx] = window_slope(window)
            deviations[row_idx] = group_X[row_idx] - means[row_idx]
        blocks.extend([means, stds, slopes, deviations])
    return np.column_stack(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append rolling transition features")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--radii", default="2,4", help="Centered windows as radii, e.g. 2,4 means 5 and 9 epochs")
    args = parser.parse_args()

    X = np.load(args.features_path).astype(np.float64)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    radii = parse_int_list(args.radii)
    group_ids = derive_group_ids(metadata, args.group_column).astype(str)
    n_features = X.shape[1] * (1 + 4 * len(radii))
    X_transition = np.empty((X.shape[0], n_features), dtype=np.float64)

    for group_id in sorted(group_ids.unique()):
        indices = np.where(group_ids.to_numpy() == group_id)[0]
        ordered = sort_group_indices(metadata, indices)
        X_transition[ordered] = add_group_transition_features(X[ordered], radii)

    X_transition = np.nan_to_num(X_transition, nan=0.0, posinf=0.0, neginf=0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_features.npy", X_transition)
    np.save(output_dir / "y_labels.npy", y)
    metadata_out = metadata.copy()
    metadata_out["transition_group"] = group_ids
    metadata_out.to_csv(output_dir / "epoch_metadata.csv", index=False)

    audit = {
        "source_features": args.features_path,
        "source_labels": args.labels_path,
        "source_metadata": args.metadata_path,
        "radii": radii,
        "window_lengths_epochs": [2 * radius + 1 for radius in radii],
        "n_rows": int(X_transition.shape[0]),
        "source_features_count": int(X.shape[1]),
        "n_features": int(X_transition.shape[1]),
        "n_groups": int(group_ids.nunique()),
        "feature_blocks": ["current"]
        + [f"{name}_radius_{radius}" for radius in radii for name in ["mean", "std", "slope", "current_minus_mean"]],
    }
    with open(output_dir / "transition_feature_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Saved transition feature set to {output_dir.resolve()}")
    print(f"  X_features.npy: {X_transition.shape}")
    print(f"  groups: {group_ids.nunique()}")


if __name__ == "__main__":
    main()
