"""Combine multiple extracted feature directories while preserving epoch metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_metadata(feature_dir: Path, n_rows: int, fallback_dataset_id: str) -> pd.DataFrame:
    metadata_path = feature_dir / "epoch_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.DataFrame({"epoch_index": np.arange(n_rows)})

    if "dataset_id" not in metadata.columns:
        metadata.insert(0, "dataset_id", fallback_dataset_id)
    if len(metadata) != n_rows:
        raise ValueError(f"{metadata_path} has {len(metadata)} rows but features have {n_rows}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine extracted sleep-stage feature directories")
    parser.add_argument("--feature-dir", action="append", required=True, help="Directory with X/y .npy files")
    parser.add_argument("--dataset-id", action="append", default=[], help="Optional fallback ID per feature dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    feature_dirs = [Path(path) for path in args.feature_dir]
    dataset_ids = args.dataset_id or [path.name for path in feature_dirs]
    if len(dataset_ids) != len(feature_dirs):
        raise ValueError("--dataset-id must be supplied once per --feature-dir when used")

    X_parts = []
    y_parts = []
    metadata_parts = []

    for feature_dir, dataset_id in zip(feature_dirs, dataset_ids):
        X = np.load(feature_dir / "X_features.npy")
        y = np.load(feature_dir / "y_labels.npy")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{feature_dir} has mismatched X/y rows: {X.shape[0]} vs {y.shape[0]}")

        metadata = load_metadata(feature_dir, X.shape[0], dataset_id)
        X_parts.append(X)
        y_parts.append(y)
        metadata_parts.append(metadata)

    X_combined = np.concatenate(X_parts, axis=0)
    y_combined = np.concatenate(y_parts, axis=0)
    metadata_combined = pd.concat(metadata_parts, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_features.npy", X_combined)
    np.save(output_dir / "y_labels.npy", y_combined)
    metadata_combined.to_csv(output_dir / "epoch_metadata.csv", index=False)

    audit = {
        "feature_dirs": [str(path) for path in feature_dirs],
        "n_rows": int(len(y_combined)),
        "n_features": int(X_combined.shape[1]),
        "class_distribution": {
            str(int(k)): int(v) for k, v in zip(*np.unique(y_combined, return_counts=True))
        },
        "dataset_distribution": metadata_combined["dataset_id"].value_counts().to_dict(),
    }
    with open(output_dir / "combine_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Saved combined features to {output_dir.resolve()}")
    print(f"  X_features.npy: {X_combined.shape}")
    print(f"  y_labels.npy: {y_combined.shape}")
    print(f"  datasets: {audit['dataset_distribution']}")


if __name__ == "__main__":
    main()
