"""Normalize extracted band-power features with recording-aware scaling."""

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


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Median/IQR z-score with a standard-deviation fallback."""
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-12, scale, std)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - center) / scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize extracted sleep-stage feature arrays")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument(
        "--method",
        choices=["global-log-robust", "group-log-robust"],
        default="group-log-robust",
    )
    args = parser.parse_args()

    X = np.load(args.features_path).astype(np.float64)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    # Feature layout is [band_mean, band_std, band_relative] repeated per band.
    absolute_columns = np.array([idx for idx in range(X.shape[1]) if idx % 3 in (0, 1)])
    X[:, absolute_columns] = np.log1p(np.clip(X[:, absolute_columns], a_min=0.0, a_max=None))

    group_ids = derive_group_ids(metadata, args.group_column).astype(str)
    X_norm = np.empty_like(X)
    if args.method == "global-log-robust":
        X_norm = robust_zscore(X)
    else:
        for group_id in sorted(group_ids.unique()):
            mask = group_ids.eq(group_id).to_numpy()
            X_norm[mask] = robust_zscore(X[mask])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_features.npy", X_norm)
    np.save(output_dir / "y_labels.npy", y)
    metadata_out = metadata.copy()
    metadata_out["normalization_group"] = group_ids
    metadata_out.to_csv(output_dir / "epoch_metadata.csv", index=False)

    audit = {
        "source_features": args.features_path,
        "source_labels": args.labels_path,
        "source_metadata": args.metadata_path,
        "method": args.method,
        "group_column": args.group_column,
        "n_rows": int(X_norm.shape[0]),
        "n_features": int(X_norm.shape[1]),
        "n_groups": int(group_ids.nunique()),
        "absolute_columns_log1p": absolute_columns.tolist(),
    }
    with open(output_dir / "normalization_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Saved normalized feature set to {output_dir.resolve()}")
    print(f"  X_features.npy: {X_norm.shape}")
    print(f"  groups: {group_ids.nunique()}")


if __name__ == "__main__":
    main()
