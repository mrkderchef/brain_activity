"""Add neighboring-epoch context features without crossing recording boundaries."""

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


def sort_group_indices(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Sort one recording/subject group by epoch time/index while preserving row IDs."""
    sort_frame = metadata.iloc[indices].copy()
    sort_frame["_row_index"] = indices
    sort_columns = []
    for column in ["epoch_start_time_sec", "epoch_index"]:
        if column in sort_frame.columns:
            sort_frame[column] = pd.to_numeric(sort_frame[column], errors="coerce")
            sort_columns.append(column)
    if not sort_columns:
        return indices
    sort_frame = sort_frame.sort_values(sort_columns + ["_row_index"], kind="mergesort")
    return sort_frame["_row_index"].to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append previous/next epoch feature context")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--window", type=int, default=1, help="Neighbor epochs on each side")
    parser.add_argument("--include-deltas", action="store_true")
    args = parser.parse_args()

    if args.window < 0:
        raise ValueError("--window must be non-negative")

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    group_ids = derive_group_ids(metadata, args.group_column).astype(str)
    X_context = np.empty((X.shape[0], X.shape[1] * (2 * args.window + 1)), dtype=X.dtype)
    for group_id in sorted(group_ids.unique()):
        group_indices = np.where(group_ids.to_numpy() == group_id)[0]
        ordered = sort_group_indices(metadata, group_indices)
        group_X = X[ordered]
        padded = np.pad(group_X, ((args.window, args.window), (0, 0)), mode="edge")
        context_rows = []
        for position in range(args.window, args.window + len(ordered)):
            context_rows.append(padded[position - args.window : position + args.window + 1].reshape(-1))
        X_context[ordered] = np.vstack(context_rows)

    if args.include_deltas and args.window >= 1:
        current_start = args.window * X.shape[1]
        current = X_context[:, current_start : current_start + X.shape[1]]
        previous = X_context[:, current_start - X.shape[1] : current_start]
        following = X_context[:, current_start + X.shape[1] : current_start + 2 * X.shape[1]]
        X_context = np.column_stack([X_context, current - previous, following - current])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_features.npy", X_context)
    np.save(output_dir / "y_labels.npy", y)
    metadata_out = metadata.copy()
    metadata_out["sequence_group"] = group_ids
    metadata_out.to_csv(output_dir / "epoch_metadata.csv", index=False)

    audit = {
        "source_features": args.features_path,
        "source_labels": args.labels_path,
        "source_metadata": args.metadata_path,
        "window": int(args.window),
        "include_deltas": bool(args.include_deltas),
        "n_rows": int(X_context.shape[0]),
        "source_features_count": int(X.shape[1]),
        "n_features": int(X_context.shape[1]),
        "n_groups": int(group_ids.nunique()),
    }
    with open(output_dir / "sequence_context_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Saved sequence-context feature set to {output_dir.resolve()}")
    print(f"  X_features.npy: {X_context.shape}")
    print(f"  groups: {group_ids.nunique()}")


if __name__ == "__main__":
    main()
