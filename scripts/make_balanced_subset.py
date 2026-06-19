"""Create a class-balanced subset from extracted feature arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a capped class-balanced feature subset")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Requested max rows per class. Defaults to the rarest available class.",
    )
    parser.add_argument(
        "--classes",
        default="0,1,2,3,4",
        help="Comma-separated stage ids to include, e.g. 0,1,2,3,4",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != y.shape[0] or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    classes = np.array([int(value.strip()) for value in args.classes.split(",") if value.strip()])
    available_counts = {int(label): int(np.sum(y == label)) for label in classes}
    missing = [label for label, count in available_counts.items() if count == 0]
    if missing:
        missing_names = [STAGE_NAMES.get(label, str(label)) for label in missing]
        raise RuntimeError(f"Cannot balance because these classes are missing: {missing_names}")

    target = args.target_per_class or min(available_counts.values())
    target = min(target, min(available_counts.values()))

    rng = np.random.default_rng(args.seed)
    chosen_indices = []
    for label in classes:
        indices = np.where(y == label)[0]
        chosen = rng.choice(indices, size=target, replace=False)
        chosen_indices.extend(chosen.tolist())

    chosen_indices = np.array(chosen_indices, dtype=int)
    rng.shuffle(chosen_indices)

    X_balanced = X[chosen_indices]
    y_balanced = y[chosen_indices]
    metadata_balanced = metadata.iloc[chosen_indices].reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_features.npy", X_balanced)
    np.save(output_dir / "y_labels.npy", y_balanced)
    metadata_balanced.to_csv(output_dir / "epoch_metadata.csv", index=False)

    class_distribution = {
        STAGE_NAMES.get(int(label), str(int(label))): int(count)
        for label, count in zip(*np.unique(y_balanced, return_counts=True))
    }
    dataset_by_class = (
        metadata_balanced.assign(label=y_balanced)
        .groupby(["label", "dataset_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    dataset_by_class["label_name"] = dataset_by_class["label"].map(STAGE_NAMES)

    audit = {
        "source_features": args.features_path,
        "source_labels": args.labels_path,
        "source_metadata": args.metadata_path,
        "seed": args.seed,
        "requested_target_per_class": args.target_per_class,
        "effective_target_per_class": int(target),
        "classes": [STAGE_NAMES.get(int(label), str(int(label))) for label in classes],
        "source_class_distribution": {
            STAGE_NAMES.get(label, str(label)): count for label, count in available_counts.items()
        },
        "balanced_class_distribution": class_distribution,
        "n_rows": int(len(y_balanced)),
        "n_features": int(X_balanced.shape[1]),
    }
    with open(output_dir / "balance_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    dataset_by_class.to_csv(output_dir / "dataset_by_class.csv", index=False)

    print(f"Saved balanced subset to {output_dir.resolve()}")
    print(f"  target per class: {target}")
    print(f"  X_features.npy: {X_balanced.shape}")
    print(f"  class distribution: {class_distribution}")


if __name__ == "__main__":
    main()
