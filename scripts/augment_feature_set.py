"""Add sleep-relevant ratio and entropy features to band-power arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURE_NAMES = [
    "delta_mean",
    "delta_std",
    "delta_relative",
    "theta_mean",
    "theta_std",
    "theta_relative",
    "alpha_mean",
    "alpha_std",
    "alpha_relative",
    "sigma_mean",
    "sigma_std",
    "sigma_relative",
    "beta_mean",
    "beta_std",
    "beta_relative",
]

ADDED_FEATURE_NAMES = [
    "delta_theta_ratio",
    "alpha_theta_ratio",
    "sigma_delta_ratio",
    "beta_alpha_ratio",
    "slow_fast_ratio",
    "relative_band_entropy",
]


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.where(np.abs(denominator) > 1e-12, denominator, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment band-power features with ratios")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    X = np.load(args.features_path).astype(np.float64)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")
    if X.shape[1] != len(BASE_FEATURE_NAMES):
        raise ValueError(f"Expected {len(BASE_FEATURE_NAMES)} base features, found {X.shape[1]}")

    rel = X[:, [2, 5, 8, 11, 14]]
    rel_sum = np.sum(rel, axis=1, keepdims=True)
    rel_norm = rel / np.where(rel_sum > 1e-12, rel_sum, 1.0)
    entropy = -np.sum(rel_norm * np.log(np.clip(rel_norm, 1e-12, None)), axis=1)

    delta = X[:, 0]
    theta = X[:, 3]
    alpha = X[:, 6]
    sigma = X[:, 9]
    beta = X[:, 12]
    ratio_features = np.column_stack(
        [
            safe_ratio(delta, theta),
            safe_ratio(alpha, theta),
            safe_ratio(sigma, delta),
            safe_ratio(beta, alpha),
            safe_ratio(delta + theta, alpha + beta),
            entropy,
        ]
    )
    ratio_features = np.nan_to_num(ratio_features, nan=0.0, posinf=0.0, neginf=0.0)

    X_augmented = np.column_stack([X, ratio_features])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_features.npy", X_augmented)
    np.save(output_dir / "y_labels.npy", y)
    metadata.to_csv(output_dir / "epoch_metadata.csv", index=False)

    feature_names = BASE_FEATURE_NAMES + ADDED_FEATURE_NAMES
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)
    with open(output_dir / "augmentation_audit.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_features": args.features_path,
                "n_rows": int(X_augmented.shape[0]),
                "base_features": int(X.shape[1]),
                "added_features": ADDED_FEATURE_NAMES,
                "n_features": int(X_augmented.shape[1]),
            },
            handle,
            indent=2,
        )

    print(f"Saved augmented feature set to {output_dir.resolve()}")
    print(f"  X_features.npy: {X_augmented.shape}")


if __name__ == "__main__":
    main()
