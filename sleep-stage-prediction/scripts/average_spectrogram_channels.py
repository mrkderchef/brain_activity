"""Average multi-channel spectrogram tensors into a single-channel tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Average spectrogram channels")
    parser.add_argument("--spectrograms-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    X = np.load(args.spectrograms_path, mmap_mode="r")
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.ndim != 4:
        raise ValueError(f"Expected spectrogram tensor with shape (rows, channels, freqs, times), got {X.shape}")
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Spectrogram, labels, and metadata row counts must match")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_avg = np.asarray(X.mean(axis=1, keepdims=True), dtype=np.float32)
    np.save(output_dir / "X_spectrograms.npy", X_avg)
    np.save(output_dir / "y_labels.npy", y)
    metadata = metadata.copy()
    metadata["channel_transform"] = f"mean_of_{X.shape[1]}_channels"
    metadata.to_csv(output_dir / "epoch_metadata.csv", index=False)
    audit = {
        "source_spectrograms_path": args.spectrograms_path,
        "source_shape": list(X.shape),
        "output_shape": list(X_avg.shape),
        "transform": "channel_mean",
    }
    with open(output_dir / "channel_average_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(f"Saved averaged spectrograms to {output_dir.resolve()}")
    print(f"  X_spectrograms.npy: {X_avg.shape}")


if __name__ == "__main__":
    main()
