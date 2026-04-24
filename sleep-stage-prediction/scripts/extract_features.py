"""Extract EEG features from ds003768 into local .npy files."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.data_loader import (
    find_eeg_files,
    get_labels_for_session,
    load_eeg_raw,
    load_sleep_stages,
)
from sleep_stage_prediction.feature_extraction import extract_features_from_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EEG band-power features from ds003768")
    parser.add_argument("--bids-root", required=True, help="Path to the ds003768 dataset root")
    parser.add_argument("--output-dir", default="outputs", help="Where X_features.npy and y_labels.npy are saved")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stages_df = load_sleep_stages(os.path.join(args.bids_root, "sourcedata"))
    eeg_files = find_eeg_files(args.bids_root)

    all_X = []
    all_y = []
    dropped_epoch_records = []
    extraction_summary = []

    for record in eeg_files:
        labels = get_labels_for_session(stages_df, record["subject"], record["session"])
        if len(labels) == 0:
            continue

        print(f"Loading {record['subject_str']} / {record['session']} ...")
        raw = load_eeg_raw(record["vhdr_path"])
        X_record = extract_features_from_raw(raw)
        n_epochs = min(X_record.shape[0], len(labels))
        X_aligned = X_record[:n_epochs]
        y_aligned = labels[:n_epochs]
        finite_mask = np.all(np.isfinite(X_aligned), axis=1)

        if not np.all(finite_mask):
            bad_indices = np.where(~finite_mask)[0]
            for bad_idx in bad_indices:
                dropped_epoch_records.append(
                    {
                        "subject": record["subject_str"],
                        "session": record["session"],
                        "epoch_index": int(bad_idx),
                        "label": int(y_aligned[bad_idx]),
                    }
                )

        all_X.append(X_aligned[finite_mask])
        all_y.append(y_aligned[finite_mask])
        extraction_summary.append(
            {
                "subject": record["subject_str"],
                "session": record["session"],
                "label_epochs": int(len(labels)),
                "aligned_epochs": int(n_epochs),
                "kept_epochs": int(np.sum(finite_mask)),
                "dropped_nonfinite_epochs": int(np.sum(~finite_mask)),
            }
        )
        print(f"  Extracted {n_epochs} epochs")

    if not all_X:
        raise RuntimeError("No features were extracted. Check dataset paths and EEG availability.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    np.save(os.path.join(output_dir, "X_features.npy"), X)
    np.save(os.path.join(output_dir, "y_labels.npy"), y)
    pd.DataFrame(extraction_summary).to_csv(os.path.join(output_dir, "extraction_summary.csv"), index=False)
    pd.DataFrame(dropped_epoch_records).to_csv(os.path.join(output_dir, "dropped_epochs.csv"), index=False)
    with open(os.path.join(output_dir, "extraction_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_recordings_with_labels": len(extraction_summary),
                "total_kept_rows": int(len(y)),
                "total_dropped_nonfinite_rows": int(len(dropped_epoch_records)),
            },
            handle,
            indent=2,
        )
    print(f"Saved features to {output_dir}")
    print(f"  X_features.npy: {X.shape}")
    print(f"  y_labels.npy: {y.shape}")


if __name__ == "__main__":
    main()
