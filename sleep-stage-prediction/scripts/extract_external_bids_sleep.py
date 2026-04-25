"""Extract sleep-stage features from an external BIDS EEG dataset."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.external_bids_sleep import (
    extract_ds006695_set_recording,
    extract_external_recording,
    find_eeg_recordings,
    find_events_file,
    write_external_extraction_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract band-power features from external BIDS sleep EEG datasets"
    )
    parser.add_argument("--bids-root", required=True, help="Path to the external BIDS dataset root")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID saved into metadata, e.g. NM000185")
    parser.add_argument("--output-dir", required=True, help="Output directory for extracted arrays")
    parser.add_argument("--epoch-duration", type=float, default=30.0)
    parser.add_argument("--label-column", default=None, help="Optional events.tsv label column override")
    parser.add_argument(
        "--preset",
        choices=["auto", "ds006695"],
        default="auto",
        help="Dataset-specific numeric sleep-stage mapping",
    )
    parser.add_argument("--limit-recordings", type=int, default=None, help="Small smoke-test limit")
    parser.add_argument("--limit-epochs", type=int, default=None, help="Small smoke-test epoch limit per recording")
    args = parser.parse_args()

    recordings = find_eeg_recordings(args.bids_root)
    if args.limit_recordings is not None:
        recordings = recordings[: args.limit_recordings]
    if not recordings:
        raise FileNotFoundError(f"No supported EEG recordings found under {args.bids_root}")

    all_X = []
    all_y = []
    metadata_rows = []
    summary_rows = []

    for eeg_path in recordings:
        events_path = find_events_file(eeg_path)
        if events_path is None:
            if args.preset == "ds006695" and eeg_path.suffix.lower() == ".set":
                print(f"Loading {eeg_path} with EEG.VisualHypnogram")
                X_record, y_record, record_metadata, summary = extract_ds006695_set_recording(
                    eeg_path=eeg_path,
                    dataset_id=args.dataset_id,
                    epoch_duration=args.epoch_duration,
                    max_epochs=args.limit_epochs,
                )
            else:
                print(f"Skipping {eeg_path}: no matching events.tsv")
                continue
        else:
            print(f"Loading {eeg_path}")
            X_record, y_record, record_metadata, summary = extract_external_recording(
                eeg_path=eeg_path,
                events_path=events_path,
                dataset_id=args.dataset_id,
                epoch_duration=args.epoch_duration,
                label_column=args.label_column,
                preset=args.preset,
                max_epochs=args.limit_epochs,
            )
        summary_rows.append(summary)
        metadata_rows.extend(record_metadata)
        if len(y_record):
            all_X.append(X_record)
            all_y.append(y_record)
        print(f"  Kept {len(y_record)} epochs")

    if not all_X:
        raise RuntimeError("No external features were extracted. Check event labels and raw files.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    write_external_extraction_outputs(
        output_dir=args.output_dir,
        X=X,
        y=y,
        metadata_rows=metadata_rows,
        summary_rows=summary_rows,
        dataset_id=args.dataset_id,
        preset=args.preset,
    )

    print(f"Saved external features to {os.path.abspath(args.output_dir)}")
    print(f"  X_features.npy: {X.shape}")
    print(f"  y_labels.npy: {y.shape}")


if __name__ == "__main__":
    main()
