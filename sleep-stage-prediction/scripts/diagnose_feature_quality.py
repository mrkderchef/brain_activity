"""Diagnose dropped/non-finite feature rows in extracted EEG features."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import mne


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.data_loader import (
    find_eeg_files,
    get_labels_for_session,
    load_sleep_stages,
)
def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose feature-quality issues per recording and epoch")
    parser.add_argument("--bids-root", required=True, help="Path to ds003768")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_DIR, "outputs", "diagnostics"))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of recordings to inspect",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stages_df = load_sleep_stages(os.path.join(args.bids_root, "sourcedata"))
    eeg_files = find_eeg_files(args.bids_root)
    if args.limit is not None:
        eeg_files = eeg_files[: args.limit]

    recording_rows: list[dict] = []
    bad_epoch_rows: list[dict] = []

    for idx, record in enumerate(eeg_files, start=1):
        labels = get_labels_for_session(stages_df, record["subject"], record["session"])
        if len(labels) == 0:
            continue

        print(f"[{idx}/{len(eeg_files)}] {record['subject_str']} / {record['session']}")
        raw = mne.io.read_raw_brainvision(record["vhdr_path"], preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        samples_per_epoch = int(30.0 * sfreq)
        signal_epochs = raw.n_times // samples_per_epoch
        aligned_epochs = min(signal_epochs, len(labels))

        dropped_nonfinite = 0
        raw_nonfinite_epochs = 0
        total_power_zero_epochs = 0

        for epoch_idx in range(aligned_epochs):
            start = epoch_idx * samples_per_epoch
            end = start + samples_per_epoch
            epoch_data = raw.get_data(start=start, stop=end)
            raw_nonfinite = int(np.size(epoch_data) - np.isfinite(epoch_data).sum())
            total_abs = float(np.sum(np.abs(epoch_data[np.isfinite(epoch_data)])))

            if raw_nonfinite > 0:
                raw_nonfinite_epochs += 1
                dropped_nonfinite += 1
                bad_epoch_rows.append(
                    {
                        "subject": record["subject_str"],
                        "session": record["session"],
                        "epoch_index": epoch_idx,
                        "label": int(labels[epoch_idx]),
                        "raw_nonfinite_values": raw_nonfinite,
                        "epoch_abs_sum": total_abs,
                    }
                )
            if total_abs == 0.0:
                total_power_zero_epochs += 1

        recording_rows.append(
            {
                "subject": record["subject_str"],
                "session": record["session"],
                "label_epochs": int(len(labels)),
                "signal_epochs": int(signal_epochs),
                "aligned_epochs": int(aligned_epochs),
                "dropped_nonfinite_epochs": int(dropped_nonfinite),
                "raw_nonfinite_epochs": int(raw_nonfinite_epochs),
                "zero_energy_epochs": int(total_power_zero_epochs),
            }
        )

    recording_df = pd.DataFrame(recording_rows)
    bad_epoch_df = pd.DataFrame(bad_epoch_rows)

    recording_csv = os.path.join(args.output_dir, "feature_quality_by_recording.csv")
    epoch_csv = os.path.join(args.output_dir, "nonfinite_epochs.csv")
    summary_json = os.path.join(args.output_dir, "feature_quality_summary.json")

    recording_df.to_csv(recording_csv, index=False)
    bad_epoch_df.to_csv(epoch_csv, index=False)

    summary = {
        "recordings_checked": int(len(recording_df)),
        "recordings_with_nonfinite_epochs": int((recording_df["dropped_nonfinite_epochs"] > 0).sum())
        if not recording_df.empty
        else 0,
        "total_nonfinite_epochs": int(recording_df["dropped_nonfinite_epochs"].sum()) if not recording_df.empty else 0,
        "total_raw_nonfinite_epochs": int(recording_df["raw_nonfinite_epochs"].sum()) if not recording_df.empty else 0,
        "total_zero_energy_epochs": int(recording_df["zero_energy_epochs"].sum()) if not recording_df.empty else 0,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved recording report: {recording_csv}")
    print(f"Saved epoch report: {epoch_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
