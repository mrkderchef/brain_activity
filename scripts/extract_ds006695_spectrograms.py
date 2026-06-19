"""Extract log-spectrogram epoch tensors from ds006695 EEGLAB files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.external_bids_sleep import (
    SLEEP_STAGE_NAMES,
    find_eeg_recordings,
    load_ds006695_hypnogram,
    load_ds006695_set_header,
)


def compute_epoch_spectrogram(
    epoch_data: np.ndarray,
    sfreq: float,
    nperseg: int,
    noverlap: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    channel_specs = []
    for channel in epoch_data:
        freqs, _, power = spectrogram(
            channel,
            fs=sfreq,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        keep = (freqs >= fmin) & (freqs <= fmax)
        log_power = np.log1p(power[keep])
        channel_specs.append(log_power.astype(np.float32))
    return np.stack(channel_specs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ds006695 log-spectrogram tensors")
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="ds006695")
    parser.add_argument("--epoch-duration", type=float, default=30.0)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--limit-recordings", type=int, default=None)
    parser.add_argument("--limit-epochs", type=int, default=None)
    args = parser.parse_args()

    recordings = [
        path for path in find_eeg_recordings(args.bids_root)
        if path.suffix.lower() == ".set"
    ]
    if args.limit_recordings is not None:
        recordings = recordings[: args.limit_recordings]
    if not recordings:
        raise FileNotFoundError(f"No EEGLAB .set recordings found under {args.bids_root}")

    specs = []
    labels_out = []
    metadata_rows = []
    summary_rows = []
    expected_shape = None

    for eeg_path in recordings:
        print(f"Loading {eeg_path}")
        labels = load_ds006695_hypnogram(eeg_path)
        header = load_ds006695_set_header(eeg_path)
        sfreq = float(header["srate"])
        n_samples = int(round(args.epoch_duration * sfreq))
        n_signal_epochs = int(header["pnts"] // n_samples)
        n_epochs = min(len(labels), n_signal_epochs)
        if args.limit_epochs is not None:
            n_epochs = min(n_epochs, args.limit_epochs)

        fdt_path = eeg_path.with_name(header["data_file"])
        signal = np.memmap(
            fdt_path,
            dtype="<f4",
            mode="r",
            shape=(header["nbchan"], header["pnts"]),
            order="F",
        )

        kept = 0
        skipped = 0
        for epoch_idx in range(n_epochs):
            start = epoch_idx * n_samples
            stop = start + n_samples
            epoch_data = np.asarray(signal[:, start:stop], dtype=np.float32)
            spec = compute_epoch_spectrogram(
                epoch_data=epoch_data,
                sfreq=sfreq,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            if expected_shape is None:
                expected_shape = spec.shape
            if spec.shape != expected_shape or not np.all(np.isfinite(spec)):
                skipped += 1
                continue

            label = int(labels[epoch_idx])
            specs.append(spec)
            labels_out.append(label)
            kept += 1
            metadata_rows.append(
                {
                    "dataset_id": args.dataset_id,
                    "recording": str(eeg_path.relative_to(eeg_path.parents[2])),
                    "events_file": "EEG.VisualHypnogram",
                    "epoch_index": int(epoch_idx),
                    "epoch_start_time_sec": float(epoch_idx * args.epoch_duration),
                    "label": label,
                    "label_name": SLEEP_STAGE_NAMES.get(label, str(label)),
                }
            )

        summary_rows.append(
            {
                "dataset_id": args.dataset_id,
                "recording": str(eeg_path),
                "kept_epochs": int(kept),
                "skipped_epochs": int(skipped + max(0, len(labels) - n_epochs)),
                "sfreq": sfreq,
                "n_channels": int(header["nbchan"]),
                "n_signal_epochs": int(n_signal_epochs),
                "n_label_epochs": int(len(labels)),
            }
        )
        print(f"  Kept {kept} epochs")

    if not specs:
        raise RuntimeError("No spectrogram epochs were extracted")

    X = np.stack(specs, axis=0).astype(np.float32)
    y = np.asarray(labels_out, dtype=np.int64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_spectrograms.npy", X)
    np.save(output_dir / "y_labels.npy", y)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "epoch_metadata.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "extraction_summary.csv", index=False)

    audit = {
        "dataset_id": args.dataset_id,
        "bids_root": args.bids_root,
        "n_rows": int(X.shape[0]),
        "spectrogram_shape": list(X.shape[1:]),
        "nperseg": int(args.nperseg),
        "noverlap": int(args.noverlap),
        "fmin": float(args.fmin),
        "fmax": float(args.fmax),
        "class_distribution": {
            SLEEP_STAGE_NAMES.get(int(label), str(int(label))): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
    }
    with open(output_dir / "spectrogram_extraction_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Saved spectrograms to {output_dir.resolve()}")
    print(f"  X_spectrograms.npy: {X.shape}")
    print(f"  y_labels.npy: {y.shape}")


if __name__ == "__main__":
    main()
