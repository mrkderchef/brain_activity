"""Utilities for external BIDS-like sleep EEG datasets."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sleep_stage_prediction.feature_extraction import extract_epoch_features


UNKNOWN_STAGE = -1
SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

STAGE_NAME_TO_INT = {
    "w": 0,
    "wake": 0,
    "sleep stage w": 0,
    "n1": 1,
    "1": 1,
    "sleep stage 1": 1,
    "sleep stage n1": 1,
    "n2": 2,
    "2": 2,
    "sleep stage 2": 2,
    "sleep stage n2": 2,
    "n3": 3,
    "3": 3,
    "4": 3,
    "sleep stage 3": 3,
    "sleep stage 4": 3,
    "sleep stage n3": 3,
    "r": 4,
    "rem": 4,
    "sleep stage r": 4,
    "sleep stage rem": 4,
}

DS006695_NUMERIC_STAGE_MAP = {
    1: 0,  # Wake
    2: 4,  # REM
    3: 1,  # N1
    4: 2,  # N2
    5: 3,  # N3
}


def find_eeg_recordings(bids_root: str) -> list[Path]:
    """Find external EEG files that MNE can read."""
    patterns = ["sub-*/eeg/*_eeg.edf", "sub-*/eeg/*_eeg.set", "sub-*/eeg/*_eeg.vhdr"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(os.path.join(bids_root, pattern)))
    return sorted(paths)


def find_events_file(eeg_path: Path) -> Path | None:
    """Find the matching BIDS events TSV for one EEG file."""
    stem = eeg_path.stem
    candidates = [
        eeg_path.with_name(stem.replace("_eeg", "_events") + ".tsv"),
        eeg_path.with_name(re.sub(r"_eeg$", "", stem) + "_events.tsv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    prefix = re.sub(r"_eeg$", "", stem)
    matches = sorted(eeg_path.parent.glob(f"{prefix}*_events.tsv"))
    return matches[0] if matches else None


def load_external_raw(eeg_path: Path) -> mne.io.BaseRaw:
    """Load EDF, EEGLAB SET, or BrainVision EEG recordings."""
    import mne

    suffix = eeg_path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    if suffix == ".set":
        return mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
    raise ValueError(f"Unsupported EEG file format: {eeg_path}")


def choose_stage_column(events: pd.DataFrame, preferred_column: str | None = None) -> str:
    """Pick the most likely sleep-stage column from an events table."""
    if preferred_column:
        if preferred_column not in events.columns:
            raise KeyError(f"Requested label column {preferred_column!r} not found")
        return preferred_column

    candidates = [
        "sleep_stage",
        "stage",
        "stage_label",
        "trial_type",
        "event_type",
        "value",
        "description",
    ]
    for column in candidates:
        if column in events.columns:
            return column
    raise KeyError(f"No recognizable sleep-stage label column found: {list(events.columns)}")


def normalize_stage_label(value: object, preset: str = "auto") -> int:
    """Normalize common sleep-stage labels into Wake/N1/N2/N3/REM integers."""
    if pd.isna(value):
        return UNKNOWN_STAGE

    if preset == "ds006695":
        try:
            return DS006695_NUMERIC_STAGE_MAP.get(int(float(value)), UNKNOWN_STAGE)
        except (TypeError, ValueError):
            pass

    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.removeprefix("stage ")
    text = text.removeprefix("sleep stage ")

    if text in STAGE_NAME_TO_INT:
        return STAGE_NAME_TO_INT[text]
    if "rem" in text:
        return 4
    if "wake" in text:
        return 0
    if re.fullmatch(r"n?[1234]", text):
        return STAGE_NAME_TO_INT[text.replace("n", "")]
    return UNKNOWN_STAGE


def load_events(
    events_path: Path,
    label_column: str | None = None,
    preset: str = "auto",
) -> pd.DataFrame:
    """Load and normalize one BIDS events TSV file."""
    events = pd.read_csv(events_path, sep="\t")
    events.columns = events.columns.str.strip()
    if "onset" not in events.columns:
        raise KeyError(f"{events_path} has no onset column")

    stage_column = choose_stage_column(events, label_column)
    events["stage_int"] = events[stage_column].apply(lambda value: normalize_stage_label(value, preset))
    events = events[events["stage_int"] != UNKNOWN_STAGE].copy()
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events.dropna(subset=["onset", "stage_int"])
    events["stage_int"] = events["stage_int"].astype(int)
    return events.sort_values("onset")


def load_ds006695_hypnogram(set_path: Path) -> np.ndarray:
    """Load ds006695 manual 30-second labels from an EEGLAB SET file."""
    from scipy.io import loadmat

    mat = loadmat(
        set_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["VisualHypnogram"],
    )
    if "VisualHypnogram" not in mat:
        raise KeyError(f"{set_path} does not contain VisualHypnogram")

    raw_labels = np.ravel(mat["VisualHypnogram"])
    labels = np.array(
        [normalize_stage_label(value, preset="ds006695") for value in raw_labels],
        dtype=int,
    )
    return labels[labels != UNKNOWN_STAGE]


def load_ds006695_set_header(set_path: Path) -> dict:
    """Load the small EEGLAB header fields needed for direct FDT access."""
    from scipy.io import loadmat

    mat = loadmat(
        set_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["data", "nbchan", "pnts", "srate"],
    )
    data_ref = mat["data"]
    if not isinstance(data_ref, str):
        raise ValueError(f"{set_path} stores data inline; direct FDT extraction expected a filename")
    return {
        "data_file": str(data_ref),
        "nbchan": int(mat["nbchan"]),
        "pnts": int(mat["pnts"]),
        "srate": float(mat["srate"]),
    }


def extract_external_recording(
    eeg_path: Path,
    events_path: Path,
    dataset_id: str,
    epoch_duration: float = 30.0,
    label_column: str | None = None,
    preset: str = "auto",
    max_epochs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Extract features and labels from one external recording."""
    events = load_events(events_path, label_column=label_column, preset=preset)
    raw = load_external_raw(eeg_path)
    sfreq = float(raw.info["sfreq"])
    n_samples = int(round(epoch_duration * sfreq))

    features = []
    labels = []
    metadata_rows = []
    skipped = 0

    if max_epochs is not None:
        events = events.head(max_epochs)

    for row_idx, row in events.reset_index(drop=True).iterrows():
        start = int(round(float(row["onset"]) * sfreq))
        stop = start + n_samples
        if start < 0 or stop > raw.n_times:
            skipped += 1
            continue

        epoch_data = raw.get_data(start=start, stop=stop)
        feature_row = extract_epoch_features(epoch_data, sfreq)
        if not np.all(np.isfinite(feature_row)):
            skipped += 1
            continue

        label = int(row["stage_int"])
        features.append(feature_row)
        labels.append(label)
        metadata_rows.append(
            {
                "dataset_id": dataset_id,
                "recording": str(eeg_path.relative_to(eeg_path.parents[2])),
                "events_file": str(events_path.relative_to(events_path.parents[2])),
                "epoch_index": int(row_idx),
                "epoch_start_time_sec": float(row["onset"]),
                "label": label,
                "label_name": SLEEP_STAGE_NAMES.get(label, str(label)),
            }
        )

    summary = {
        "dataset_id": dataset_id,
        "recording": str(eeg_path),
        "events_file": str(events_path),
        "event_rows_with_known_labels": int(len(events)),
        "kept_epochs": int(len(labels)),
        "skipped_epochs": int(skipped),
        "sfreq": sfreq,
    }
    if not features:
        return np.empty((0, 0)), np.empty((0,), dtype=int), metadata_rows, summary
    return np.vstack(features), np.array(labels, dtype=int), metadata_rows, summary


def extract_ds006695_set_recording(
    eeg_path: Path,
    dataset_id: str,
    epoch_duration: float = 30.0,
    max_epochs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Extract ds006695 features when labels are stored as VisualHypnogram."""
    labels = load_ds006695_hypnogram(eeg_path)
    header = load_ds006695_set_header(eeg_path)
    sfreq = header["srate"]
    n_samples = int(round(epoch_duration * sfreq))
    n_signal_epochs = header["pnts"] // n_samples
    n_epochs = min(len(labels), n_signal_epochs)
    if max_epochs is not None:
        n_epochs = min(n_epochs, max_epochs)
    fdt_path = eeg_path.with_name(header["data_file"])
    if not fdt_path.exists():
        raise FileNotFoundError(f"Missing EEGLAB FDT data file: {fdt_path}")

    signal = np.memmap(
        fdt_path,
        dtype="<f4",
        mode="r",
        shape=(header["nbchan"], header["pnts"]),
        order="F",
    )

    features = []
    kept_labels = []
    metadata_rows = []
    skipped = 0

    for epoch_idx in range(n_epochs):
        start = epoch_idx * n_samples
        stop = start + n_samples
        epoch_data = np.asarray(signal[:, start:stop], dtype=np.float64)
        feature_row = extract_epoch_features(epoch_data, sfreq)
        if not np.all(np.isfinite(feature_row)):
            skipped += 1
            continue

        label = int(labels[epoch_idx])
        features.append(feature_row)
        kept_labels.append(label)
        metadata_rows.append(
            {
                "dataset_id": dataset_id,
                "recording": str(eeg_path.relative_to(eeg_path.parents[2])),
                "events_file": "EEG.VisualHypnogram",
                "epoch_index": int(epoch_idx),
                "epoch_start_time_sec": float(epoch_idx * epoch_duration),
                "label": label,
                "label_name": SLEEP_STAGE_NAMES.get(label, str(label)),
            }
        )

    summary = {
        "dataset_id": dataset_id,
        "recording": str(eeg_path),
        "events_file": "EEG.VisualHypnogram",
        "event_rows_with_known_labels": int(len(labels)),
        "kept_epochs": int(len(kept_labels)),
        "skipped_epochs": int(skipped + max(0, len(labels) - n_epochs)),
        "sfreq": sfreq,
        "reader": "direct_eeglab_fdt",
    }
    if not features:
        return np.empty((0, 0)), np.empty((0,), dtype=int), metadata_rows, summary
    return np.vstack(features), np.array(kept_labels, dtype=int), metadata_rows, summary


def write_external_extraction_outputs(
    output_dir: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata_rows: list[dict],
    summary_rows: list[dict],
    dataset_id: str,
    preset: str,
) -> None:
    """Write external extraction arrays and audit files."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_features.npy"), X)
    np.save(os.path.join(output_dir, "y_labels.npy"), y)
    pd.DataFrame(metadata_rows).to_csv(os.path.join(output_dir, "epoch_metadata.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "extraction_summary.csv"), index=False)
    with open(os.path.join(output_dir, "extraction_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_id": dataset_id,
                "preset": preset,
                "total_kept_rows": int(len(y)),
                "class_distribution": {
                    SLEEP_STAGE_NAMES.get(int(k), str(int(k))): int(v)
                    for k, v in zip(*np.unique(y, return_counts=True))
                }
                if len(y)
                else {},
            },
            handle,
            indent=2,
        )
