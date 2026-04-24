"""Audit labels, extraction outputs, and class balance for ds003768."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

SLEEP_STAGE_MAP = {"W": 0, "1": 1, "2": 2, "3": 3, "R": 4}
VALID_STAGE_STRINGS = list(SLEEP_STAGE_MAP.keys())


def load_stage_tables(sourcedata_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(sourcedata_dir, "sub-*-sleep-stage.tsv"))):
        df = pd.read_csv(path, sep="\t")
        df.columns = df.columns.str.strip()
        if "subject" not in df.columns:
            df["subject"] = int(Path(path).stem.split("-")[1])
        frames.append(df)

    stages = pd.concat(frames, ignore_index=True)
    stages["stage_str"] = stages["30-sec_epoch_sleep_stage"].astype(str).str.strip()
    return stages


def build_clean_labels(stages: pd.DataFrame) -> pd.DataFrame:
    clean = stages[stages["stage_str"].isin(VALID_STAGE_STRINGS)].copy()
    clean["stage_int"] = clean["stage_str"].map(SLEEP_STAGE_MAP)
    return clean


def count_signal_epochs(vhdr_path: str) -> int:
    text = Path(vhdr_path).read_text(encoding="utf-8", errors="ignore")
    n_channels = int(re.search(r"NumberOfChannels=(\d+)", text).group(1))
    binary_format = re.search(r"BinaryFormat=([A-Z0-9_]+)", text).group(1)
    bytes_per_sample = {"INT_16": 2, "INT_32": 4, "IEEE_FLOAT_32": 4}[binary_format]
    eeg_path = vhdr_path.replace(".vhdr", ".eeg")
    n_samples = os.path.getsize(eeg_path) // (n_channels * bytes_per_sample)
    return int(n_samples // (30 * 5000))


def collect_eeg_records(bids_root: str) -> list[dict]:
    records = []
    for vhdr_path in sorted(glob.glob(os.path.join(bids_root, "sub-*/eeg/*.vhdr"))):
        stem = Path(vhdr_path).stem
        parts = stem.split("_")
        subject = int(parts[0].replace("sub-", ""))
        session = "_".join(part for part in parts if part.startswith("task-") or part.startswith("run-"))
        records.append(
            {
                "subject": subject,
                "subject_str": parts[0],
                "session": session,
                "vhdr_path": vhdr_path,
                "key": f"{parts[0]}_{session}",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ds003768 labels and extracted features")
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_DIR, "outputs"))
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    stages = load_stage_tables(os.path.join(args.bids_root, "sourcedata"))
    clean = build_clean_labels(stages)
    eeg_records = collect_eeg_records(args.bids_root)

    clean_counts = clean["stage_str"].value_counts().to_dict()
    excluded_counts = stages.loc[~stages["stage_str"].isin(VALID_STAGE_STRINGS), "stage_str"].value_counts(dropna=False).to_dict()
    clean_keys = set(
        clean.apply(lambda row: f"sub-{int(row['subject']):02d}_{row['session']}", axis=1).unique().tolist()
    )
    eeg_keys = {record["key"] for record in eeg_records}
    eeg_without_clean_labels = sorted(eeg_keys - clean_keys)

    aligned_rows = []
    for record in eeg_records:
        labels = clean[(clean["subject"] == record["subject"]) & (clean["session"] == record["session"])].sort_values(
            "epoch_start_time_sec"
        )
        if len(labels) == 0:
            continue
        signal_epochs = count_signal_epochs(record["vhdr_path"])
        aligned_rows.append(
            {
                "key": record["key"],
                "label_epochs": int(len(labels)),
                "signal_epochs": int(signal_epochs),
                "signal_minus_labels": int(signal_epochs - len(labels)),
            }
        )

    extracted_counts = None
    extracted_total = None
    if os.path.exists(os.path.join(args.output_dir, "X_features.npy")) and os.path.exists(
        os.path.join(args.output_dir, "y_labels.npy")
    ):
        y = np.load(os.path.join(args.output_dir, "y_labels.npy"))
        extracted_total = int(len(y))
        extracted_counts = {
            stage: int((y == stage_idx).sum())
            for stage, stage_idx in SLEEP_STAGE_MAP.items()
            if int((y == stage_idx).sum()) > 0
        }

    report = {
        "total_label_rows": int(len(stages)),
        "clean_label_rows": int(len(clean)),
        "clean_stage_counts": {key: int(value) for key, value in clean_counts.items()},
        "excluded_stage_counts": {str(key): int(value) for key, value in excluded_counts.items()},
        "eeg_recordings": int(len(eeg_records)),
        "recordings_with_clean_labels": int(len(aligned_rows)),
        "eeg_without_clean_labels": eeg_without_clean_labels,
        "sum_label_epochs_for_recordings": int(sum(row["label_epochs"] for row in aligned_rows)),
        "sum_signal_epochs_for_recordings": int(sum(row["signal_epochs"] for row in aligned_rows)),
        "subjects_with_n3": sorted(clean.loc[clean["stage_str"] == "3", "subject"].unique().tolist()),
        "extracted_total_rows": extracted_total,
        "extracted_stage_counts": extracted_counts,
    }

    print(json.dumps(report, indent=2))

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
