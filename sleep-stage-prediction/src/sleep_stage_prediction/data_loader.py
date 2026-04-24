"""Load EEG recordings and sleep stage labels from ds003768."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd


SLEEP_STAGE_MAP = {"W": 0, "1": 1, "2": 2, "3": 3, "R": 4}
SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
EPOCH_DURATION = 30.0


def load_sleep_stages(sourcedata_dir: str) -> pd.DataFrame:
    """Load and combine all sleep stage TSV files."""
    tsv_files = sorted(glob.glob(os.path.join(sourcedata_dir, "sub-*-sleep-stage.tsv")))
    if not tsv_files:
        raise FileNotFoundError(f"No sleep-stage TSV files found in {sourcedata_dir}")

    frames = []
    for path in tsv_files:
        df = pd.read_csv(path, sep="\t")
        df.columns = df.columns.str.strip()
        if "subject" not in df.columns:
            subject_str = Path(path).stem.split("-")[1]
            df["subject"] = int(subject_str)
        df["stage_str"] = df["30-sec_epoch_sleep_stage"].astype(str).str.strip()
        df["stage_int"] = df["stage_str"].map(SLEEP_STAGE_MAP)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["stage_int"])
    combined["stage_int"] = combined["stage_int"].astype(int)
    return combined


def get_labels_for_session(stages_df: pd.DataFrame, subject_id: int, session: str) -> np.ndarray:
    """Return sleep stage labels for one subject/session recording."""
    mask = (stages_df["subject"] == subject_id) & (stages_df["session"] == session)
    subset = stages_df[mask].sort_values("epoch_start_time_sec")
    return subset["stage_int"].to_numpy()


def load_eeg_raw(vhdr_path: str) -> mne.io.Raw:
    """Load one BrainVision recording."""
    return mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)


def find_eeg_files(bids_root: str) -> list[dict]:
    """Find all EEG header files and derive session metadata."""
    vhdr_files = sorted(glob.glob(os.path.join(bids_root, "sub-*/eeg/*.vhdr")))
    records = []
    for vhdr in vhdr_files:
        fname = Path(vhdr).stem
        parts = fname.split("_")
        subject_str = parts[0]
        session = "_".join(
            part for part in parts if part.startswith("task-") or part.startswith("run-")
        )
        records.append(
            {
                "subject": int(subject_str.replace("sub-", "")),
                "subject_str": subject_str,
                "session": session,
                "vhdr_path": vhdr,
            }
        )
    return records


def is_annex_pointer(filepath: str) -> bool:
    """Return True when a file looks like a git-annex pointer instead of real EEG data."""
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            first_line = handle.readline()
    except (UnicodeDecodeError, PermissionError):
        return False

    return first_line.startswith("../../.git/annex") or first_line.startswith("/annex/")


def check_data_availability(bids_root: str) -> dict:
    """Split EEG files into available binaries vs. annex pointers."""
    available = []
    missing = []
    for record in find_eeg_files(bids_root):
        eeg_bin = record["vhdr_path"].replace(".vhdr", ".eeg")
        if os.path.exists(eeg_bin) and not is_annex_pointer(eeg_bin):
            available.append(record)
        else:
            missing.append(record)
    return {"available": available, "missing": missing}
