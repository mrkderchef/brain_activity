"""
data_loader.py – Laden von EEG-Daten und Schlafstadien-Labels aus dem BIDS-Dataset ds003768.

Hinweis: Die EEG-Binärdaten (.eeg) liegen als git-annex Pointer vor und müssen
zuerst von OpenNeuro heruntergeladen werden:
    datalad install https://github.com/OpenNeuroDatasets/ds003768.git
    datalad get ds003768/sub-*/eeg/*
"""

import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

SLEEP_STAGE_MAP = {"W": 0, "1": 1, "2": 2, "3": 3, "R": 4}
SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
EPOCH_DURATION = 30.0  # Sekunden


# ---------------------------------------------------------------------------
# Labels laden
# ---------------------------------------------------------------------------

def load_sleep_stages(sourcedata_dir: str) -> pd.DataFrame:
    """Lädt alle Schlafstadien-TSV-Dateien und gibt einen kombinierten DataFrame zurück."""
    tsv_files = sorted(glob.glob(os.path.join(sourcedata_dir, "sub-*-sleep-stage.tsv")))
    if not tsv_files:
        raise FileNotFoundError(f"Keine sleep-stage TSV-Dateien in {sourcedata_dir}")

    frames = []
    for f in tsv_files:
        df = pd.read_csv(f, sep="\t")
        df.columns = df.columns.str.strip()
        # Schlafstadium als String behandeln, dann mappen
        df["stage_str"] = df["30-sec_epoch_sleep_stage"].astype(str).str.strip()
        df["stage_int"] = df["stage_str"].map(SLEEP_STAGE_MAP)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["stage_int"])
    combined["stage_int"] = combined["stage_int"].astype(int)
    return combined


def get_labels_for_session(stages_df: pd.DataFrame, subject_id: int, session: str) -> np.ndarray:
    """Gibt die Schlafstadien-Labels für eine bestimmte Subject/Session-Kombination zurück."""
    mask = (stages_df["subject"] == subject_id) & (stages_df["session"] == session)
    subset = stages_df[mask].sort_values("epoch_start_time_sec")
    return subset["stage_int"].values


# ---------------------------------------------------------------------------
# EEG laden
# ---------------------------------------------------------------------------

def load_eeg_raw(vhdr_path: str) -> mne.io.Raw:
    """Lädt eine EEG-Aufnahme aus der BrainVision .vhdr-Datei."""
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    return raw


def find_eeg_files(bids_root: str) -> list[dict]:
    """Findet alle EEG .vhdr-Dateien im BIDS-Verzeichnis und extrahiert Metadaten."""
    vhdr_files = sorted(glob.glob(os.path.join(bids_root, "sub-*/eeg/*.vhdr")))
    records = []
    for vhdr in vhdr_files:
        fname = Path(vhdr).stem  # z.B. sub-01_task-sleep_run-1_eeg
        parts = fname.split("_")
        sub = parts[0]  # sub-01
        task_session = "_".join(p for p in parts if p.startswith("task-") or p.startswith("run-"))
        sub_id = int(sub.replace("sub-", ""))
        records.append({
            "subject": sub_id,
            "subject_str": sub,
            "session": task_session,
            "vhdr_path": vhdr,
        })
    return records


def is_annex_pointer(filepath: str) -> bool:
    """Prüft ob eine Datei ein git-annex Pointer ist (kein echtes Binary)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            return first_line.startswith("../../.git/annex") or first_line.startswith("/annex/")
    except (UnicodeDecodeError, PermissionError):
        return False  # Binärdatei → echte Daten


def check_data_availability(bids_root: str) -> dict:
    """Prüft ob echte EEG-Daten oder nur Annex-Pointer vorhanden sind."""
    eeg_files = find_eeg_files(bids_root)
    available = []
    missing = []
    for rec in eeg_files:
        eeg_bin = rec["vhdr_path"].replace(".vhdr", ".eeg")
        if os.path.exists(eeg_bin) and not is_annex_pointer(eeg_bin):
            available.append(rec)
        else:
            missing.append(rec)
    return {"available": available, "missing": missing}
