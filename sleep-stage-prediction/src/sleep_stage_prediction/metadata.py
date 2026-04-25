"""Metadata helpers for dataset-aware sleep-stage experiments."""

from __future__ import annotations

import re

import pandas as pd


def derive_group_ids(metadata: pd.DataFrame, group_column: str | None = None) -> pd.Series:
    """Return group IDs suitable for subject/recording-aware validation."""
    if group_column:
        if group_column not in metadata.columns:
            raise KeyError(f"Group column {group_column!r} not found in metadata")
        values = metadata[group_column].astype(str)
        if "dataset_id" in metadata.columns:
            return metadata["dataset_id"].astype(str) + ":" + values
        return values

    if "subject" in metadata.columns and metadata["subject"].notna().any():
        values = metadata["subject"].fillna("unknown").astype(str)
        return metadata["dataset_id"].fillna("unknown").astype(str) + ":" + values

    if "recording" in metadata.columns and metadata["recording"].notna().any():
        recordings = metadata["recording"].fillna("unknown").astype(str)
        subjects = recordings.str.extract(r"(sub-[A-Za-z0-9]+)", expand=False)
        values = subjects.fillna(recordings).fillna("unknown").astype(str)
        if "dataset_id" in metadata.columns:
            return metadata["dataset_id"].fillna("unknown").astype(str) + ":" + values
        return values

    if "dataset_id" in metadata.columns:
        return metadata["dataset_id"].fillna("unknown").astype(str) + ":unknown"

    return pd.Series([f"row-{idx}" for idx in range(len(metadata))], index=metadata.index)
