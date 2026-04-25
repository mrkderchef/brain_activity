"""Analyze N1 errors under subject-aware cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES, create_model


N1_LABEL = 1


def sort_group_indices(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    sort_frame = metadata.iloc[indices].copy()
    sort_frame["_row_index"] = indices
    sort_columns = []
    for column in ["epoch_start_time_sec", "epoch_index"]:
        if column in sort_frame.columns:
            sort_frame[column] = pd.to_numeric(sort_frame[column], errors="coerce")
            sort_columns.append(column)
    if sort_columns:
        sort_frame = sort_frame.sort_values(sort_columns + ["_row_index"], kind="mergesort")
    return sort_frame["_row_index"].to_numpy(dtype=int)


def stage_name(label: int | float | str) -> str:
    if pd.isna(label):
        return "boundary"
    return SLEEP_STAGE_NAMES.get(int(label), f"Stage-{int(label)}")


def add_true_neighbor_context(metadata: pd.DataFrame, y: np.ndarray, group_column: str | None) -> pd.DataFrame:
    context = pd.DataFrame(index=metadata.index)
    context["true_label"] = y
    context["prev_true_label"] = np.nan
    context["next_true_label"] = np.nan

    group_ids = derive_group_ids(metadata, group_column).astype(str)
    for group_id in sorted(group_ids.unique()):
        indices = np.where(group_ids.to_numpy() == group_id)[0]
        ordered = sort_group_indices(metadata, indices)
        labels = y[ordered]
        prev_labels = np.empty(len(ordered), dtype=float)
        next_labels = np.empty(len(ordered), dtype=float)
        prev_labels[:] = np.nan
        next_labels[:] = np.nan
        prev_labels[1:] = labels[:-1]
        next_labels[:-1] = labels[1:]
        context.loc[ordered, "prev_true_label"] = prev_labels
        context.loc[ordered, "next_true_label"] = next_labels

    context["prev_true_stage"] = context["prev_true_label"].map(stage_name)
    context["next_true_stage"] = context["next_true_label"].map(stage_name)
    context["true_transition"] = context["prev_true_stage"] + " -> " + context["next_true_stage"]
    return context


def counts_table(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    if frame.empty:
        return []
    return (
        frame.groupby(columns, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze N1 cross-validation errors")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--sequence-group-column", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    groups = derive_group_ids(metadata, args.group_column).astype(str)
    if groups.nunique() < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} groups, found {groups.nunique()}")

    sequence_group_column = args.sequence_group_column
    if sequence_group_column is None:
        sequence_group_column = "sequence_group" if "sequence_group" in metadata.columns else args.group_column

    model = create_model(parallel_jobs=args.parallel_jobs)
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups.to_numpy(), n_jobs=args.parallel_jobs)

    labels = sorted(int(label) for label in np.unique(y))
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    predictions = metadata.copy()
    predictions["cv_group"] = groups
    predictions["derived_subject"] = groups.str.replace(r"^.*?:", "", regex=True)
    predictions["true_label"] = y
    predictions["pred_label"] = y_pred
    predictions["true_stage"] = predictions["true_label"].map(SLEEP_STAGE_NAMES)
    predictions["pred_stage"] = predictions["pred_label"].map(SLEEP_STAGE_NAMES)
    predictions = predictions.join(add_true_neighbor_context(metadata, y, sequence_group_column), rsuffix="_context")
    predictions["is_correct"] = predictions["true_label"] == predictions["pred_label"]
    predictions["is_n1"] = predictions["true_label"] == N1_LABEL
    predictions["is_n1_false_negative"] = predictions["is_n1"] & (predictions["pred_label"] != N1_LABEL)
    predictions["is_n1_false_positive"] = (predictions["true_label"] != N1_LABEL) & (predictions["pred_label"] == N1_LABEL)

    n1_rows = predictions[predictions["is_n1"]]
    n1_false_negatives = predictions[predictions["is_n1_false_negative"]]
    n1_false_positives = predictions[predictions["is_n1_false_positive"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "cv_predictions_with_n1_context.csv", index=False)

    summary = {
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "n_groups": int(groups.nunique()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "n1_support": int(len(n1_rows)),
        "n1_correct": int((n1_rows["pred_label"] == N1_LABEL).sum()),
        "n1_false_negatives": int(len(n1_false_negatives)),
        "n1_false_positives": int(len(n1_false_positives)),
        "n1_false_negative_predicted_as": counts_table(n1_false_negatives, ["pred_stage"]),
        "n1_false_negative_by_subject": counts_table(n1_false_negatives, ["derived_subject", "pred_stage"]),
        "n1_false_negative_by_neighbor_context": counts_table(
            n1_false_negatives,
            ["prev_true_stage", "next_true_stage", "pred_stage"],
        ),
        "n1_false_positive_true_stage": counts_table(n1_false_positives, ["true_stage"]),
        "n1_true_neighbor_context": counts_table(n1_rows, ["prev_true_stage", "next_true_stage"]),
    }
    with open(output_dir / "n1_error_analysis.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved N1 error analysis to {output_dir}")
    print(f"N1 recall: {report['N1']['recall']:.4f}")
    print(f"N1 false negatives: {len(n1_false_negatives)}")
    print("N1 false negatives predicted as:")
    print(pd.DataFrame(summary["n1_false_negative_predicted_as"]).to_string(index=False))


if __name__ == "__main__":
    main()
