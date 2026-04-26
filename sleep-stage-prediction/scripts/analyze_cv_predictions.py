"""Analyze cross-validated sleep-stage predictions by fold and subject."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


def summarize_predictions(frame: pd.DataFrame, labels: list[int]) -> dict:
    y_true = frame["true_label"].to_numpy(dtype=int)
    y_pred = frame["pred_label"].to_numpy(dtype=int)
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    row = {
        "n_epochs": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "n1_precision": float(report.get("N1", {}).get("precision", np.nan)),
        "n1_recall": float(report.get("N1", {}).get("recall", np.nan)),
        "n1_f1": float(report.get("N1", {}).get("f1-score", np.nan)),
    }
    for label in labels:
        stage_name = SLEEP_STAGE_NAMES.get(label, f"Stage-{label}")
        row[f"support_{stage_name}"] = int(np.sum(y_true == label))
        row[f"recall_{stage_name}"] = float(report[stage_name]["recall"])
    return row


def summarize_groups(frame: pd.DataFrame, labels: list[int], group_column: str) -> pd.DataFrame:
    rows = []
    for group_id, group_frame in frame.groupby(group_column, sort=True):
        row = {group_column: group_id}
        if "fold" in group_frame.columns:
            row["fold"] = int(group_frame["fold"].iloc[0])
        row.update(summarize_predictions(group_frame, labels))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"]).reset_index(drop=True)


def summarize_confusions(frame: pd.DataFrame, labels: list[int]) -> pd.DataFrame:
    matrix = confusion_matrix(frame["true_label"], frame["pred_label"], labels=labels)
    rows = []
    for true_pos, true_label in enumerate(labels):
        for pred_pos, pred_label in enumerate(labels):
            count = int(matrix[true_pos, pred_pos])
            if count == 0:
                continue
            rows.append(
                {
                    "true_stage": SLEEP_STAGE_NAMES.get(true_label, f"Stage-{true_label}"),
                    "pred_stage": SLEEP_STAGE_NAMES.get(pred_label, f"Stage-{pred_label}"),
                    "count": count,
                    "is_correct": bool(true_label == pred_label),
                }
            )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def summarize_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    if "prediction_confidence" not in frame.columns:
        return pd.DataFrame()
    rows = []
    working = frame.copy()
    working["is_correct"] = working["true_label"] == working["pred_label"]
    for is_correct, group_frame in working.groupby("is_correct"):
        row = {
            "subset": "correct" if is_correct else "incorrect",
            "n_epochs": int(len(group_frame)),
            "mean_confidence": float(group_frame["prediction_confidence"].mean()),
            "median_confidence": float(group_frame["prediction_confidence"].median()),
        }
        if "prediction_entropy" in group_frame.columns:
            row["mean_entropy"] = float(group_frame["prediction_entropy"].mean())
            row["median_entropy"] = float(group_frame["prediction_entropy"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 8) -> str:
    if frame.empty:
        return "_No rows available._"
    subset = frame.head(max_rows)
    columns = list(subset.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in subset.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    output_path: Path,
    overall: dict,
    group_summary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    confusion_summary: pd.DataFrame,
    confidence_summary: pd.DataFrame,
    group_column: str,
) -> None:
    worst_groups = group_summary[
        [group_column, "fold", "n_epochs", "balanced_accuracy", "macro_f1", "n1_f1"]
        if "fold" in group_summary.columns
        else [group_column, "n_epochs", "balanced_accuracy", "macro_f1", "n1_f1"]
    ]
    lines = [
        "# CV Prediction Analysis",
        "",
        "## Overall",
        "",
        f"- Epochs: `{overall['n_epochs']}`",
        f"- Balanced accuracy: `{overall['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{overall['macro_f1']:.4f}`",
        f"- N1 F1: `{overall.get('n1_f1', float('nan')):.4f}`",
        "",
        "## Fold Summary",
        "",
        markdown_table(fold_summary),
        "",
        "## Weakest Subjects Or Recordings",
        "",
        markdown_table(worst_groups),
        "",
        "## Largest Confusions",
        "",
        markdown_table(confusion_summary[~confusion_summary["is_correct"]]),
        "",
    ]
    if not confidence_summary.empty:
        lines.extend(
            [
                "## Confidence",
                "",
                markdown_table(confidence_summary),
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved CV predictions")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_path)
    required = {"true_label", "pred_label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")

    labels = sorted(int(label) for label in set(predictions["true_label"]).union(predictions["pred_label"]))
    group_column = args.group_column or "analysis_group"
    if args.group_column is None:
        predictions[group_column] = derive_group_ids(predictions)
    elif args.group_column not in predictions.columns:
        raise KeyError(f"Group column {args.group_column!r} not found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = summarize_predictions(predictions, labels)
    group_summary = summarize_groups(predictions, labels, group_column)
    confusion_summary = summarize_confusions(predictions, labels)
    confidence_summary = summarize_confidence(predictions)

    if "fold" in predictions.columns:
        fold_summary = summarize_groups(predictions, labels, "fold")
    else:
        fold_summary = pd.DataFrame()

    group_summary.to_csv(output_dir / "subject_metrics.csv", index=False)
    confusion_summary.to_csv(output_dir / "confusion_summary.csv", index=False)
    if not fold_summary.empty:
        fold_summary.to_csv(output_dir / "fold_metrics_from_predictions.csv", index=False)
    if not confidence_summary.empty:
        confidence_summary.to_csv(output_dir / "confidence_summary.csv", index=False)
    with open(output_dir / "overall_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    write_summary(
        output_dir / "prediction_analysis.md",
        overall,
        group_summary,
        fold_summary,
        confusion_summary,
        confidence_summary,
        group_column,
    )
    print(f"Saved CV prediction analysis to {output_dir}")
    print(
        f"Overall: balanced_accuracy={overall['balanced_accuracy']:.4f}, "
        f"macro_f1={overall['macro_f1']:.4f}, n1_f1={overall.get('n1_f1', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
