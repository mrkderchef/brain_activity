"""Analyze Sleep-EDF spectrogram CV results by held-out record and stage mix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def summarize_group(group: pd.DataFrame) -> dict:
    true_column = "y_true" if "y_true" in group.columns else "true_label"
    pred_column = "y_pred" if "y_pred" in group.columns else "pred_label"
    y_true = group[true_column].astype(int).to_numpy()
    y_pred = group[pred_column].astype(int).to_numpy()
    labels = sorted(STAGE_NAMES)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[STAGE_NAMES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    row = {
        "analysis_group": str(group["analysis_group"].iloc[0]),
        "subject": str(group["subject"].iloc[0]) if "subject" in group.columns else str(group["analysis_group"].iloc[0]),
        "fold": int(group["fold"].iloc[0]) if "fold" in group.columns else -1,
        "n_epochs": int(len(group)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "n1_f1": float(report["N1"]["f1-score"]),
    }
    for label, stage_name in STAGE_NAMES.items():
        support = int((y_true == label).sum())
        row[f"support_{stage_name}"] = support
        row[f"share_{stage_name}"] = float(support / len(group)) if len(group) else 0.0
        row[f"recall_{stage_name}"] = float(report[stage_name]["recall"])
    return row


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    subset = frame[columns] if max_rows is None else frame[columns].head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in subset.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_path: Path, by_record: pd.DataFrame, overall: dict, confusion: np.ndarray) -> None:
    ranking = by_record.sort_values("balanced_accuracy")
    lines = [
        "# Sleep-EDF Result Analysis",
        "",
        "## Overall",
        "",
        f"- Accuracy: `{overall['accuracy']:.4f}`",
        f"- Balanced accuracy: `{overall['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{overall['macro_f1']:.4f}`",
        f"- N1 F1: `{overall['n1_f1']:.4f}`",
        "",
        "## Held-Out Record Ranking",
        "",
        markdown_table(
            ranking,
            [
                "subject",
                "fold",
                "n_epochs",
                "balanced_accuracy",
                "macro_f1",
                "n1_f1",
                "share_Wake",
                "share_N1",
                "share_N2",
                "share_N3",
                "share_REM",
            ],
        ),
        "",
        "## Confusion Matrix",
        "",
        "| true/pred | Wake | N1 | N2 | N3 | REM |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, stage_name in STAGE_NAMES.items():
        lines.append("| " + " | ".join([stage_name] + [str(int(value)) for value in confusion[label]]) + " |")
    lines.extend(
        [
            "",
            "Interpretation: low-ranked records are candidate domain/night-shift cases. Check whether their stage mix is unusual before treating the model as uniformly strong.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Sleep-EDF CV predictions by record")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_path)
    if "analysis_group" not in predictions.columns:
        predictions["analysis_group"] = predictions.get("subject", predictions.get("recording", "unknown"))
    if "subject" not in predictions.columns:
        predictions["subject"] = predictions["analysis_group"]

    by_record = pd.DataFrame(
        summarize_group(group)
        for _, group in predictions.groupby("analysis_group", sort=True)
    ).sort_values(["fold", "subject"])
    overall = summarize_group(predictions.assign(analysis_group="overall", subject="overall", fold=-1))
    labels = sorted(STAGE_NAMES)
    confusion = confusion_matrix(
        predictions["true_label" if "true_label" in predictions.columns else "y_true"].astype(int),
        predictions["pred_label" if "pred_label" in predictions.columns else "y_pred"].astype(int),
        labels=labels,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_record.to_csv(output_dir / "record_metrics.csv", index=False)
    with open(output_dir / "overall_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    pd.DataFrame(confusion, index=[STAGE_NAMES[label] for label in labels], columns=[STAGE_NAMES[label] for label in labels]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    write_report(output_dir / "sleep_edf_result_analysis.md", by_record, overall, confusion)
    print(f"Saved Sleep-EDF result analysis to {output_dir}")


if __name__ == "__main__":
    main()
