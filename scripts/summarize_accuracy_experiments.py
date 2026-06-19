"""Summarize marathon sleep-stage accuracy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "macro_f1",
    "n1_precision",
    "n1_recall",
    "n1_f1",
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_training_row(experiment_dir: Path) -> dict | None:
    metrics = load_json(experiment_dir / "metrics.json")
    if metrics is None:
        return None
    summary = metrics.get("summary", {})
    row = {
        "experiment": experiment_dir.name,
        "variant": "raw",
        "output_dir": str(experiment_dir),
        "model": metrics.get("model"),
        "sequence_length": metrics.get("sequence_length"),
        "sequence_radius": metrics.get("sequence_radius"),
        "block_length": metrics.get("block_length"),
        "block_stride": metrics.get("block_stride"),
        "normalization": metrics.get("normalization"),
        "epochs": metrics.get("epochs"),
        "batch_size": metrics.get("batch_size"),
        "hidden_size": metrics.get("hidden_size"),
        "dropout": metrics.get("dropout"),
        "learning_rate": metrics.get("learning_rate"),
        "weight_decay": metrics.get("weight_decay"),
        "loss": metrics.get("loss"),
        "label_smoothing": metrics.get("label_smoothing"),
        "balanced_sampling": metrics.get("balanced_sampling"),
        "augment": metrics.get("augment"),
    }
    row.update({key: summary.get(key) for key in METRIC_KEYS})
    return row


def collect_viterbi_rows(experiment_dir: Path) -> list[dict]:
    viterbi_dir = experiment_dir.parent / f"{experiment_dir.name}_viterbi"
    summary_path = viterbi_dir / "viterbi_smoothing_summary.csv"
    if not summary_path.exists():
        return []
    frame = pd.read_csv(summary_path)
    rows = []
    for _, values in frame.iterrows():
        row = {
            "experiment": experiment_dir.name,
            "variant": f"viterbi_w{values['transition_weight']}",
            "output_dir": str(viterbi_dir),
            "model": "viterbi_postprocess",
            "transition_weight": values["transition_weight"],
        }
        row.update({key: values.get(key) for key in METRIC_KEYS})
        rows.append(row)
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        (output_dir / "accuracy_experiment_summary.md").write_text(
            "# Accuracy Experiment Summary\n\nNo completed experiments found yet.\n",
            encoding="utf-8",
        )
        return

    sort_columns = ["balanced_accuracy", "macro_f1", "accuracy"]
    frame = frame.sort_values(sort_columns, ascending=False)
    frame.to_csv(output_dir / "accuracy_experiment_results.csv", index=False)

    top_columns = [
        "experiment",
        "variant",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "n1_f1",
        "model",
        "output_dir",
    ]
    raw_columns = [
        "experiment",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "n1_f1",
        "model",
        "epochs",
        "sequence_length",
        "block_length",
    ]
    best = frame.iloc[0]
    raw_frame = frame[frame["variant"] == "raw"].copy()
    raw_frame = raw_frame.sort_values(sort_columns, ascending=False)

    lines = [
        "# Accuracy Experiment Summary",
        "",
        "## Best Result",
        "",
        f"- Experiment: `{best['experiment']}`",
        f"- Variant: `{best['variant']}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- Accuracy: `{best['accuracy']:.4f}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- N1 F1: `{best['n1_f1']:.4f}`",
        "",
        "## Top Results",
        "",
        markdown_table(frame.head(15), top_columns),
        "",
        "## Raw Model Ranking",
        "",
        markdown_table(raw_frame.head(15), raw_columns),
        "",
    ]
    (output_dir / "accuracy_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed accuracy experiments")
    parser.add_argument("--root-dir", default="outputs/accuracy_marathon")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else root_dir
    rows = []
    for experiment_dir in sorted(root_dir.glob("exp_*")):
        if not experiment_dir.is_dir() or experiment_dir.name.endswith("_viterbi"):
            continue
        row = collect_training_row(experiment_dir)
        if row is None:
            continue
        rows.append(row)
        rows.extend(collect_viterbi_rows(experiment_dir))

    write_report(rows, output_dir)
    print(f"Saved experiment summary to {output_dir}")


if __name__ == "__main__":
    main()
