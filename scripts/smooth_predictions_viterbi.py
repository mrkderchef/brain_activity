"""Apply fold-aware Viterbi transition smoothing to CV prediction probabilities."""

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

from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


def sort_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = []
    for column in ["epoch_start_time_sec", "epoch_index"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            sort_columns.append(column)
    if sort_columns:
        return frame.sort_values(sort_columns, kind="mergesort")
    return frame.sort_index()


def probability_columns(labels: list[int]) -> list[str]:
    return [f"prob_{SLEEP_STAGE_NAMES.get(label, f'Stage-{label}')}" for label in labels]


def estimate_transition_model(
    frame: pd.DataFrame,
    labels: list[int],
    label_to_pos: dict[int, int],
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = len(labels)
    transition_counts = np.full((n_classes, n_classes), smoothing, dtype=np.float64)
    prior_counts = np.full(n_classes, smoothing, dtype=np.float64)

    for _, sequence_frame in frame.groupby("recording", sort=False):
        ordered = sort_sequence(sequence_frame.copy())
        y = ordered["true_label"].to_numpy(dtype=int)
        if len(y) == 0:
            continue
        prior_counts[label_to_pos[int(y[0])]] += 1.0
        for prev_label, next_label in zip(y[:-1], y[1:]):
            transition_counts[label_to_pos[int(prev_label)], label_to_pos[int(next_label)]] += 1.0

    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    prior = prior_counts / prior_counts.sum()
    return prior, transition


def viterbi_decode(
    emissions: np.ndarray,
    prior: np.ndarray,
    transition: np.ndarray,
    transition_weight: float,
) -> np.ndarray:
    emissions = np.clip(emissions, 1e-12, 1.0)
    log_emissions = np.log(emissions)
    log_prior = np.log(np.clip(prior, 1e-12, 1.0))
    log_transition = transition_weight * np.log(np.clip(transition, 1e-12, 1.0))

    n_steps, n_classes = emissions.shape
    scores = np.empty((n_steps, n_classes), dtype=np.float64)
    backpointers = np.zeros((n_steps, n_classes), dtype=np.int64)
    scores[0] = log_prior + log_emissions[0]
    for step in range(1, n_steps):
        candidate_scores = scores[step - 1][:, None] + log_transition
        backpointers[step] = np.argmax(candidate_scores, axis=0)
        scores[step] = np.max(candidate_scores, axis=0) + log_emissions[step]

    path = np.empty(n_steps, dtype=np.int64)
    path[-1] = int(np.argmax(scores[-1]))
    for step in range(n_steps - 2, -1, -1):
        path[step] = backpointers[step + 1, path[step + 1]]
    return path


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "n1_precision": float(report.get("N1", {}).get("precision", np.nan)),
        "n1_recall": float(report.get("N1", {}).get("recall", np.nan)),
        "n1_f1": float(report.get("N1", {}).get("f1-score", np.nan)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def apply_smoothing(
    predictions: pd.DataFrame,
    labels: list[int],
    transition_weight: float,
    transition_smoothing: float,
) -> tuple[pd.DataFrame, dict]:
    label_to_pos = {label: pos for pos, label in enumerate(labels)}
    pos_to_label = {pos: label for label, pos in label_to_pos.items()}
    prob_columns = probability_columns(labels)
    missing = [column for column in prob_columns if column not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions file is missing probability columns: {missing}")
    if "fold" not in predictions.columns:
        raise ValueError("Predictions file needs a fold column for fold-aware transition estimation")

    smoothed_parts = []
    fold_rows = []
    for fold, fold_frame in predictions.groupby("fold", sort=True):
        train_frame = predictions[predictions["fold"] != fold]
        prior, transition = estimate_transition_model(
            train_frame,
            labels=labels,
            label_to_pos=label_to_pos,
            smoothing=transition_smoothing,
        )
        fold_parts = []
        for _, sequence_frame in fold_frame.groupby("recording", sort=False):
            ordered = sort_sequence(sequence_frame.copy())
            emissions = ordered[prob_columns].to_numpy(dtype=np.float64)
            path_positions = viterbi_decode(
                emissions=emissions,
                prior=prior,
                transition=transition,
                transition_weight=transition_weight,
            )
            ordered["smoothed_label"] = [pos_to_label[int(pos)] for pos in path_positions]
            ordered["smoothed_stage"] = ordered["smoothed_label"].map(SLEEP_STAGE_NAMES)
            fold_parts.append(ordered)
        smoothed_fold = pd.concat(fold_parts).sort_index()
        fold_summary = summarize(
            smoothed_fold["true_label"].to_numpy(dtype=int),
            smoothed_fold["smoothed_label"].to_numpy(dtype=int),
            labels,
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "transition_weight": float(transition_weight),
                **{
                    key: fold_summary[key]
                    for key in ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1", "n1_f1"]
                },
            }
        )
        smoothed_parts.append(smoothed_fold)

    smoothed = pd.concat(smoothed_parts).sort_index()
    summary = summarize(
        smoothed["true_label"].to_numpy(dtype=int),
        smoothed["smoothed_label"].to_numpy(dtype=int),
        labels,
    )
    return smoothed, {"summary": summary, "fold_metrics": fold_rows}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
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


def write_report(output_path: Path, raw_summary: dict, summary_frame: pd.DataFrame, best_row: pd.Series) -> None:
    lines = [
        "# Viterbi Transition Smoothing",
        "",
        "## Raw Baseline",
        "",
        f"- Balanced accuracy: `{raw_summary['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{raw_summary['macro_f1']:.4f}`",
        f"- N1 F1: `{raw_summary['n1_f1']:.4f}`",
        "",
        "## Smoothing Sweep",
        "",
        markdown_table(summary_frame),
        "",
        "## Best Sensitivity Setting",
        "",
        f"- Transition weight: `{best_row['transition_weight']:.4f}`",
        f"- Balanced accuracy: `{best_row['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{best_row['macro_f1']:.4f}`",
        f"- N1 F1: `{best_row['n1_f1']:.4f}`",
        "",
        "This sweep is a sensitivity analysis. A thesis headline result should use a pre-declared transition weight or nested validation.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth CV prediction probabilities with Viterbi transitions")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--transition-weights", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--transition-smoothing", type=float, default=1.0)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_path)
    labels = sorted(int(label) for label in set(predictions["true_label"]).union(predictions["pred_label"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = summarize(
        predictions["true_label"].to_numpy(dtype=int),
        predictions["pred_label"].to_numpy(dtype=int),
        labels,
    )
    rows = []
    results = {}
    for transition_weight in [float(value) for value in args.transition_weights.split(",")]:
        smoothed, result = apply_smoothing(
            predictions,
            labels=labels,
            transition_weight=transition_weight,
            transition_smoothing=args.transition_smoothing,
        )
        summary = result["summary"]
        rows.append(
            {
                "transition_weight": float(transition_weight),
                **{
                    key: summary[key]
                    for key in [
                        "accuracy",
                        "balanced_accuracy",
                        "cohen_kappa",
                        "macro_f1",
                        "n1_precision",
                        "n1_recall",
                        "n1_f1",
                    ]
                },
            }
        )
        weight_tag = str(transition_weight).replace(".", "p")
        smoothed.to_csv(output_dir / f"cv_predictions_viterbi_w{weight_tag}.csv", index=False)
        pd.DataFrame(result["fold_metrics"]).to_csv(output_dir / f"fold_metrics_viterbi_w{weight_tag}.csv", index=False)
        results[str(transition_weight)] = result

    summary_frame = pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    summary_frame.to_csv(output_dir / "viterbi_smoothing_summary.csv", index=False)
    with open(output_dir / "viterbi_smoothing_results.json", "w", encoding="utf-8") as handle:
        json.dump({"raw_summary": raw_summary, "smoothing_results": results}, handle, indent=2)
    best_row = summary_frame.iloc[0]
    write_report(output_dir / "viterbi_smoothing_report.md", raw_summary, summary_frame, best_row)

    print(f"Saved Viterbi smoothing results to {output_dir}")
    print(
        f"Raw balanced_accuracy={raw_summary['balanced_accuracy']:.4f}; "
        f"best smoothed balanced_accuracy={best_row['balanced_accuracy']:.4f} "
        f"at transition_weight={best_row['transition_weight']:.4f}"
    )


if __name__ == "__main__":
    main()
