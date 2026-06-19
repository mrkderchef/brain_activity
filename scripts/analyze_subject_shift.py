"""Analyze subject-level domain shift for spectrogram sleep-stage models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


def make_frequency_axis(n_freqs: int, sfreq: float, nperseg: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    keep = (freqs >= fmin) & (freqs <= fmax)
    kept = freqs[keep]
    if len(kept) != n_freqs:
        return np.linspace(fmin, fmax, n_freqs)
    return kept


def add_z_scores(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        std = values.std(ddof=0)
        if std == 0 or pd.isna(std):
            result[f"z_{column}"] = 0.0
        else:
            result[f"z_{column}"] = (values - values.mean()) / std
    return result


def subject_from_group(group_id: str) -> str:
    if ":" in group_id:
        return group_id.split(":", 1)[1]
    return group_id


def summarize_subject_spectrograms(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    group_ids: pd.Series,
    freqs: np.ndarray,
) -> pd.DataFrame:
    rows = []
    band_defs = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "sigma": (12.0, 16.0),
        "beta": (16.0, 30.0),
        "high": (30.0, 40.0),
    }
    group_array = group_ids.to_numpy()
    for group_id in sorted(group_ids.unique()):
        idx = np.where(group_array == group_id)[0]
        subject_x = np.asarray(X[idx], dtype=np.float32)
        subject_y = y[idx]
        channel_means = subject_x.mean(axis=(0, 2, 3))
        row = {
            "analysis_group": group_id,
            "subject": subject_from_group(group_id),
            "n_epochs": int(len(idx)),
            "spectrogram_mean": float(subject_x.mean()),
            "spectrogram_std": float(subject_x.std()),
            "spectrogram_p01": float(np.percentile(subject_x, 1)),
            "spectrogram_p99": float(np.percentile(subject_x, 99)),
            "channel_mean_range": float(channel_means.max() - channel_means.min()),
            "channel_mean_std": float(channel_means.std()),
            "epoch_power_std": float(subject_x.mean(axis=(1, 2, 3)).std()),
            "freq_profile_std": float(subject_x.mean(axis=(0, 1, 3)).std()),
        }
        total_band_power = 0.0
        for band_name, (low, high) in band_defs.items():
            keep = (freqs >= low) & (freqs < high)
            if not np.any(keep):
                row[f"band_{band_name}_mean"] = np.nan
                continue
            band_power = float(subject_x[:, :, keep, :].mean())
            row[f"band_{band_name}_mean"] = band_power
            total_band_power += band_power
        for band_name in band_defs:
            value = row[f"band_{band_name}_mean"]
            row[f"band_{band_name}_share"] = float(value / total_band_power) if total_band_power and not pd.isna(value) else np.nan
        for label, count in zip(*np.unique(subject_y, return_counts=True)):
            stage_name = SLEEP_STAGE_NAMES.get(int(label), f"Stage-{int(label)}")
            row[f"support_{stage_name}"] = int(count)
            row[f"share_{stage_name}"] = float(count / len(subject_y))
        rows.append(row)
    frame = pd.DataFrame(rows).fillna(0)
    for stage_name in SLEEP_STAGE_NAMES.values():
        if f"support_{stage_name}" not in frame.columns:
            frame[f"support_{stage_name}"] = 0
        if f"share_{stage_name}" not in frame.columns:
            frame[f"share_{stage_name}"] = 0.0
    return frame


def summarize_correlations(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number])
    excluded = {
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "macro_f1",
        "n1_precision",
        "n1_recall",
        "n1_f1",
        "recall_Wake",
        "recall_N1",
        "recall_N2",
        "recall_N3",
        "recall_REM",
    }
    rows = []
    for column in numeric.columns:
        if column in excluded or column.startswith("support_") or column == "fold":
            continue
        corr = numeric[[target, column]].corr().iloc[0, 1]
        if pd.notna(corr):
            rows.append({"feature": column, "correlation_with_balanced_accuracy": float(corr)})
    return pd.DataFrame(rows).assign(abs_corr=lambda x: x["correlation_with_balanced_accuracy"].abs()).sort_values(
        "abs_corr",
        ascending=False,
    )


def markdown_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
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


def write_report(output_path: Path, merged: pd.DataFrame, correlations: pd.DataFrame, outliers: pd.DataFrame) -> None:
    weakest_columns = [
        "subject",
        "balanced_accuracy",
        "macro_f1",
        "n1_f1",
        "share_N1",
        "share_N2",
        "spectrogram_mean",
        "spectrogram_std",
        "channel_mean_range",
    ]
    if "fold" in merged.columns:
        weakest_columns.insert(1, "fold")
    outlier_columns = [
        "subject",
        "balanced_accuracy",
        "shift_score",
        "z_spectrogram_mean",
        "z_spectrogram_std",
        "z_channel_mean_range",
        "z_epoch_power_std",
        "z_band_delta_share",
        "z_band_beta_share",
    ]
    lines = [
        "# Subject Shift Analysis",
        "",
        "## Weakest Subjects",
        "",
        markdown_table(merged[weakest_columns].sort_values("balanced_accuracy")),
        "",
        "## Largest Spectrogram Shift Scores",
        "",
        markdown_table(outliers[outlier_columns]),
        "",
        "## Strongest Numeric Associations",
        "",
        markdown_table(correlations[["feature", "correlation_with_balanced_accuracy"]]),
        "",
        "Interpretation note: correlations use only 19 subjects, so they are diagnostic clues rather than statistical claims.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze subject-level shift in spectrogram experiments")
    parser.add_argument("--spectrograms-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--subject-metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sfreq", type=float, default=500.0)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=40.0)
    args = parser.parse_args()

    X = np.load(args.spectrograms_path, mmap_mode="r")
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Spectrogram, label, and metadata row counts must match")

    group_ids = derive_group_ids(metadata)
    freqs = make_frequency_axis(X.shape[2], args.sfreq, args.nperseg, args.fmin, args.fmax)
    subject_stats = summarize_subject_spectrograms(X, y, metadata, group_ids, freqs)
    performance = pd.read_csv(args.subject_metrics_path)
    merged = performance.merge(subject_stats, on="analysis_group", how="left", suffixes=("", "_spectrogram"))

    shift_columns = [
        "spectrogram_mean",
        "spectrogram_std",
        "channel_mean_range",
        "channel_mean_std",
        "epoch_power_std",
        "freq_profile_std",
        "band_delta_share",
        "band_theta_share",
        "band_alpha_share",
        "band_sigma_share",
        "band_beta_share",
        "band_high_share",
    ]
    merged = add_z_scores(merged, shift_columns)
    merged["shift_score"] = np.sqrt(np.mean(np.square(merged[[f"z_{column}" for column in shift_columns]]), axis=1))
    merged = merged.sort_values("balanced_accuracy").reset_index(drop=True)
    correlations = summarize_correlations(merged, "balanced_accuracy")
    outliers = merged.sort_values("shift_score", ascending=False).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "subject_shift_metrics.csv", index=False)
    correlations.to_csv(output_dir / "subject_shift_correlations.csv", index=False)
    outliers.to_csv(output_dir / "subject_shift_outliers.csv", index=False)
    write_report(output_dir / "subject_shift_report.md", merged, correlations, outliers)

    summary = {
        "n_subjects": int(len(merged)),
        "weakest_subjects": merged[["subject", "balanced_accuracy", "macro_f1", "n1_f1"]].head(5).to_dict(orient="records"),
        "largest_shift_subjects": outliers[["subject", "shift_score", "balanced_accuracy"]].head(5).to_dict(orient="records"),
        "top_correlations": correlations[["feature", "correlation_with_balanced_accuracy"]].head(8).to_dict(orient="records"),
    }
    with open(output_dir / "subject_shift_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved subject-shift analysis to {output_dir}")
    print("Weakest subjects:")
    for row in summary["weakest_subjects"]:
        print(
            f"  {row['subject']}: balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, n1_f1={row['n1_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
