"""Create a thesis-friendly model selection summary from experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "macro_f1",
    "n1_precision",
    "n1_recall",
    "n1_f1",
]


def infer_feature_set(path: Path) -> str:
    text = path.as_posix().lower()
    if "transition" in text:
        return "transition"
    if "seq1" in text or "sequence" in text:
        return "sequence_context"
    if "normalized" in text:
        return "augmented_normalized"
    if "augmented" in text:
        return "augmented"
    return "unknown"


def add_row(rows: list[dict], row: dict) -> None:
    for column in METRIC_COLUMNS:
        row.setdefault(column, None)
    rows.append(row)


def read_model_comparison(path: Path, outputs_root: Path, rows: list[dict]) -> None:
    frame = pd.read_csv(path)
    for record in frame.to_dict(orient="records"):
        add_row(
            rows,
            {
                "source_file": str(path.relative_to(outputs_root.parent)),
                "experiment": path.parent.name,
                "feature_set": infer_feature_set(path),
                "model": record["model"],
                "selection_role": "candidate",
                "is_thresholded": False,
                **{column: record.get(column) for column in METRIC_COLUMNS},
            },
        )


def read_group_metrics(path: Path, outputs_root: Path, rows: list[dict]) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("cv") != "StratifiedGroupKFold":
        return
    report = metrics.get("classification_report", {})
    n1 = report.get("N1", {})
    macro = report.get("macro avg", {})
    add_row(
        rows,
        {
            "source_file": str(path.relative_to(outputs_root.parent)),
            "experiment": path.parent.name,
            "feature_set": infer_feature_set(path),
            "model": "random_forest",
            "selection_role": "candidate",
            "is_thresholded": False,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "cohen_kappa": metrics.get("cohen_kappa"),
            "macro_f1": macro.get("f1-score"),
            "n1_precision": n1.get("precision"),
            "n1_recall": n1.get("recall"),
            "n1_f1": n1.get("f1-score"),
        },
    )


def read_optuna_results(path: Path, outputs_root: Path, rows: list[dict]) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    summary = result.get("best_summary", {})
    add_row(
        rows,
        {
            "source_file": str(path.relative_to(outputs_root.parent)),
            "experiment": path.parent.name,
            "feature_set": infer_feature_set(path),
            "model": f"random_forest_optuna_{result.get('objective', 'unknown')}",
            "selection_role": "sensitivity" if result.get("objective") != "balanced_accuracy" else "candidate",
            "is_thresholded": False,
            **{column: summary.get(column) for column in METRIC_COLUMNS},
        },
    )


def read_n1_focus(path: Path, outputs_root: Path, rows: list[dict]) -> None:
    frame = pd.read_csv(path)
    for record in frame.to_dict(orient="records"):
        threshold = record.get("n1_threshold")
        is_thresholded = pd.notna(threshold)
        model = "random_forest_n1_threshold" if is_thresholded else "random_forest_n1_weighted"
        add_row(
            rows,
            {
                "source_file": str(path.relative_to(outputs_root.parent)),
                "experiment": path.parent.name,
                "feature_set": infer_feature_set(path),
                "model": model,
                "selection_role": "sensitivity",
                "is_thresholded": bool(is_thresholded),
                "n1_weight": record.get("n1_weight"),
                "n1_threshold": threshold,
                **{column: record.get(column) for column in METRIC_COLUMNS},
            },
        )


def write_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    primary = summary[
        (summary["selection_role"] == "candidate")
        & (~summary["is_thresholded"])
        & summary["balanced_accuracy"].notna()
    ].sort_values(["balanced_accuracy", "macro_f1"], ascending=False)

    sensitivity = summary[
        summary["n1_f1"].notna()
    ].sort_values(["n1_f1", "balanced_accuracy"], ascending=False)

    best = primary.iloc[0]
    best_n1 = sensitivity.iloc[0]

    primary_columns = [
        "model",
        "feature_set",
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "macro_f1",
        "n1_recall",
        "n1_f1",
        "experiment",
    ]
    top_candidates = primary[primary_columns].head(12)
    lines = [
        "# Model Selection Summary",
        "",
        "Selection rule: choose the main model by subject-wise balanced accuracy, using macro F1 as the tie-breaker. N1-focused thresholding and N1-weighted sensitivity models are reported separately, not selected as the main model.",
        "",
        "## Recommended Main Model",
        "",
        f"- Model: `{best['model']}`",
        f"- Feature set: `{best['feature_set']}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        f"- N1 F1: `{best['n1_f1']:.4f}`",
        f"- Source experiment: `{best['experiment']}`",
        "",
        "## Top Main-Model Candidates",
        "",
        markdown_table(top_candidates),
        "",
        "## Best N1-Sensitive Result",
        "",
        f"- Model: `{best_n1['model']}`",
        f"- Feature set: `{best_n1['feature_set']}`",
        f"- Balanced accuracy: `{best_n1['balanced_accuracy']:.4f}`",
        f"- N1 recall: `{best_n1['n1_recall']:.4f}`",
        f"- N1 F1: `{best_n1['n1_f1']:.4f}`",
        f"- Source experiment: `{best_n1['experiment']}`",
        "",
        "This N1-sensitive result is useful for discussion, but it is not the recommended main model because it is optimized toward N1 rather than balanced 5-class performance.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model-selection outputs")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in outputs_root.rglob("model_comparison_summary.csv"):
        read_model_comparison(path, outputs_root, rows)
    for path in outputs_root.rglob("metrics.json"):
        read_group_metrics(path, outputs_root, rows)
    for path in outputs_root.rglob("optuna_rf_results.json"):
        read_optuna_results(path, outputs_root, rows)
    for path in outputs_root.rglob("n1_focus_summary.csv"):
        read_n1_focus(path, outputs_root, rows)

    if not rows:
        raise RuntimeError(f"No model-selection results found under {outputs_root}")

    summary = pd.DataFrame(rows)
    for column in METRIC_COLUMNS:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
        summary[f"_{column}_round"] = summary[column].round(6)
    summary = summary.drop_duplicates(
        subset=[
            "feature_set",
            "model",
            "selection_role",
            "is_thresholded",
            "_accuracy_round",
            "_balanced_accuracy_round",
            "_macro_f1_round",
            "_n1_f1_round",
        ],
        keep="first",
    )
    summary = summary.drop(columns=[column for column in summary.columns if column.startswith("_")])
    summary = summary.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "model_selection_all_results.csv", index=False)

    main_candidates = summary[
        (summary["selection_role"] == "candidate")
        & (~summary["is_thresholded"])
        & summary["balanced_accuracy"].notna()
    ].sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    main_candidates.to_csv(output_dir / "model_selection_main_candidates.csv", index=False)
    write_markdown(summary, output_dir / "model_selection_summary.md")

    best = main_candidates.iloc[0]
    print(f"Saved model selection summary to {output_dir}")
    print(
        "Recommended main model: "
        f"{best['model']} on {best['feature_set']} "
        f"(balanced_accuracy={best['balanced_accuracy']:.4f}, macro_f1={best['macro_f1']:.4f})"
    )


if __name__ == "__main__":
    main()
