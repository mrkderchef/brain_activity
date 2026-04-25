"""Compare classifiers under the same stratified group cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


def optional_import(module_name: str, class_name: str):
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise RuntimeError(
            f"Model requires optional dependency {module_name!r}. "
            f"Install it first, e.g. python -m pip install {module_name}"
        ) from exc
    return getattr(module, class_name)


def make_model(name: str, parallel_jobs: int) -> Pipeline:
    if name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=parallel_jobs,
        )
    elif name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=parallel_jobs,
        )
    elif name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
    elif name == "gradient_boosting_fast":
        classifier = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.06,
            max_depth=2,
            subsample=0.8,
            random_state=42,
        )
    elif name == "hist_gradient_boosting_fast":
        classifier = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            l2_regularization=0.1,
            max_leaf_nodes=15,
            random_state=42,
        )
    elif name == "xgboost":
        XGBClassifier = optional_import("xgboost", "XGBClassifier")
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=parallel_jobs,
        )
    elif name == "lightgbm":
        LGBMClassifier = optional_import("lightgbm", "LGBMClassifier")
        classifier = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            n_jobs=parallel_jobs,
            verbose=-1,
        )
    elif name == "catboost":
        CatBoostClassifier = optional_import("catboost", "CatBoostClassifier")
        classifier = CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=42,
            thread_count=parallel_jobs,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


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
        "n1_precision": float(report.get("N1", {}).get("precision", 0.0)),
        "n1_recall": float(report.get("N1", {}).get("recall", 0.0)),
        "n1_f1": float(report.get("N1", {}).get("f1-score", 0.0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare group-CV models")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument(
        "--models",
        default="random_forest,extra_trees,gradient_boosting",
        help="Comma-separated model names",
    )
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    groups = derive_group_ids(metadata, args.group_column).astype(str)
    if groups.nunique() < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} groups, found {groups.nunique()}")

    labels = sorted(int(label) for label in np.unique(y))
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = {}
    predictions = metadata.copy()
    predictions["true_label"] = y
    predictions["true_stage"] = predictions["true_label"].map(SLEEP_STAGE_NAMES)
    predictions["cv_group"] = groups

    for model_name in models:
        y_pred = np.empty_like(y)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups.to_numpy()), start=1):
            model = make_model(model_name, args.parallel_jobs)
            fit_kwargs = {}
            if model_name in {"gradient_boosting", "gradient_boosting_fast", "hist_gradient_boosting_fast", "xgboost"}:
                fit_kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", y[train_idx])
            model.fit(X[train_idx], y[train_idx], **fit_kwargs)
            y_pred[test_idx] = np.asarray(model.predict(X[test_idx])).reshape(-1)

        summary = summarize(y, y_pred, labels)
        details[model_name] = summary
        predictions[f"{model_name}_pred_label"] = y_pred
        predictions[f"{model_name}_pred_stage"] = pd.Series(y_pred).map(SLEEP_STAGE_NAMES)
        rows.append(
            {
                "model": model_name,
                "accuracy": summary["accuracy"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "cohen_kappa": summary["cohen_kappa"],
                "macro_f1": summary["macro_f1"],
                "n1_precision": summary["n1_precision"],
                "n1_recall": summary["n1_recall"],
                "n1_f1": summary["n1_f1"],
            }
        )
        print(f"{model_name}: balanced_accuracy={summary['balanced_accuracy']:.4f}, n1_f1={summary['n1_f1']:.4f}")

    summary_table = pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    summary_table.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    predictions.to_csv(output_dir / "model_cv_predictions.csv", index=False)

    result = {
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "group_column": args.group_column,
        "n_groups": int(groups.nunique()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            SLEEP_STAGE_NAMES.get(int(label), str(int(label))): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
        "summary": summary_table.to_dict(orient="records"),
        "models": details,
    }
    with open(output_dir / "model_comparison_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved model comparison to {output_dir}")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
