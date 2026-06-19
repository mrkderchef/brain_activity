"""Tune Random Forest hyperparameters with Optuna and group-aware CV."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


N1_LABEL = 1


def parse_max_depth(value: str | None) -> int | None:
    if value in {None, "none", "None", ""}:
        return None
    return int(value)


def make_class_weight(mode: str, labels: np.ndarray, n1_weight_multiplier: float):
    if mode in {"balanced", "balanced_subsample"} and abs(n1_weight_multiplier - 1.0) < 1e-12:
        return mode

    if mode == "none":
        base = {int(label): 1.0 for label in np.unique(labels)}
    else:
        counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
        total = float(len(labels))
        n_classes = float(len(counts))
        base = {label: total / (n_classes * count) for label, count in counts.items()}

    base[N1_LABEL] = base.get(N1_LABEL, 1.0) * n1_weight_multiplier
    return base


def suggest_params(trial: optuna.Trial, y: np.ndarray, parallel_jobs: int, search_space: str) -> dict:
    class_weight_mode = trial.suggest_categorical("class_weight_mode", ["balanced", "balanced_subsample", "custom"])
    n1_weight_multiplier = trial.suggest_float("n1_weight_multiplier", 1.0, 4.0, step=0.25)
    if search_space == "fast":
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=100)
        max_depth_choice = trial.suggest_categorical("max_depth", ["12", "16", "20", "none"])
        max_samples_choice = trial.suggest_categorical("max_samples", ["0.65", "0.8", "0.95"])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3])
    else:
        n_estimators = trial.suggest_int("n_estimators", 200, 700, step=100)
        max_depth_choice = trial.suggest_categorical("max_depth", ["none", "12", "16", "20", "30"])
        max_samples_choice = trial.suggest_categorical("max_samples", ["none", "0.65", "0.8", "0.95"])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.75])

    params = {
        "n_estimators": n_estimators,
        "max_depth": parse_max_depth(max_depth_choice),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": max_features,
        "bootstrap": True,
        "max_samples": None if max_samples_choice == "none" else float(max_samples_choice),
        "class_weight": make_class_weight(class_weight_mode, y, n1_weight_multiplier),
        "random_state": 42,
        "n_jobs": parallel_jobs,
    }
    return params


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
        "n1_precision": float(report["N1"]["precision"]),
        "n1_recall": float(report["N1"]["recall"]),
        "n1_f1": float(report["N1"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def objective_value(summary: dict, objective_name: str, n1_weight: float) -> float:
    if objective_name == "balanced_accuracy":
        return summary["balanced_accuracy"]
    if objective_name == "n1_f1":
        return summary["n1_f1"]
    if objective_name == "combined":
        return (1.0 - n1_weight) * summary["balanced_accuracy"] + n1_weight * summary["n1_f1"]
    raise ValueError(f"Unknown objective: {objective_name}")


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[int],
    params: dict,
    n_splits: int,
) -> tuple[dict, np.ndarray]:
    y_pred = np.empty_like(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return summarize(y, y_pred, labels), y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Random Forest with Optuna and group-aware CV")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--objective", choices=["balanced_accuracy", "n1_f1", "combined"], default="combined")
    parser.add_argument("--combined-n1-weight", type=float, default=0.35)
    parser.add_argument("--study-name", default="random_forest_optuna")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-space", choices=["fast", "broad"], default="fast")
    args = parser.parse_args()

    X = np.load(args.features_path)
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Feature, label, and metadata row counts must match")

    groups = derive_group_ids(metadata, args.group_column).astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} groups, found {n_groups}")

    labels = sorted(int(label) for label in np.unique(y))
    if N1_LABEL not in labels:
        raise RuntimeError("N1 is not present in the supplied labels")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=args.study_name)

    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, y, args.parallel_jobs, args.search_space)
        summary, _ = run_cv(X, y, groups, labels, params, args.n_splits)
        score = objective_value(summary, args.objective, args.combined_n1_weight)
        trial.set_user_attr("summary", {key: value for key, value in summary.items() if key != "classification_report"})
        trial_rows.append(
            {
                "trial": trial.number,
                "score": score,
                **{key: summary[key] for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "cohen_kappa",
                    "macro_f1",
                    "n1_precision",
                    "n1_recall",
                    "n1_f1",
                ]},
                **trial.params,
            }
        )
        pd.DataFrame(trial_rows).sort_values("score", ascending=False).to_csv(
            output_dir / "optuna_trials_partial.csv",
            index=False,
        )
        print(
            f"trial={trial.number} score={score:.4f} "
            f"bal_acc={summary['balanced_accuracy']:.4f} n1_f1={summary['n1_f1']:.4f}"
            ,
            flush=True,
        )
        return score

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    best_params = suggest_params(study.best_trial, y, args.parallel_jobs, args.search_space)
    best_summary, best_pred = run_cv(X, y, groups, labels, best_params, args.n_splits)
    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X, y)

    trials = pd.DataFrame(trial_rows).sort_values("score", ascending=False)
    trials.to_csv(output_dir / "optuna_trials.csv", index=False)

    predictions = metadata.copy()
    predictions["cv_group"] = groups
    predictions["true_label"] = y
    predictions["pred_label"] = best_pred
    predictions["true_stage"] = predictions["true_label"].map(SLEEP_STAGE_NAMES)
    predictions["pred_stage"] = predictions["pred_label"].map(SLEEP_STAGE_NAMES)
    predictions.to_csv(output_dir / "best_cv_predictions.csv", index=False)
    joblib.dump(final_model, output_dir / "random_forest_optuna_best.joblib")

    result = {
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "n_trials": len(study.trials),
        "objective": args.objective,
        "combined_n1_weight": float(args.combined_n1_weight),
        "search_space": args.search_space,
        "n_groups": int(n_groups),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "best_trial": int(study.best_trial.number),
        "best_score": float(study.best_value),
        "best_trial_params": study.best_trial.params,
        "best_model_params": {
            key: value for key, value in best_params.items() if key not in {"class_weight"}
        },
        "best_class_weight": best_params["class_weight"],
        "best_summary": best_summary,
    }
    with open(output_dir / "optuna_rf_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved Optuna RF tuning results to {output_dir}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(
        f"Best CV: balanced_accuracy={best_summary['balanced_accuracy']:.4f}, "
        f"macro_f1={best_summary['macro_f1']:.4f}, n1_f1={best_summary['n1_f1']:.4f}, "
        f"n1_recall={best_summary['n1_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
