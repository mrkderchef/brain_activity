"""Visualization helpers and TVB export for sleep stage results."""

from __future__ import annotations

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .feature_extraction import FREQ_BANDS


SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB", 3: "#2C3E50", 4: "#9B59B6"}


def plot_confusion_matrix(y_true, y_pred, output_dir: str = "outputs") -> str:
    labels = sorted(set(y_true) | set(y_pred))
    names = [SLEEP_STAGE_NAMES.get(label, str(label)) for label in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[0])
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=axes[1],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (%)")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frequency_band_by_stage(X, y, output_dir: str = "outputs") -> str:
    band_names = list(FREQ_BANDS.keys())
    rel_indices = [idx * 3 + 2 for idx in range(len(band_names))]

    records = []
    for row_idx, label in enumerate(y):
        for band_idx, band_name in enumerate(band_names):
            records.append(
                {
                    "Sleep Stage": SLEEP_STAGE_NAMES.get(label, str(label)),
                    "Frequency Band": band_name.capitalize(),
                    "Relative Power": X[row_idx, rel_indices[band_idx]],
                }
            )
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df,
        x="Frequency Band",
        y="Relative Power",
        hue="Sleep Stage",
        palette=[
            STAGE_COLORS[key]
            for key in sorted(SLEEP_STAGE_NAMES)
            if SLEEP_STAGE_NAMES[key] in df["Sleep Stage"].values
        ],
        ax=ax,
    )
    ax.set_title("EEG Frequency Band Power by Sleep Stage")
    ax.legend(title="Sleep Stage", loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "frequency_bands_by_stage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importances, output_dir: str = "outputs") -> str:
    feat_labels = []
    for band_name in FREQ_BANDS:
        feat_labels.extend([f"{band_name}\nmean", f"{band_name}\nstd", f"{band_name}\nrelative"])

    idx = np.argsort(importances)[::-1]
    band_colors = ["#E74C3C", "#F39C12", "#3498DB", "#2C3E50", "#9B59B6"]
    colors = [band_colors[i // 3 % len(band_colors)] for i in idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(importances)), importances[idx], color=colors)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feat_labels[i] for i in idx], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hypnogram(y_true, y_pred, output_dir: str = "outputs", max_epochs: int = 200) -> str:
    n = min(len(y_true), max_epochs)
    time_min = np.arange(n) * 0.5

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    for ax, data, title in [
        (axes[0], y_true[:n], "True"),
        (axes[1], y_pred[:n], "Predicted"),
    ]:
        for idx in range(len(data) - 1):
            ax.fill_between(
                [time_min[idx], time_min[idx + 1]],
                data[idx],
                data[idx],
                step="post",
                alpha=0.7,
                color=STAGE_COLORS.get(data[idx], "gray"),
            )
        ax.step(time_min, data, where="post", color="black", linewidth=0.5)
        ax.set_ylabel("Sleep Stage")
        ax.set_yticks(list(SLEEP_STAGE_NAMES.keys()))
        ax.set_yticklabels(list(SLEEP_STAGE_NAMES.values()))
        ax.set_title(title)
        ax.invert_yaxis()

    axes[1].set_xlabel("Time (minutes)")
    patches = [mpatches.Patch(color=STAGE_COLORS[key], label=value) for key, value in SLEEP_STAGE_NAMES.items()]
    fig.legend(handles=patches, loc="upper right", ncol=5, fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "hypnogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_psd_by_stage(X, y, output_dir: str = "outputs") -> str:
    band_names = list(FREQ_BANDS.keys())
    mean_indices = [idx * 3 for idx in range(len(band_names))]
    stages = sorted(set(y))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, len(band_names), endpoint=False).tolist()
    angles += angles[:1]

    for stage in stages:
        mask = y == stage
        means = [np.mean(X[mask, idx]) for idx in mean_indices]
        total = sum(means) if sum(means) > 0 else 1
        means_norm = [value / total for value in means]
        means_norm += means_norm[:1]
        ax.plot(angles, means_norm, "o-", label=SLEEP_STAGE_NAMES[stage], color=STAGE_COLORS[stage], linewidth=2)
        ax.fill(angles, means_norm, alpha=0.1, color=STAGE_COLORS[stage])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name.capitalize() for name in band_names])
    ax.set_title("Frequency Profile by Sleep Stage", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = os.path.join(output_dir, "psd_radar_by_stage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def prepare_tvb_data(X, y, y_proba, output_dir: str = "outputs") -> str:
    tvb_dir = os.path.join(output_dir, "tvb_export")
    os.makedirs(tvb_dir, exist_ok=True)

    band_names = list(FREQ_BANDS.keys())
    mean_indices = [idx * 3 for idx in range(len(band_names))]

    band_timeseries = X[:, mean_indices]
    ts_df = pd.DataFrame(band_timeseries, columns=[name.capitalize() for name in band_names])
    ts_df["time_sec"] = np.arange(len(X)) * 30.0
    ts_df["sleep_stage"] = y
    ts_df.to_csv(os.path.join(tvb_dir, "band_power_timeseries.csv"), index=False)

    if y_proba is not None:
        n_classes = y_proba.shape[1]
        proba_cols = [SLEEP_STAGE_NAMES.get(i, f"Stage-{i}") for i in range(n_classes)]
        proba_df = pd.DataFrame(y_proba, columns=proba_cols)
        proba_df["time_sec"] = np.arange(len(y_proba)) * 30.0
        proba_df.to_csv(os.path.join(tvb_dir, "prediction_probabilities.csv"), index=False)

    for stage in sorted(set(y)):
        mask = y == stage
        if np.sum(mask) < 5:
            continue
        corr = np.corrcoef(X[mask][:, mean_indices].T)
        corr_df = pd.DataFrame(
            corr,
            index=[name.capitalize() for name in band_names],
            columns=[name.capitalize() for name in band_names],
        )
        corr_df.to_csv(os.path.join(tvb_dir, f"connectivity_{SLEEP_STAGE_NAMES.get(stage, stage)}.csv"))

    manifest = {
        "description": "Sleep stage prediction data for TVB visualization",
        "dataset": "ds003768 - Simultaneous EEG and fMRI during sleep",
        "epoch_duration_sec": 30.0,
        "frequency_bands": {key: list(value) for key, value in FREQ_BANDS.items()},
        "sleep_stages": SLEEP_STAGE_NAMES,
        "files": {
            "timeseries": "band_power_timeseries.csv",
            "probabilities": "prediction_probabilities.csv",
            "connectivity_pattern": "connectivity_{stage}.csv",
        },
    }
    with open(os.path.join(tvb_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return tvb_dir


def create_all_visualizations(X, y, y_pred, y_proba, feature_importances, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(y, y_pred, output_dir)
    plot_frequency_band_by_stage(X, y, output_dir)
    plot_feature_importance(feature_importances, output_dir)
    plot_hypnogram(y, y_pred, output_dir)
    plot_psd_by_stage(X, y, output_dir)
    return prepare_tvb_data(X, y_pred, y_proba, output_dir)
