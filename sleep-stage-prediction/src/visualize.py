"""
visualize.py – Visualisierungen der Schlafphasen-Klassifikation und TVB-Vorbereitung.

Enthält:
1. Confusion Matrix Heatmap
2. Frequenzband-Power pro Schlafstadium
3. Feature Importance Plot
4. Hypnogramm mit Vorhersagen
5. TVB-kompatible Datenaufbereitung
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.feature_extraction import FREQ_BANDS

SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB", 3: "#2C3E50", 4: "#9B59B6"}


def plot_confusion_matrix(y_true, y_pred, output_dir="outputs"):
    """Confusion Matrix als Heatmap."""
    labels = sorted(set(y_true) | set(y_pred))
    names = [SLEEP_STAGE_NAMES.get(l, str(l)) for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalisiert (Prozent)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_xlabel("Vorhergesagt")
    axes[0].set_ylabel("Tatsächlich")
    axes[0].set_title("Confusion Matrix (absolut)")

    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[1])
    axes[1].set_xlabel("Vorhergesagt")
    axes[1].set_ylabel("Tatsächlich")
    axes[1].set_title("Confusion Matrix (% pro Klasse)")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


def plot_frequency_band_by_stage(X, y, output_dir="outputs"):
    """
    Zeigt die mittlere Bandleistung (relative Power) pro Schlafstadium.
    Dies ist der Kern-Plot: Frequenz → Schlafphase.
    """
    band_names = list(FREQ_BANDS.keys())

    # Relative-Power-Indizes (jedes 3. Feature ab Index 2)
    rel_indices = [i * 3 + 2 for i in range(len(band_names))]

    records = []
    for i, label in enumerate(y):
        for j, band in enumerate(band_names):
            records.append({
                "Schlafstadium": SLEEP_STAGE_NAMES.get(label, str(label)),
                "Frequenzband": band.capitalize(),
                "Relative Power": X[i, rel_indices[j]],
            })
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, x="Frequenzband", y="Relative Power", hue="Schlafstadium",
                palette=[STAGE_COLORS[k] for k in sorted(SLEEP_STAGE_NAMES.keys())
                         if SLEEP_STAGE_NAMES[k] in df["Schlafstadium"].values],
                ax=ax)
    ax.set_title("EEG-Frequenzband-Power pro Schlafstadium", fontsize=14)
    ax.set_ylabel("Relative Bandleistung")
    ax.legend(title="Schlafstadium", loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "frequency_bands_by_stage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


def plot_feature_importance(importances, output_dir="outputs"):
    """Feature Importance des Random Forest."""
    band_names = list(FREQ_BANDS.keys())
    feat_labels = []
    for band in band_names:
        feat_labels.extend([f"{band}\nmean", f"{band}\nstd", f"{band}\nrelative"])

    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    band_colors = ["#E74C3C", "#F39C12", "#3498DB", "#2C3E50", "#9B59B6"]
    for i in idx:
        band_idx = i // 3
        colors.append(band_colors[band_idx % len(band_colors)])

    ax.bar(range(len(importances)), importances[idx], color=colors)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feat_labels[i] for i in idx], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance – Welche Frequenzbänder sind am wichtigsten?", fontsize=13)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


def plot_hypnogram(y_true, y_pred, output_dir="outputs", max_epochs=200):
    """Hypnogramm: Tatsächliche vs. vorhergesagte Schlafstadien über die Zeit."""
    n = min(len(y_true), max_epochs)
    time_min = np.arange(n) * 0.5  # 30-Sekunden-Epochen in Minuten

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    for ax, data, title in [(axes[0], y_true[:n], "Tatsächlich (Scored)"),
                             (axes[1], y_pred[:n], "Vorhergesagt (ML)")]:
        for i in range(len(data) - 1):
            ax.fill_between([time_min[i], time_min[i + 1]], data[i], data[i],
                            step="post", alpha=0.7,
                            color=STAGE_COLORS.get(data[i], "gray"))
        ax.step(time_min, data, where="post", color="black", linewidth=0.5)
        ax.set_ylabel("Schlafstadium")
        ax.set_yticks(list(SLEEP_STAGE_NAMES.keys()))
        ax.set_yticklabels(list(SLEEP_STAGE_NAMES.values()))
        ax.set_title(title)
        ax.invert_yaxis()

    axes[1].set_xlabel("Zeit (Minuten)")

    # Legende
    patches = [mpatches.Patch(color=STAGE_COLORS[k], label=v) for k, v in SLEEP_STAGE_NAMES.items()]
    fig.legend(handles=patches, loc="upper right", ncol=5, fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "hypnogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


def plot_psd_by_stage(X, y, output_dir="outputs"):
    """
    Zeigt mittlere absolute Power pro Frequenzband als Radar/Spider-Chart
    für jede Schlafphase.
    """
    band_names = list(FREQ_BANDS.keys())
    mean_indices = [i * 3 for i in range(len(band_names))]

    stages = sorted(set(y))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(band_names), endpoint=False).tolist()
    angles += angles[:1]

    for stage in stages:
        mask = y == stage
        means = [np.mean(X[mask, idx]) for idx in mean_indices]
        # Normalisiere auf 0-1 für Vergleichbarkeit
        total = sum(means) if sum(means) > 0 else 1
        means_norm = [m / total for m in means]
        means_norm += means_norm[:1]

        ax.plot(angles, means_norm, "o-", label=SLEEP_STAGE_NAMES[stage],
                color=STAGE_COLORS[stage], linewidth=2)
        ax.fill(angles, means_norm, alpha=0.1, color=STAGE_COLORS[stage])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([b.capitalize() for b in band_names])
    ax.set_title("Frequenzprofil pro Schlafstadium", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = os.path.join(output_dir, "psd_radar_by_stage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
    return path


# ---------------------------------------------------------------------------
# TVB-Vorbereitung
# ---------------------------------------------------------------------------

def prepare_tvb_data(X, y, y_proba, output_dir="outputs"):
    """
    Bereitet Daten für die Visualisierung in The Virtual Brain (TVB) vor.

    Exportiert:
    1. Zeitreihen der Frequenzband-Power (TVB RegionTimeSeriesFormat)
    2. Schlafstadien-Vorhersagen mit Wahrscheinlichkeiten
    3. Konnektivitäts-Proxy basierend auf Frequenzband-Korrelationen
    """
    tvb_dir = os.path.join(output_dir, "tvb_export")
    os.makedirs(tvb_dir, exist_ok=True)

    band_names = list(FREQ_BANDS.keys())

    # 1. Frequenzband-Zeitreihen (für TVB Time Series Visualizer)
    mean_indices = [i * 3 for i in range(len(band_names))]
    band_timeseries = X[:, mean_indices]
    ts_df = pd.DataFrame(band_timeseries, columns=[b.capitalize() for b in band_names])
    ts_df["time_sec"] = np.arange(len(X)) * 30.0
    ts_df["sleep_stage"] = y
    ts_path = os.path.join(tvb_dir, "band_power_timeseries.csv")
    ts_df.to_csv(ts_path, index=False)
    print(f"  TVB Zeitreihen: {ts_path}")

    # 2. Vorhersage-Wahrscheinlichkeiten (für TVB State Variables)
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        proba_cols = [SLEEP_STAGE_NAMES.get(i, f"Stage-{i}") for i in range(n_classes)]
        proba_df = pd.DataFrame(y_proba, columns=proba_cols)
        proba_df["time_sec"] = np.arange(len(y_proba)) * 30.0
        proba_path = os.path.join(tvb_dir, "prediction_probabilities.csv")
        proba_df.to_csv(proba_path, index=False)
        print(f"  TVB Vorhersage-Wahrscheinlichkeiten: {proba_path}")

    # 3. Konnektivitätsmatrix pro Schlafphase (Frequenzband-Korrelationen)
    stages = sorted(set(y))
    for stage in stages:
        mask = y == stage
        if np.sum(mask) < 5:
            continue
        stage_data = X[mask][:, mean_indices]
        corr = np.corrcoef(stage_data.T)
        corr_df = pd.DataFrame(corr,
                                index=[b.capitalize() for b in band_names],
                                columns=[b.capitalize() for b in band_names])
        stage_name = SLEEP_STAGE_NAMES.get(stage, str(stage))
        corr_path = os.path.join(tvb_dir, f"connectivity_{stage_name}.csv")
        corr_df.to_csv(corr_path)

    print(f"  TVB Konnektivitätsmatrizen: {tvb_dir}/connectivity_*.csv")

    # 4. TVB-kompatibles JSON Manifest
    manifest = {
        "description": "Sleep stage prediction data for TVB visualization",
        "dataset": "ds003768 - Simultaneous EEG and fMRI during sleep",
        "epoch_duration_sec": 30.0,
        "frequency_bands": {k: list(v) for k, v in FREQ_BANDS.items()},
        "sleep_stages": SLEEP_STAGE_NAMES,
        "files": {
            "timeseries": "band_power_timeseries.csv",
            "probabilities": "prediction_probabilities.csv",
            "connectivity_pattern": "connectivity_{stage}.csv",
        }
    }
    import json
    manifest_path = os.path.join(tvb_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  TVB Manifest: {manifest_path}")

    return tvb_dir


def create_all_visualizations(X, y, y_pred, y_proba, feature_importances,
                               output_dir="outputs"):
    """Erstellt alle Visualisierungen."""
    os.makedirs(output_dir, exist_ok=True)
    print("\nErstelle Visualisierungen ...")

    plot_confusion_matrix(y, y_pred, output_dir)
    plot_frequency_band_by_stage(X, y, output_dir)
    plot_feature_importance(feature_importances, output_dir)
    plot_hypnogram(y, y_pred, output_dir)
    plot_psd_by_stage(X, y, output_dir)

    print("\nBereite TVB-Export vor ...")
    tvb_dir = prepare_tvb_data(X, y_pred, y_proba, output_dir)

    print(f"\nAlle Visualisierungen gespeichert in: {output_dir}/")
    return tvb_dir
