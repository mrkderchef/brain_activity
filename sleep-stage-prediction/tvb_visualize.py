"""
tvb_visualize.py – Integration mit The Virtual Brain (TVB) zur Gehirn-Visualisierung.

Dieses Skript liest die exportierten Daten und visualisiert sie mit TVB.
Kann eigenständig ausgeführt werden, nachdem main.py gelaufen ist.

Verwendung:
    python tvb_visualize.py
    python tvb_visualize.py --export-dir outputs/tvb_export
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB", 3: "#2C3E50", 4: "#9B59B6"}


def plot_tvb_brain_state_timeline(export_dir: str, output_dir: str = None):
    """
    Erstellt eine TVB-inspirierte Visualisierung der Gehirnzustände über die Zeit.
    Zeigt Frequenzband-Power als Heatmap zusammen mit Schlafstadien.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(export_dir), "tvb_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Daten laden
    ts_path = os.path.join(export_dir, "band_power_timeseries.csv")
    proba_path = os.path.join(export_dir, "prediction_probabilities.csv")

    ts_df = pd.read_csv(ts_path)
    proba_df = pd.read_csv(proba_path) if os.path.exists(proba_path) else None

    time_min = ts_df["time_sec"].values / 60.0
    bands = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
    band_data = ts_df[bands].values
    stages = ts_df["sleep_stage"].values

    # -- Figure 1: Brain State Timeline --
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 3, 2, 1], hspace=0.3)

    # 1. Hypnogramm
    ax_hyp = fig.add_subplot(gs[0])
    # Effizientes Plotting: eine Linie + Farbhintergrund pro zusammenhängendem Block
    ax_hyp.step(time_min, stages, where="post", color="black", linewidth=0.8)
    for stage_id, color in STAGE_COLORS.items():
        mask = stages == stage_id
        if np.any(mask):
            ax_hyp.fill_between(time_min, -0.5, max(stages) + 0.5,
                                where=mask, color=color, alpha=0.15, step="post")
    ax_hyp.set_yticks(list(SLEEP_STAGE_NAMES.keys()))
    ax_hyp.set_yticklabels(list(SLEEP_STAGE_NAMES.values()))
    ax_hyp.invert_yaxis()
    ax_hyp.set_title("Schlafstadien-Verlauf (Hypnogramm)", fontsize=12, fontweight="bold")
    ax_hyp.set_xlim(time_min[0], time_min[-1])

    # 2. Frequenzband-Heatmap (TVB-Style)
    ax_heat = fig.add_subplot(gs[1])
    # Normalisiere pro Band für bessere Sichtbarkeit
    band_range = band_data.max(axis=0) - band_data.min(axis=0)
    band_norm = (band_data - band_data.min(axis=0)) / (band_range + 1e-10)
    im = ax_heat.imshow(band_norm.T, aspect="auto", cmap="inferno",
                        extent=[time_min[0], time_min[-1], -0.5, len(bands) - 0.5],
                        origin="lower", interpolation="bilinear")
    ax_heat.set_yticks(range(len(bands)))
    ax_heat.set_yticklabels(bands)
    ax_heat.set_title("EEG-Frequenzband-Aktivität (normalisiert)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, label="Normalisierte Power", shrink=0.8)

    # 3. Vorhersage-Wahrscheinlichkeiten
    if proba_df is not None:
        ax_proba = fig.add_subplot(gs[2])
        proba_time = proba_df["time_sec"].values / 60.0
        for stage_id, name in SLEEP_STAGE_NAMES.items():
            if name in proba_df.columns:
                ax_proba.fill_between(proba_time, proba_df[name].values,
                                      alpha=0.4, color=STAGE_COLORS[stage_id], label=name)
                ax_proba.plot(proba_time, proba_df[name].values,
                              color=STAGE_COLORS[stage_id], linewidth=0.5)
        ax_proba.set_ylabel("P(Schlafstadium)")
        ax_proba.set_title("Vorhersage-Wahrscheinlichkeiten", fontsize=12, fontweight="bold")
        ax_proba.legend(loc="upper right", ncol=5, fontsize=8)
        ax_proba.set_xlim(time_min[0], time_min[-1])
        ax_proba.set_ylim(0, 1)

    # 4. Dominante Frequenz
    ax_dom = fig.add_subplot(gs[3])
    band_freqs = [2.25, 6.0, 10.0, 14.0, 23.0]  # Mittlere Frequenzen
    dominant_freq = np.array([band_freqs[np.argmax(band_data[i])] for i in range(len(band_data))])
    colors_arr = np.array([STAGE_COLORS.get(s, "gray") for s in stages])
    for stage_id, color in STAGE_COLORS.items():
        mask = stages == stage_id
        if np.any(mask):
            ax_dom.scatter(time_min[mask], dominant_freq[mask], c=color,
                           s=3, alpha=0.5, label=SLEEP_STAGE_NAMES[stage_id], rasterized=True)
    ax_dom.set_ylabel("Dominante\nFrequenz (Hz)")
    ax_dom.set_xlabel("Zeit (Minuten)")
    ax_dom.set_title("Dominante EEG-Frequenz", fontsize=12, fontweight="bold")
    ax_dom.set_xlim(time_min[0], time_min[-1])

    plt.suptitle("TVB Brain State Visualization – Sleep Stage Dynamics",
                 fontsize=15, fontweight="bold", y=1.01)
    path = os.path.join(output_dir, "tvb_brain_state_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

    # -- Figure 2: Konnektivitätsmatrizen --
    plot_connectivity_matrices(export_dir, output_dir)

    return output_dir


def plot_connectivity_matrices(export_dir: str, output_dir: str):
    """Zeigt die Frequenzband-Konnektivität pro Schlafstadium (TVB-Style)."""
    import glob
    conn_files = sorted(glob.glob(os.path.join(export_dir, "connectivity_*.csv")))
    if not conn_files:
        return

    n = len(conn_files)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, conn_file in zip(axes, conn_files):
        stage_name = os.path.basename(conn_file).replace("connectivity_", "").replace(".csv", "")
        df = pd.read_csv(conn_file, index_col=0)
        im = ax.imshow(df.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_title(f"{stage_name}", fontsize=12, fontweight="bold")

    fig.suptitle("Frequenzband-Konnektivität pro Schlafstadium (TVB-Style)",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=axes, label="Korrelation", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "tvb_connectivity_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def main():
    parser = argparse.ArgumentParser(description="TVB-Visualisierung der Schlafphasen")
    parser.add_argument("--export-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "outputs", "tvb_export"))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.export_dir):
        print(f"FEHLER: TVB-Export nicht gefunden: {args.export_dir}")
        print("Bitte zuerst main.py ausführen.")
        sys.exit(1)

    print("Erstelle TVB-Visualisierungen ...")
    plot_tvb_brain_state_timeline(args.export_dir, args.output_dir)
    print("Fertig!")


if __name__ == "__main__":
    main()
