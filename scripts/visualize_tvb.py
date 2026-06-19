"""Create TVB-style visualizations from exported CSV files."""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLEEP_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#3498DB", 3: "#2C3E50", 4: "#9B59B6"}


def plot_connectivity_matrices(export_dir: str, output_dir: str) -> None:
    conn_files = sorted(glob.glob(os.path.join(export_dir, "connectivity_*.csv")))
    if not conn_files:
        return

    fig, axes = plt.subplots(1, len(conn_files), figsize=(5 * len(conn_files), 4.5))
    if len(conn_files) == 1:
        axes = [axes]

    for ax, conn_file in zip(axes, conn_files):
        stage_name = os.path.basename(conn_file).replace("connectivity_", "").replace(".csv", "")
        df = pd.read_csv(conn_file, index_col=0)
        im = ax.imshow(df.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_title(stage_name)

    fig.suptitle("Frequency-Band Connectivity by Sleep Stage")
    plt.colorbar(im, ax=axes, label="Correlation", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tvb_connectivity_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(export_dir: str, output_dir: str) -> None:
    ts_df = pd.read_csv(os.path.join(export_dir, "band_power_timeseries.csv"))
    proba_path = os.path.join(export_dir, "prediction_probabilities.csv")
    proba_df = pd.read_csv(proba_path) if os.path.exists(proba_path) else None

    time_min = ts_df["time_sec"].to_numpy() / 60.0
    bands = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
    band_data = ts_df[bands].to_numpy()
    stages = ts_df["sleep_stage"].to_numpy()

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 3, 2, 1], hspace=0.3)

    ax_hyp = fig.add_subplot(gs[0])
    ax_hyp.step(time_min, stages, where="post", color="black", linewidth=0.8)
    for stage_id, color in STAGE_COLORS.items():
        mask = stages == stage_id
        if np.any(mask):
            ax_hyp.fill_between(time_min, -0.5, max(stages) + 0.5, where=mask, color=color, alpha=0.15, step="post")
    ax_hyp.set_yticks(list(SLEEP_STAGE_NAMES.keys()))
    ax_hyp.set_yticklabels(list(SLEEP_STAGE_NAMES.values()))
    ax_hyp.invert_yaxis()
    ax_hyp.set_xlim(time_min[0], time_min[-1])
    ax_hyp.set_title("Sleep Stage Timeline")

    ax_heat = fig.add_subplot(gs[1])
    band_range = band_data.max(axis=0) - band_data.min(axis=0)
    band_norm = (band_data - band_data.min(axis=0)) / (band_range + 1e-10)
    im = ax_heat.imshow(
        band_norm.T,
        aspect="auto",
        cmap="inferno",
        extent=[time_min[0], time_min[-1], -0.5, len(bands) - 0.5],
        origin="lower",
        interpolation="bilinear",
    )
    ax_heat.set_yticks(range(len(bands)))
    ax_heat.set_yticklabels(bands)
    ax_heat.set_title("Normalized EEG Band Activity")
    plt.colorbar(im, ax=ax_heat, label="Normalized Power", shrink=0.8)

    if proba_df is not None:
        ax_proba = fig.add_subplot(gs[2])
        proba_time = proba_df["time_sec"].to_numpy() / 60.0
        for stage_id, name in SLEEP_STAGE_NAMES.items():
            if name in proba_df.columns:
                ax_proba.fill_between(proba_time, proba_df[name].to_numpy(), alpha=0.4, color=STAGE_COLORS[stage_id], label=name)
        ax_proba.set_ylabel("Probability")
        ax_proba.set_ylim(0, 1)
        ax_proba.set_xlim(time_min[0], time_min[-1])
        ax_proba.set_title("Prediction Probabilities")
        ax_proba.legend(loc="upper right", ncol=5, fontsize=8)

    ax_dom = fig.add_subplot(gs[3])
    band_freqs = [2.25, 6.0, 10.0, 14.0, 23.0]
    dominant_freq = np.array([band_freqs[np.argmax(row)] for row in band_data])
    for stage_id, color in STAGE_COLORS.items():
        mask = stages == stage_id
        if np.any(mask):
            ax_dom.scatter(time_min[mask], dominant_freq[mask], c=color, s=3, alpha=0.5, label=SLEEP_STAGE_NAMES[stage_id], rasterized=True)
    ax_dom.set_ylabel("Dominant\nFrequency (Hz)")
    ax_dom.set_xlabel("Time (minutes)")
    ax_dom.set_xlim(time_min[0], time_min[-1])
    ax_dom.set_title("Dominant EEG Frequency")

    fig.savefig(os.path.join(output_dir, "tvb_brain_state_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TVB-style plots from a TVB export directory")
    parser.add_argument("--export-dir", default=os.path.join("outputs", "tvb_export"))
    parser.add_argument("--output-dir", default=os.path.join("outputs", "tvb_plots"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_timeline(args.export_dir, args.output_dir)
    plot_connectivity_matrices(args.export_dir, args.output_dir)
    print(f"Saved TVB plots to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
