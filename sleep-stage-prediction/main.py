"""
main.py – Hauptskript für das Sleep Stage Prediction Projekt.

Dieses Skript orchestriert:
1. Daten laden (Labels + EEG)
2. Feature-Extraktion (Frequenzband-Power)
3. ML-Modell trainieren & evaluieren
4. Visualisierungen erstellen
5. TVB-Export vorbereiten

Verwendung:
    python main.py                          # Automatisch: echte Daten oder Demo
    python main.py --demo                   # Erzwingt synthetische Demo-Daten
    python main.py --bids-root /pfad/zu/ds003768-master

Wenn keine echten EEG-Binärdaten vorhanden sind (git-annex nicht gefetcht),
wird automatisch der Demo-Modus mit synthetischen Features aktiviert.
"""

import argparse
import os
import sys

import numpy as np

# Projektverzeichnis zum Pfad hinzufügen
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import (
    load_sleep_stages,
    find_eeg_files,
    load_eeg_raw,
    check_data_availability,
)
from src.feature_extraction import (
    extract_features_from_raw,
    generate_synthetic_features,
    FREQ_BANDS,
)
from src.train_model import train_and_evaluate, predict_sleep_stage
from src.visualize import create_all_visualizations


def get_default_bids_root():
    """Findet das BIDS-Verzeichnis relativ zum Projektordner."""
    candidates = [
        os.path.join(PROJECT_DIR, "..", "ds003768-master"),
        os.path.join(PROJECT_DIR, "..", "ds003768"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return None


def run_demo_mode(bids_root: str, output_dir: str):
    """
    Demo-Modus: Verwendet echte Schlafstadien-Labels, aber
    synthetische EEG-Features (da Binärdaten nicht vorhanden).
    """
    print("\n" + "=" * 60)
    print(" DEMO-MODUS: Synthetische EEG-Features")
    print(" (Echte EEG-Binärdaten nicht verfügbar)")
    print("=" * 60)
    print("\nUm mit echten Daten zu arbeiten:")
    print("  pip install datalad")
    print("  datalad install https://github.com/OpenNeuroDatasets/ds003768.git")
    print("  datalad get ds003768/sub-*/eeg/*")
    print("=" * 60)

    # Labels laden
    sourcedata_dir = os.path.join(bids_root, "sourcedata")
    print(f"\nLade Schlafstadien-Labels aus: {sourcedata_dir}")
    stages_df = load_sleep_stages(sourcedata_dir)

    print(f"  Gesamtanzahl Epochen: {len(stages_df)}")
    print(f"  Schlafstadien-Verteilung:")
    for stage_int, count in stages_df["stage_int"].value_counts().sort_index().items():
        stage_name = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}.get(stage_int, "?")
        print(f"    {stage_name} (={stage_int}): {count} Epochen")

    y = stages_df["stage_int"].values

    # Synthetische Features erzeugen
    print("\nErzeuge synthetische EEG-Frequenzband-Features ...")
    X = generate_synthetic_features(y)
    print(f"  Feature-Matrix: {X.shape} (Epochen × Features)")

    return X, y


def run_real_mode(bids_root: str, output_dir: str):
    """Modus mit echten EEG-Daten."""
    print("\n" + "=" * 60)
    print(" ECHTDATEN-MODUS: EEG-Feature-Extraktion")
    print("=" * 60)

    sourcedata_dir = os.path.join(bids_root, "sourcedata")
    stages_df = load_sleep_stages(sourcedata_dir)
    eeg_files = find_eeg_files(bids_root)

    all_X = []
    all_y = []

    for rec in eeg_files:
        session = rec["session"]
        subject_id = rec["subject"]
        vhdr_path = rec["vhdr_path"]

        # Labels für diese Session holen
        from src.data_loader import get_labels_for_session
        labels = get_labels_for_session(stages_df, subject_id, session)
        if len(labels) == 0:
            print(f"  [SKIP] {rec['subject_str']} / {session}: Keine Labels")
            continue

        print(f"  Lade {rec['subject_str']} / {session} ...")
        try:
            raw = load_eeg_raw(vhdr_path)
        except Exception as e:
            print(f"  [FEHLER] Konnte EEG nicht laden: {e}")
            continue

        X = extract_features_from_raw(raw)
        n = min(X.shape[0], len(labels))
        all_X.append(X[:n])
        all_y.append(labels[:n])
        print(f"    → {n} Epochen extrahiert")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nGesamte Feature-Matrix: {X.shape}")

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Sleep Stage Prediction from EEG Frequency Bands")
    parser.add_argument("--bids-root", type=str, default=None,
                        help="Pfad zum BIDS-Dataset (ds003768-master)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo-Modus mit synthetischen Features erzwingen")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_DIR, "outputs"),
                        help="Ausgabeverzeichnis")
    args = parser.parse_args()

    bids_root = args.bids_root or get_default_bids_root()
    if bids_root is None:
        print("FEHLER: BIDS-Verzeichnis nicht gefunden.")
        print("Bitte --bids-root angeben oder ds003768-master neben diesem Ordner platzieren.")
        sys.exit(1)

    print(f"BIDS-Root: {bids_root}")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prüfe ob echte Daten vorhanden
    if args.demo:
        use_demo = True
    else:
        availability = check_data_availability(bids_root)
        n_available = len(availability["available"])
        n_missing = len(availability["missing"])
        print(f"Datenverfügbarkeit: {n_available} verfügbar, {n_missing} als Annex-Pointer")
        use_demo = (n_available == 0)

    # Daten laden & Features extrahieren
    if use_demo:
        X, y = run_demo_mode(bids_root, output_dir)
    else:
        X, y = run_real_mode(bids_root, output_dir)

    # NaN/Inf bereinigen
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    # Modell trainieren & evaluieren
    results = train_and_evaluate(X, y, n_splits=5, output_dir=output_dir)

    # Vorhersagen für Visualisierung
    y_pred, y_proba = predict_sleep_stage(results["model"], X)

    # Visualisierungen
    tvb_dir = create_all_visualizations(
        X, y, y_pred, y_proba,
        results["feature_importances"],
        output_dir,
    )

    # Zusammenfassung
    print("\n" + "=" * 60)
    print(" ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"  Accuracy:       {results['accuracy']:.4f}")
    print(f"  Cohen's Kappa:  {results['kappa']:.4f}")
    print(f"  Epochen:        {len(y)}")
    print(f"  Features:       {X.shape[1]} (5 Frequenzbänder × 3 Metriken)")
    print(f"  Modell:         {output_dir}/sleep_stage_model.joblib")
    print(f"  Plots:          {output_dir}/*.png")
    print(f"  TVB-Export:     {tvb_dir}/")
    print("=" * 60)
    print("\n  Frequenzbänder und ihre Schlafphasen-Assoziationen:")
    print("    Delta  (0.5-4 Hz)  → N3 (Tiefschlaf)")
    print("    Theta  (4-8 Hz)    → N1 (Leichtschlaf)")
    print("    Alpha  (8-12 Hz)   → Wake (Entspannte Wachheit)")
    print("    Sigma  (12-16 Hz)  → N2 (Schlafspindeln)")
    print("    Beta   (16-30 Hz)  → Wake (Alertness)")
    print()
    print("  TVB-Visualisierung:")
    print(f"    Lade die Dateien aus {tvb_dir}/ in TVB Web GUI")
    print("    oder verwende tvb-library für programmatische Visualisierung.")


if __name__ == "__main__":
    main()
