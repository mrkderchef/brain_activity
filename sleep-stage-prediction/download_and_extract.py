"""
download_and_extract.py – Lädt das ds003768 Dataset von OpenNeuro herunter
und extrahiert EEG-Frequenzband-Features für die Schlafphasen-Klassifikation.

Features werden als X_features.npy und y_labels.npy gespeichert.
Diese können dann auf Kaggle als Dataset hochgeladen werden.

Verwendung:
    python download_and_extract.py                         # Alle 33 Probanden
    python download_and_extract.py --subjects 1 5          # Nur sub-01 bis sub-05
    python download_and_extract.py --skip-download         # Nur Feature-Extraktion (Daten schon da)
    python download_and_extract.py --resume                # Download fortsetzen nach Unterbrechung

Pause/Resume:
    - Ctrl+C drückt man zum Pausieren
    - Danach einfach "python download_and_extract.py --resume" zum Weitermachen
    - Bereits heruntergeladene Dateien werden übersprungen
    - Bereits extrahierte Features werden aus dem Checkpoint geladen

Ergebnis:
    outputs/X_features.npy   – Feature-Matrix (n_epochs × 15)
    outputs/y_labels.npy     – Schlafstadien-Labels
    outputs/checkpoint.npz   – Checkpoint für Resume-Fähigkeit
"""

import argparse
import gc
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'sigma': (12.0, 16.0),
    'beta':  (16.0, 30.0),
}

SLEEP_STAGE_MAP = {'W': 0, '1': 1, '2': 2, '3': 3, 'R': 4}
SLEEP_STAGE_NAMES = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}


# ---------------------------------------------------------------------------
# Graceful Shutdown (Ctrl+C → Checkpoint speichern)
# ---------------------------------------------------------------------------

_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print('\n\n⚠️  Zweites Ctrl+C – sofortiger Abbruch!')
        sys.exit(1)
    _shutdown_requested = True
    print('\n\n⏸️  Pause angefordert! Speichere Checkpoint nach aktuellem Probanden ...')
    print('   (Nochmal Ctrl+C für sofortigen Abbruch)')

signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Feature-Extraktion
# ---------------------------------------------------------------------------

def compute_bandpower(data, sfreq, band, nperseg=None):
    if nperseg is None:
        nperseg = min(int(4 * sfreq), data.shape[-1])
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])
    return trapezoid(psd[..., freq_mask], freqs[freq_mask], axis=-1)


def extract_epoch_features(epoch_data, sfreq):
    total_power = compute_bandpower(epoch_data, sfreq, (0.5, 30.0))
    total_power_mean = np.mean(total_power)
    features = []
    for fmin, fmax in FREQ_BANDS.values():
        bp = compute_bandpower(epoch_data, sfreq, (fmin, fmax))
        features.append(np.mean(bp))
        features.append(np.std(bp))
        rel = np.mean(bp) / total_power_mean if total_power_mean > 0 else 0.0
        features.append(rel)
    return np.array(features, dtype=np.float64)


def extract_features_from_raw(raw, epoch_duration=30.0):
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samples_per_epoch = int(epoch_duration * sfreq)
    n_epochs = data.shape[1] // n_samples_per_epoch
    features_list = []
    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        features_list.append(extract_epoch_features(data[:, start:start + n_samples_per_epoch], sfreq))
    return np.array(features_list)


# ---------------------------------------------------------------------------
# Daten-Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_sleep_stages(sourcedata_dir):
    tsv_files = sorted(glob.glob(os.path.join(sourcedata_dir, 'sub-*-sleep-stage.tsv')))
    if not tsv_files:
        raise FileNotFoundError(f'Keine sleep-stage TSV in {sourcedata_dir}')
    frames = []
    for f in tsv_files:
        df = pd.read_csv(f, sep='\t')
        df.columns = df.columns.str.strip()
        # Subject-ID aus Dateiname extrahieren (nicht alle TSVs haben eine 'subject'-Spalte)
        fname = Path(f).stem  # z.B. "sub-02-sleep-stage"
        sub_id = int(fname.split('-')[1])  # → 2
        df['subject'] = sub_id
        df['stage_str'] = df['30-sec_epoch_sleep_stage'].astype(str).str.strip()
        df['stage_int'] = df['stage_str'].map(SLEEP_STAGE_MAP)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True).dropna(subset=['stage_int'])
    combined['stage_int'] = combined['stage_int'].astype(int)
    return combined


def find_eeg_files(bids_root):
    vhdr_files = sorted(glob.glob(os.path.join(bids_root, 'sub-*/eeg/*.vhdr')))
    records = []
    for vhdr in vhdr_files:
        fname = Path(vhdr).stem
        parts = fname.split('_')
        sub = parts[0]
        session = '_'.join(p for p in parts if p.startswith('task-') or p.startswith('run-'))
        records.append({
            'subject': int(sub.replace('sub-', '')),
            'subject_str': sub,
            'session': session,
            'vhdr_path': vhdr,
        })
    return records


def get_labels_for_session(stages_df, subject_id, session):
    mask = (stages_df['subject'] == subject_id) & (stages_df['session'] == session)
    return stages_df[mask].sort_values('epoch_start_time_sec')['stage_int'].values


# ---------------------------------------------------------------------------
# Checkpoint-Management
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir, all_X, all_y, processed_keys):
    """Speichert aktuellen Fortschritt."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    X = np.concatenate(all_X, axis=0) if all_X else np.empty((0, 15))
    y = np.concatenate(all_y, axis=0) if all_y else np.empty(0, dtype=int)
    np.savez_compressed(
        checkpoint_path,
        X=X, y=y,
        processed_keys=np.array(list(processed_keys)),
    )
    print(f'💾 Checkpoint gespeichert: {checkpoint_path}')
    print(f'   {len(processed_keys)} Aufnahmen, {X.shape[0]} Epochen bisher')
    return checkpoint_path


def load_checkpoint(output_dir):
    """Lädt Checkpoint falls vorhanden."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        processed_keys = set(data['processed_keys'].tolist())
        print(f'📂 Checkpoint geladen: {len(processed_keys)} Aufnahmen, {X.shape[0]} Epochen')
        return X, y, processed_keys
    return None, None, set()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data(data_dir, subjects):
    """Lädt EEG-Daten von OpenNeuro. Bereits vorhandene Dateien werden übersprungen."""
    import openneuro

    include_patterns = ['sourcedata/*', 'dataset_description.json', 'README', 'CHANGES']
    for sub_id in subjects:
        include_patterns.append(f'sub-{sub_id:02d}/eeg/*')

    print(f'\n📥 Starte Download von {len(subjects)} Probanden ...')
    print(f'   Zielverzeichnis: {data_dir}')
    print(f'   Bereits vorhandene Dateien werden übersprungen.')
    print(f'   Ctrl+C zum Pausieren – danach mit --resume weitermachen.\n')

    openneuro.download(
        dataset='ds003768',
        target_dir=data_dir,
        include=include_patterns,
    )
    print('\n✅ Download abgeschlossen!')


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download ds003768 + Feature-Extraktion für Schlafphasen-Klassifikation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python download_and_extract.py                    # Alles (33 Probanden)
  python download_and_extract.py --subjects 1 10    # Sub-01 bis Sub-10
  python download_and_extract.py --resume           # Nach Pause weitermachen
  python download_and_extract.py --skip-download    # Nur Features extrahieren
        """,
    )
    parser.add_argument('--subjects', type=int, nargs=2, default=[1, 33],
                        metavar=('START', 'END'),
                        help='Proband-Range (inklusiv), z.B. --subjects 1 10')
    parser.add_argument('--data-dir', type=str, default='./ds003768',
                        help='Verzeichnis für die heruntergeladenen Daten')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Verzeichnis für Features und Checkpoints')
    parser.add_argument('--skip-download', action='store_true',
                        help='Download überspringen (Daten schon vorhanden)')
    parser.add_argument('--resume', action='store_true',
                        help='Fortsetzen nach Unterbrechung (Checkpoint laden)')
    args = parser.parse_args()

    subjects = list(range(args.subjects[0], args.subjects[1] + 1))
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(' Sleep Stage Prediction – Download & Feature-Extraktion')
    print('=' * 60)
    print(f'  Probanden:       sub-{subjects[0]:02d} bis sub-{subjects[-1]:02d} ({len(subjects)} total)')
    print(f'  Datenverzeichnis: {data_dir}')
    print(f'  Output:           {output_dir}')
    print(f'  Resume-Modus:     {"Ja" if args.resume else "Nein"}')
    print('=' * 60)

    # ---- Schritt 1: Download ----
    if not args.skip_download:
        try:
            download_data(data_dir, subjects)
        except KeyboardInterrupt:
            print('\n\n⏸️  Download pausiert.')
            print(f'   Zum Weitermachen: python {sys.argv[0]} --resume')
            print(f'   Bereits geladene Dateien bleiben erhalten.\n')
            # Weiter zur Feature-Extraktion mit dem was da ist
            pass

    # ---- Schritt 2: Feature-Extraktion ----
    global _shutdown_requested
    _shutdown_requested = False

    print('\n' + '=' * 60)
    print(' Feature-Extraktion')
    print('=' * 60)

    # Checkpoint laden falls Resume
    all_X_list = []
    all_y_list = []
    processed_keys = set()

    if args.resume:
        X_ckpt, y_ckpt, processed_keys = load_checkpoint(output_dir)
        if X_ckpt is not None and X_ckpt.shape[0] > 0:
            all_X_list.append(X_ckpt)
            all_y_list.append(y_ckpt)

    # Labels laden
    sourcedata_dir = os.path.join(data_dir, 'sourcedata')
    if not os.path.exists(sourcedata_dir):
        print(f'❌ sourcedata nicht gefunden: {sourcedata_dir}')
        print('   Bitte erst Download durchführen (ohne --skip-download).')
        sys.exit(1)

    stages_df = load_sleep_stages(sourcedata_dir)
    eeg_files = find_eeg_files(data_dir)
    print(f'Gefundene EEG-Aufnahmen: {len(eeg_files)}')

    # MNE erst hier importieren (langsamer Import)
    import mne

    total = len(eeg_files)
    new_processed = 0
    t_start = time.time()

    for idx, rec in enumerate(eeg_files):
        # Pause-Check
        if _shutdown_requested:
            print(f'\n⏸️  Pause nach {new_processed} neuen Aufnahmen.')
            save_checkpoint(output_dir, all_X_list, all_y_list, processed_keys)
            print(f'\n   Zum Weitermachen: python {sys.argv[0]} --resume --skip-download')
            sys.exit(0)

        key = f"{rec['subject_str']}_{rec['session']}"

        # Schon verarbeitet?
        if key in processed_keys:
            continue

        labels = get_labels_for_session(stages_df, rec['subject'], rec['session'])
        if len(labels) == 0:
            processed_keys.add(key)
            continue

        elapsed = time.time() - t_start
        print(f'[{idx+1}/{total}] {key} ...', end=' ', flush=True)

        try:
            raw = mne.io.read_raw_brainvision(rec['vhdr_path'], preload=True, verbose=False)
            X_sub = extract_features_from_raw(raw)
            n = min(X_sub.shape[0], len(labels))
            all_X_list.append(X_sub[:n])
            all_y_list.append(labels[:n])
            processed_keys.add(key)
            new_processed += 1
            print(f'{n} Epochen ✓  ({elapsed:.0f}s)')
            del raw, X_sub
            gc.collect()
        except Exception as e:
            processed_keys.add(key)
            print(f'FEHLER: {e}')

        # Auto-Checkpoint alle 10 Aufnahmen
        if new_processed > 0 and new_processed % 10 == 0:
            save_checkpoint(output_dir, all_X_list, all_y_list, processed_keys)

    # ---- Schritt 3: Ergebnis speichern ----
    if not all_X_list:
        print('\n❌ Keine Features extrahiert! Sind EEG-Dateien vorhanden?')
        sys.exit(1)

    X = np.concatenate(all_X_list, axis=0)
    y = np.concatenate(all_y_list, axis=0)

    # NaN/Inf bereinigen
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    # Speichern
    x_path = os.path.join(output_dir, 'X_features.npy')
    y_path = os.path.join(output_dir, 'y_labels.npy')
    np.save(x_path, X)
    np.save(y_path, y)

    # Auch Checkpoint aktualisieren
    save_checkpoint(output_dir, [X], [y], processed_keys)

    elapsed_total = time.time() - t_start

    print('\n' + '=' * 60)
    print(' FERTIG!')
    print('=' * 60)
    print(f'  Feature-Matrix:  {X.shape} (Epochen × Features)')
    print(f'  Klassen:         {dict(zip(*np.unique(y, return_counts=True)))}')
    print(f'  X gespeichert:   {x_path} ({os.path.getsize(x_path) / 1024:.1f} KB)')
    print(f'  y gespeichert:   {y_path} ({os.path.getsize(y_path) / 1024:.1f} KB)')
    print(f'  Dauer:           {elapsed_total / 60:.1f} Minuten')
    print('=' * 60)
    print()
    print('  Nächster Schritt:')
    print('  1. Lade X_features.npy und y_labels.npy als Kaggle Dataset hoch')
    print('  2. Importiere kaggle_sleep_stage_prediction.py als Kaggle Notebook')
    print('  3. Setze USE_PREEXTRACTED = True und FEATURES_DIR auf dein Dataset')
    print('  4. Run All !')


if __name__ == '__main__':
    main()
