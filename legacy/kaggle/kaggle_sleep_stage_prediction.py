# %% [markdown]
# # 🧠 Sleep Stage Prediction from EEG Frequency Bands
#
# **Dataset:** [ds003768 – Simultaneous EEG and fMRI signals during sleep](https://openneuro.org/datasets/ds003768)
#
# **Ziel:** Vorhersage der Schlafphase (Wake, N1, N2, N3) anhand der EEG-Frequenzband-Power
#
# | Frequenzband | Hz-Bereich | Dominante Phase |
# |---|---|---|
# | Delta | 0.5–4 Hz | N3 (Tiefschlaf) |
# | Theta | 4–8 Hz | N1 (Leichtschlaf) |
# | Alpha | 8–12 Hz | Wake |
# | Sigma | 12–16 Hz | N2 (Schlafspindeln) |
# | Beta | 16–30 Hz | Wake (Alertness) |
#
# **Pipeline:**
# 1. Daten von OpenNeuro herunterladen
# 2. EEG → Frequenzband-Features (30s Epochen)
# 3. Random Forest Classifier trainieren
# 4. Visualisierungen + TVB-Export

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
!pip install -q mne openneuro-py

import os
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import welch
from scipy.integrate import trapezoid

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, cohen_kappa_score
)
import joblib
import mne

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

print('Setup complete ✓')

# %% [markdown]
# ## 2. Daten laden
#
# **Option A (empfohlen):** Vorextrahierte Features als Kaggle Dataset hochladen
# - Lade `X_features.npy` und `y_labels.npy` als Kaggle Dataset hoch
# - Setze `USE_PREEXTRACTED = True`
# - Kein 270 GB Download nötig!
#
# **Option B:** EEG-Rohdaten von OpenNeuro (nur bei schneller Verbindung)

# %%
# ======================== KONFIGURATION ========================
USE_PREEXTRACTED = True   # True = vorextrahierte Features, False = Rohdaten

# Option A: Pfad zum Kaggle Dataset mit Features
FEATURES_DIR = '/kaggle/input/sleep-eeg-features'

# Option B: Nur wenn USE_PREEXTRACTED = False
SUBJECTS = [f'{i:02d}' for i in range(1, 34)]
DATA_DIR = '/kaggle/working/ds003768'

OUTPUT_DIR = '/kaggle/working/outputs'
# ===============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_PREEXTRACTED:
    print('Modus: Vorextrahierte Features laden')
    print(f'Features-Verzeichnis: {FEATURES_DIR}')
else:
    include_patterns = ['sourcedata/*', 'dataset_description.json', 'README', 'CHANGES']
    for sub in SUBJECTS:
        include_patterns.append(f'sub-{sub}/eeg/*')
    print(f'Modus: EEG-Rohdaten von OpenNeuro ({len(SUBJECTS)} Probanden)')

# %%
# Download nur wenn Rohdaten-Modus aktiv
if not USE_PREEXTRACTED:
    import openneuro
    if not os.path.exists(os.path.join(DATA_DIR, 'sourcedata')):
        openneuro.download(dataset='ds003768', target_dir=DATA_DIR, include=include_patterns)
        print('Download abgeschlossen ✓')
    else:
        print('Daten bereits vorhanden ✓')
else:
    print('Überspringe Download (vorextrahierte Features werden genutzt) ✓')

# %% [markdown]
# ## 3-5. Feature-Extraktion (oder vorextrahierte Features laden)

# %%
SLEEP_STAGE_MAP = {'W': 0, '1': 1, '2': 2, '3': 3, 'R': 4}
SLEEP_STAGE_NAMES = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
STAGE_COLORS = {0: '#E74C3C', 1: '#F39C12', 2: '#3498DB', 3: '#2C3E50', 4: '#9B59B6'}

FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'sigma': (12.0, 16.0),
    'beta':  (16.0, 30.0),
}

if USE_PREEXTRACTED:
    # ===== OPTION A: Vorextrahierte Features laden =====
    X = np.load(os.path.join(FEATURES_DIR, 'X_features.npy'))
    y = np.load(os.path.join(FEATURES_DIR, 'y_labels.npy'))
    print(f'Features geladen: X={X.shape}, y={y.shape}')

else:
    # ===== OPTION B: EEG-Rohdaten verarbeiten =====
    def load_sleep_stages(sourcedata_dir):
        tsv_files = sorted(glob.glob(os.path.join(sourcedata_dir, 'sub-*-sleep-stage.tsv')))
        frames = []
        for f in tsv_files:
            df = pd.read_csv(f, sep='\t')
            df.columns = df.columns.str.strip()
            df['stage_str'] = df['30-sec_epoch_sleep_stage'].astype(str).str.strip()
            df['stage_int'] = df['stage_str'].map(SLEEP_STAGE_MAP)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True).dropna(subset=['stage_int'])
        combined['stage_int'] = combined['stage_int'].astype(int)
        return combined

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

    def find_eeg_files(bids_root):
        vhdr_files = sorted(glob.glob(os.path.join(bids_root, 'sub-*/eeg/*.vhdr')))
        records = []
        for vhdr in vhdr_files:
            fname = Path(vhdr).stem
            parts = fname.split('_')
            sub = parts[0]
            session = '_'.join(p for p in parts if p.startswith('task-') or p.startswith('run-'))
            records.append({'subject': int(sub.replace('sub-', '')), 'subject_str': sub,
                            'session': session, 'vhdr_path': vhdr})
        return records

    def get_labels_for_session(stages_df, subject_id, session):
        mask = (stages_df['subject'] == subject_id) & (stages_df['session'] == session)
        return stages_df[mask].sort_values('epoch_start_time_sec')['stage_int'].values

    stages_df = load_sleep_stages(os.path.join(DATA_DIR, 'sourcedata'))
    eeg_files = find_eeg_files(DATA_DIR)
    print(f'Gefundene EEG-Aufnahmen: {len(eeg_files)}')

    import gc
    all_X, all_y, skipped = [], [], []
    for idx, rec in enumerate(eeg_files):
        labels = get_labels_for_session(stages_df, rec['subject'], rec['session'])
        if len(labels) == 0:
            skipped.append(f"{rec['subject_str']}/{rec['session']}: keine Labels")
            continue
        print(f"[{idx+1}/{len(eeg_files)}] {rec['subject_str']} / {rec['session']} ...", end=' ', flush=True)
        try:
            raw = mne.io.read_raw_brainvision(rec['vhdr_path'], preload=True, verbose=False)
            X_sub = extract_features_from_raw(raw)
            n = min(X_sub.shape[0], len(labels))
            all_X.append(X_sub[:n]); all_y.append(labels[:n])
            print(f'{n} Epochen ✓')
            del raw, X_sub; gc.collect()
        except Exception as e:
            skipped.append(f"{rec['subject_str']}/{rec['session']}: {e}")
            print(f'FEHLER: {e}')

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

# NaN/Inf bereinigen
mask = np.all(np.isfinite(X), axis=1)
X, y = X[mask], y[mask]

print(f'\nFeature-Matrix: {X.shape} (Epochen × Features)')
print(f'Klassen: {dict(zip(*np.unique(y, return_counts=True)))}')

# %%
# Features speichern
np.save(os.path.join(OUTPUT_DIR, 'X_features.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
print(f'Features gespeichert: {os.path.getsize(os.path.join(OUTPUT_DIR, "X_features.npy")) / 1024:.1f} KB')

# %% [markdown]
# ## 6. Daten-Exploration

# %%
# Klassen-Verteilung
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar Chart
unique, counts = np.unique(y, return_counts=True)
names = [SLEEP_STAGE_NAMES[u] for u in unique]
colors = [STAGE_COLORS[u] for u in unique]
axes[0].bar(names, counts, color=colors)
axes[0].set_ylabel('Anzahl Epochen')
axes[0].set_title('Schlafstadien-Verteilung')
for i, (n, c) in enumerate(zip(names, counts)):
    axes[0].text(i, c + 20, str(c), ha='center', fontweight='bold')

# Pie Chart
axes[1].pie(counts, labels=names, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Anteil pro Schlafstadium')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# Feature-Statistiken
band_names = list(FREQ_BANDS.keys())
feat_labels = []
for b in band_names:
    feat_labels.extend([f'{b}_mean', f'{b}_std', f'{b}_relative'])

feat_df = pd.DataFrame(X, columns=feat_labels)
feat_df['stage'] = [SLEEP_STAGE_NAMES[yi] for yi in y]
feat_df.groupby('stage')[feat_labels].mean().round(4)

# %% [markdown]
# ## 7. Modell-Training (Random Forest)

# %%
# Pipeline: Standardisierung + Random Forest
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ))
])

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

acc = accuracy_score(y, y_pred_cv)
kappa = cohen_kappa_score(y, y_pred_cv)

target_names = [SLEEP_STAGE_NAMES[i] for i in sorted(np.unique(y))]
print(classification_report(y, y_pred_cv, target_names=target_names))
print(f'Accuracy:      {acc:.4f}')
print(f"Cohen's Kappa: {kappa:.4f}")

# %%
# Finales Modell auf allen Daten trainieren
model.fit(X, y)
y_pred, y_proba = model.predict(X), model.predict_proba(X)

# Feature Importances
feature_importances = model.named_steps['classifier'].feature_importances_

# Modell speichern
joblib.dump(model, os.path.join(OUTPUT_DIR, 'sleep_stage_model.joblib'))
print('Modell gespeichert ✓')

# Metriken als JSON
metrics = {
    'accuracy': float(acc),
    'cohen_kappa': float(kappa),
    'n_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
    'class_distribution': {SLEEP_STAGE_NAMES.get(k, str(k)): int(v)
                           for k, v in zip(*np.unique(y, return_counts=True))},
}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'Metriken: Accuracy={acc:.4f}, Kappa={kappa:.4f}')

# %% [markdown]
# ## 8. Visualisierungen

# %%
# 8a. Confusion Matrix
labels_cm = sorted(set(y) | set(y_pred_cv))
names_cm = [SLEEP_STAGE_NAMES.get(l, str(l)) for l in labels_cm]
cm = confusion_matrix(y, y_pred_cv, labels=labels_cm)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names_cm, yticklabels=names_cm, ax=axes[0])
axes[0].set_xlabel('Vorhergesagt'); axes[0].set_ylabel('Tatsächlich')
axes[0].set_title('Confusion Matrix (absolut)')

sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=names_cm, yticklabels=names_cm, ax=axes[1])
axes[1].set_xlabel('Vorhergesagt'); axes[1].set_ylabel('Tatsächlich')
axes[1].set_title('Confusion Matrix (% pro Klasse)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# 8b. Frequenzband-Power pro Schlafstadium (KERN-PLOT)
rel_indices = [i * 3 + 2 for i in range(len(FREQ_BANDS))]

records = []
for i, label in enumerate(y):
    for j, band in enumerate(FREQ_BANDS):
        records.append({
            'Schlafstadium': SLEEP_STAGE_NAMES[label],
            'Frequenzband': band.capitalize(),
            'Relative Power': X[i, rel_indices[j]],
        })

plot_df = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(12, 7))
stage_order = [SLEEP_STAGE_NAMES[k] for k in sorted(SLEEP_STAGE_NAMES.keys())
               if SLEEP_STAGE_NAMES[k] in plot_df['Schlafstadium'].values]
palette = [STAGE_COLORS[k] for k in sorted(SLEEP_STAGE_NAMES.keys())
           if SLEEP_STAGE_NAMES[k] in plot_df['Schlafstadium'].values]

sns.boxplot(data=plot_df, x='Frequenzband', y='Relative Power',
            hue='Schlafstadium', hue_order=stage_order, palette=palette, ax=ax)
ax.set_title('EEG-Frequenzband-Power pro Schlafstadium', fontsize=14)
ax.legend(title='Schlafstadium', loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'frequency_bands_by_stage.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# 8c. Feature Importance
idx_fi = np.argsort(feature_importances)[::-1]
band_colors = ['#E74C3C', '#F39C12', '#3498DB', '#2C3E50', '#9B59B6']
colors_fi = [band_colors[i // 3 % len(band_colors)] for i in idx_fi]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(feature_importances)), feature_importances[idx_fi], color=colors_fi)
ax.set_xticks(range(len(feature_importances)))
ax.set_xticklabels([feat_labels[i] for i in idx_fi], rotation=45, ha='right')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance – Welche Frequenzbänder sind am wichtigsten?', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# 8d. Radar-Chart: Frequenzprofil pro Schlafstadium
mean_indices = [i * 3 for i in range(len(FREQ_BANDS))]
stages_present = sorted(set(y))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2 * np.pi, len(FREQ_BANDS), endpoint=False).tolist()
angles += angles[:1]

for stage in stages_present:
    mask_s = y == stage
    means = [np.mean(X[mask_s, idx]) for idx in mean_indices]
    total = sum(means) if sum(means) > 0 else 1
    means_norm = [m / total for m in means] + [means[0] / total]
    ax.plot(angles, means_norm, 'o-', label=SLEEP_STAGE_NAMES[stage],
            color=STAGE_COLORS[stage], linewidth=2)
    ax.fill(angles, means_norm, alpha=0.1, color=STAGE_COLORS[stage])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([b.capitalize() for b in FREQ_BANDS])
ax.set_title('Frequenzprofil pro Schlafstadium', y=1.08, fontsize=13)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'psd_radar_by_stage.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# 8e. Hypnogramm (erste 200 Epochen)
max_epochs = min(200, len(y))
time_min = np.arange(max_epochs) * 0.5

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
for ax, data, title in [(axes[0], y[:max_epochs], 'Tatsächlich (Scored)'),
                         (axes[1], y_pred_cv[:max_epochs], 'Vorhergesagt (ML)')]:
    ax.step(time_min, data, where='post', color='black', linewidth=0.8)
    for stage_id, color in STAGE_COLORS.items():
        m = data == stage_id
        if np.any(m):
            ax.fill_between(time_min, -0.5, max(data) + 0.5,
                            where=m, color=color, alpha=0.15, step='post')
    ax.set_yticks(list(SLEEP_STAGE_NAMES.keys()))
    ax.set_yticklabels(list(SLEEP_STAGE_NAMES.values()))
    ax.set_title(title)
    ax.invert_yaxis()

axes[1].set_xlabel('Zeit (Minuten)')
patches = [mpatches.Patch(color=STAGE_COLORS[k], label=v) for k, v in SLEEP_STAGE_NAMES.items()]
fig.legend(handles=patches, loc='upper right', ncol=5, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'hypnogram.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. TVB-Export
#
# Exportiert die Daten für **The Virtual Brain (TVB)**:
# - Frequenzband-Zeitreihen
# - Vorhersage-Wahrscheinlichkeiten
# - Konnektivitätsmatrizen pro Schlafphase

# %%
tvb_dir = os.path.join(OUTPUT_DIR, 'tvb_export')
os.makedirs(tvb_dir, exist_ok=True)

# 1. Frequenzband-Zeitreihen
band_timeseries = X[:, mean_indices]
ts_df = pd.DataFrame(band_timeseries, columns=[b.capitalize() for b in FREQ_BANDS])
ts_df['time_sec'] = np.arange(len(X)) * 30.0
ts_df['sleep_stage'] = y_pred
ts_df.to_csv(os.path.join(tvb_dir, 'band_power_timeseries.csv'), index=False)

# 2. Vorhersage-Wahrscheinlichkeiten
n_classes = y_proba.shape[1]
proba_cols = [SLEEP_STAGE_NAMES.get(i, f'Stage-{i}') for i in range(n_classes)]
proba_df = pd.DataFrame(y_proba, columns=proba_cols)
proba_df['time_sec'] = np.arange(len(y_proba)) * 30.0
proba_df.to_csv(os.path.join(tvb_dir, 'prediction_probabilities.csv'), index=False)

# 3. Konnektivitätsmatrizen pro Schlafphase
for stage in sorted(set(y_pred)):
    m = y_pred == stage
    if np.sum(m) < 5:
        continue
    corr = np.corrcoef(X[m][:, mean_indices].T)
    corr_df = pd.DataFrame(corr,
                           index=[b.capitalize() for b in FREQ_BANDS],
                           columns=[b.capitalize() for b in FREQ_BANDS])
    stage_name = SLEEP_STAGE_NAMES.get(stage, str(stage))
    corr_df.to_csv(os.path.join(tvb_dir, f'connectivity_{stage_name}.csv'))

# 4. Manifest
manifest = {
    'description': 'Sleep stage prediction data for TVB visualization',
    'dataset': 'ds003768 - Simultaneous EEG and fMRI during sleep',
    'epoch_duration_sec': 30.0,
    'frequency_bands': {k: list(v) for k, v in FREQ_BANDS.items()},
    'sleep_stages': {str(k): v for k, v in SLEEP_STAGE_NAMES.items()},
}
with open(os.path.join(tvb_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print('TVB-Export abgeschlossen ✓')
for f_name in os.listdir(tvb_dir):
    size = os.path.getsize(os.path.join(tvb_dir, f_name)) / 1024
    print(f'  {f_name}: {size:.1f} KB')

# %% [markdown]
# ## 10. TVB Brain State Visualization

# %%
# TVB-Style Brain State Timeline
ts = pd.read_csv(os.path.join(tvb_dir, 'band_power_timeseries.csv'))
proba = pd.read_csv(os.path.join(tvb_dir, 'prediction_probabilities.csv'))

time_min_all = ts['time_sec'].values / 60.0
bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
band_data = ts[bands].values
stages_all = ts['sleep_stage'].values

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 3, 2, 1], hspace=0.3)

# Hypnogramm
ax1 = fig.add_subplot(gs[0])
ax1.step(time_min_all, stages_all, where='post', color='black', linewidth=0.8)
for sid, col in STAGE_COLORS.items():
    m = stages_all == sid
    if np.any(m):
        ax1.fill_between(time_min_all, -0.5, max(stages_all)+0.5,
                         where=m, color=col, alpha=0.15, step='post')
ax1.set_yticks(list(SLEEP_STAGE_NAMES.keys()))
ax1.set_yticklabels(list(SLEEP_STAGE_NAMES.values()))
ax1.invert_yaxis()
ax1.set_title('Schlafstadien-Verlauf', fontsize=12, fontweight='bold')
ax1.set_xlim(time_min_all[0], time_min_all[-1])

# Frequenzband-Heatmap
ax2 = fig.add_subplot(gs[1])
band_range = band_data.max(axis=0) - band_data.min(axis=0)
band_norm = (band_data - band_data.min(axis=0)) / (band_range + 1e-10)
im = ax2.imshow(band_norm.T, aspect='auto', cmap='inferno',
                extent=[time_min_all[0], time_min_all[-1], -0.5, len(bands)-0.5],
                origin='lower', interpolation='bilinear')
ax2.set_yticks(range(len(bands))); ax2.set_yticklabels(bands)
ax2.set_title('EEG-Frequenzband-Aktivität (normalisiert)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Normalisierte Power', shrink=0.8)

# Vorhersage-Wahrscheinlichkeiten
ax3 = fig.add_subplot(gs[2])
proba_time = proba['time_sec'].values / 60.0
for sid, name in SLEEP_STAGE_NAMES.items():
    if name in proba.columns:
        ax3.fill_between(proba_time, proba[name].values, alpha=0.4,
                         color=STAGE_COLORS[sid], label=name)
ax3.set_ylabel('P(Schlafstadium)'); ax3.set_ylim(0, 1)
ax3.set_title('Vorhersage-Wahrscheinlichkeiten', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', ncol=5, fontsize=8)
ax3.set_xlim(time_min_all[0], time_min_all[-1])

# Dominante Frequenz
ax4 = fig.add_subplot(gs[3])
band_freqs = [2.25, 6.0, 10.0, 14.0, 23.0]
dom_freq = np.array([band_freqs[np.argmax(band_data[i])] for i in range(len(band_data))])
for sid, col in STAGE_COLORS.items():
    m = stages_all == sid
    if np.any(m):
        ax4.scatter(time_min_all[m], dom_freq[m], c=col, s=3, alpha=0.5,
                    label=SLEEP_STAGE_NAMES[sid], rasterized=True)
ax4.set_ylabel('Dominante\nFrequenz (Hz)'); ax4.set_xlabel('Zeit (Minuten)')
ax4.set_title('Dominante EEG-Frequenz', fontsize=12, fontweight='bold')
ax4.set_xlim(time_min_all[0], time_min_all[-1])

plt.suptitle('TVB Brain State Visualization – Sleep Stage Dynamics',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUTPUT_DIR, 'tvb_brain_state_timeline.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# Konnektivitätsmatrizen pro Schlafstadium
conn_files = sorted(glob.glob(os.path.join(tvb_dir, 'connectivity_*.csv')))
n_conn = len(conn_files)
if n_conn > 0:
    fig, axes = plt.subplots(1, n_conn, figsize=(5 * n_conn, 4.5))
    if n_conn == 1: axes = [axes]
    for ax, conn_file in zip(axes, conn_files):
        stage_name = os.path.basename(conn_file).replace('connectivity_', '').replace('.csv', '')
        df = pd.read_csv(conn_file, index_col=0)
        im = ax.imshow(df.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
        ax.set_title(stage_name, fontsize=12, fontweight='bold')
    fig.suptitle('Frequenzband-Konnektivität pro Schlafstadium', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes, label='Korrelation', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tvb_connectivity_matrices.png'), dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 11. Zusammenfassung

# %%
print('=' * 60)
print(' ZUSAMMENFASSUNG')
print('=' * 60)
print(f'  Probanden:      {len(SUBJECTS)}')
print(f'  Epochen:        {len(y)}')
print(f'  Features:       {X.shape[1]} (5 Frequenzbänder × 3 Metriken)')
print(f'  Accuracy:       {acc:.4f}')
print(f"  Cohen's Kappa:  {kappa:.4f}")
print(f'  Modell:         Random Forest (300 Trees)')
print('=' * 60)
print()
print('  Frequenzbänder → Schlafphasen:')
print('    Delta  (0.5-4 Hz)  → N3 (Tiefschlaf)')
print('    Theta  (4-8 Hz)    → N1 (Leichtschlaf)')
print('    Alpha  (8-12 Hz)   → Wake (Entspannte Wachheit)')
print('    Sigma  (12-16 Hz)  → N2 (Schlafspindeln)')
print('    Beta   (16-30 Hz)  → Wake (Alertness)')
print()
print('  Output-Dateien:')
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f_name)
    if os.path.isfile(p):
        size = os.path.getsize(p) / 1024
        print(f'    {f_name}: {size:.1f} KB')
print(f'    tvb_export/: {len(os.listdir(tvb_dir))} Dateien')
