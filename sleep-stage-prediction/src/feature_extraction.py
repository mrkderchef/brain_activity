"""
feature_extraction.py – Extraktion von EEG-Frequenzband-Features für die Schlafphasen-Klassifikation.

Frequenzbänder:
    Delta  (0.5 – 4 Hz)   → dominant in N3 (Tiefschlaf)
    Theta  (4   – 8 Hz)   → prominent in N1
    Alpha  (8   – 12 Hz)  → Wachzustand, nimmt beim Einschlafen ab
    Sigma  (12  – 16 Hz)  → Schlafspindeln, charakteristisch für N2
    Beta   (16  – 30 Hz)  → Wachheit, Alertness
"""

import numpy as np
import mne
from scipy.signal import welch
from scipy.integrate import trapezoid


# ---------------------------------------------------------------------------
# Frequenzband-Definitionen
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

FEATURE_NAMES = []
for band in FREQ_BANDS:
    FEATURE_NAMES.extend([f"{band}_mean", f"{band}_std", f"{band}_relative"])


# ---------------------------------------------------------------------------
# Bandpower-Berechnung
# ---------------------------------------------------------------------------

def compute_bandpower(data: np.ndarray, sfreq: float, band: tuple[float, float],
                      nperseg: int = None) -> float:
    """Berechnet die absolute Bandleistung für ein Frequenzband mittels Welch-PSD."""
    if nperseg is None:
        nperseg = min(int(4 * sfreq), data.shape[-1])  # 4-Sekunden-Fenster

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])

    # Integration über das Frequenzband (Trapezregel)
    band_power = trapezoid(psd[..., freq_mask], freqs[freq_mask], axis=-1)
    return band_power


def compute_total_power(data: np.ndarray, sfreq: float, nperseg: int = None) -> float:
    """Berechnet die Gesamtleistung über 0.5–30 Hz."""
    return compute_bandpower(data, sfreq, (0.5, 30.0), nperseg)


# ---------------------------------------------------------------------------
# Feature-Extraktion pro Epoche
# ---------------------------------------------------------------------------

def extract_epoch_features(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extrahiert Frequenzband-Features aus einer einzelnen 30-Sekunden-Epoche.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_samples)
    sfreq : float, Sampling-Frequenz in Hz

    Returns
    -------
    features : ndarray, shape (n_features,)
        Pro Band: Mean-Power, Std-Power (über Kanäle), Relative-Power
    """
    total_power = compute_total_power(epoch_data, sfreq)
    total_power_mean = np.mean(total_power)

    features = []
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        bp = compute_bandpower(epoch_data, sfreq, (fmin, fmax))  # shape: (n_channels,)
        features.append(np.mean(bp))                              # Mean über Kanäle
        features.append(np.std(bp))                                # Std über Kanäle
        # Relative Bandleistung (% der Gesamtleistung)
        rel = np.mean(bp) / total_power_mean if total_power_mean > 0 else 0.0
        features.append(rel)

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature-Extraktion über gesamte Aufnahme
# ---------------------------------------------------------------------------

def extract_features_from_raw(raw: mne.io.Raw, epoch_duration: float = 30.0) -> np.ndarray:
    """
    Segmentiert die Raw-EEG-Daten in Epochen und extrahiert Features.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_features)
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # (n_channels, n_samples)
    n_samples_per_epoch = int(epoch_duration * sfreq)
    n_epochs = data.shape[1] // n_samples_per_epoch

    features_list = []
    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        end = start + n_samples_per_epoch
        epoch_data = data[:, start:end]
        feats = extract_epoch_features(epoch_data, sfreq)
        features_list.append(feats)

    return np.array(features_list)


# ---------------------------------------------------------------------------
# Synthetische Demo-Features (wenn keine echten EEG-Daten vorhanden)
# ---------------------------------------------------------------------------

def generate_synthetic_features(labels: np.ndarray, n_channels: int = 32,
                                rng_seed: int = 42) -> np.ndarray:
    """
    Erzeugt realistische synthetische Frequenzband-Features basierend auf
    bekannten neurophysiologischen Mustern der Schlafphasen.

    Dies ermöglicht das Testen der Pipeline ohne echte EEG-Rohdaten.
    """
    rng = np.random.RandomState(rng_seed)
    n_epochs = len(labels)

    # Typische relative Bandleistung pro Schlafphase (Literaturwerte)
    # Format: [delta, theta, alpha, sigma, beta]
    stage_profiles = {
        0: [0.15, 0.10, 0.35, 0.15, 0.25],  # Wake: hohe Alpha + Beta
        1: [0.25, 0.30, 0.15, 0.15, 0.15],   # N1: Theta dominant
        2: [0.30, 0.15, 0.10, 0.30, 0.15],   # N2: Sigma (Spindeln) + Delta
        3: [0.55, 0.15, 0.05, 0.10, 0.15],   # N3: Delta dominant
        4: [0.15, 0.25, 0.10, 0.15, 0.35],   # REM: gemischte Aktivität
    }

    features_list = []
    for label in labels:
        profile = np.array(stage_profiles.get(label, stage_profiles[0]))
        total_power_sim = rng.uniform(50, 200)  # µV²

        feats = []
        for j, band_name in enumerate(FREQ_BANDS):
            mean_bp = profile[j] * total_power_sim + rng.normal(0, 5)
            mean_bp = max(mean_bp, 0.01)
            std_bp = mean_bp * rng.uniform(0.1, 0.4)
            rel_bp = profile[j] + rng.normal(0, 0.03)
            feats.extend([mean_bp, std_bp, rel_bp])

        features_list.append(feats)

    return np.array(features_list, dtype=np.float64)
