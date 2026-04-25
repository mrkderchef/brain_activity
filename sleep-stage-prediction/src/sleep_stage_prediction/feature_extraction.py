"""Feature extraction for sleep stage classification."""

from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

FEATURE_NAMES = []
for band_name in FREQ_BANDS:
    FEATURE_NAMES.extend([f"{band_name}_mean", f"{band_name}_std", f"{band_name}_relative"])


def compute_bandpower(
    data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute absolute band power using Welch PSD."""
    if nperseg is None:
        nperseg = min(int(4 * sfreq), data.shape[-1])

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])
    return trapezoid(psd[..., freq_mask], freqs[freq_mask], axis=-1)


def compute_total_power(data: np.ndarray, sfreq: float, nperseg: int | None = None) -> np.ndarray:
    """Compute total power between 0.5 and 30 Hz."""
    return compute_bandpower(data, sfreq, (0.5, 30.0), nperseg=nperseg)


def extract_epoch_features(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract mean, std, and relative power for each frequency band."""
    total_power = compute_total_power(epoch_data, sfreq)
    total_power_mean = np.mean(total_power)

    features = []
    for fmin, fmax in FREQ_BANDS.values():
        band_power = compute_bandpower(epoch_data, sfreq, (fmin, fmax))
        features.append(np.mean(band_power))
        features.append(np.std(band_power))
        rel_power = np.mean(band_power) / total_power_mean if total_power_mean > 0 else 0.0
        features.append(rel_power)

    return np.array(features, dtype=np.float64)


def extract_features_from_raw(raw: mne.io.Raw, epoch_duration: float = 30.0) -> np.ndarray:
    """Split a raw recording into epochs and extract band-power features."""
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    n_samples_per_epoch = int(epoch_duration * sfreq)
    n_epochs = data.shape[1] // n_samples_per_epoch

    features = []
    for epoch_idx in range(n_epochs):
        start = epoch_idx * n_samples_per_epoch
        end = start + n_samples_per_epoch
        features.append(extract_epoch_features(data[:, start:end], sfreq))

    return np.array(features)


def generate_synthetic_features(
    labels: np.ndarray,
    n_channels: int = 32,
    rng_seed: int = 42,
) -> np.ndarray:
    """Generate synthetic features for demo runs when raw EEG is unavailable."""
    del n_channels
    rng = np.random.RandomState(rng_seed)

    stage_profiles = {
        0: [0.15, 0.10, 0.35, 0.15, 0.25],
        1: [0.25, 0.30, 0.15, 0.15, 0.15],
        2: [0.30, 0.15, 0.10, 0.30, 0.15],
        3: [0.55, 0.15, 0.05, 0.10, 0.15],
        4: [0.15, 0.25, 0.10, 0.15, 0.35],
    }

    features = []
    for label in labels:
        profile = np.array(stage_profiles.get(label, stage_profiles[0]))
        total_power = rng.uniform(50, 200)

        row = []
        for idx, _band_name in enumerate(FREQ_BANDS):
            mean_bp = max(profile[idx] * total_power + rng.normal(0, 5), 0.01)
            std_bp = mean_bp * rng.uniform(0.1, 0.4)
            rel_bp = profile[idx] + rng.normal(0, 0.03)
            row.extend([mean_bp, std_bp, rel_bp])
        features.append(row)

    return np.array(features, dtype=np.float64)
