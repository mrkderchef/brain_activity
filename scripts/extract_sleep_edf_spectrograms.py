"""Extract log-spectrogram epoch tensors from Sleep-EDF PSG/Hypnogram pairs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.signal import spectrogram

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.external_bids_sleep import SLEEP_STAGE_NAMES, normalize_stage_label


MOVEMENT_OR_UNKNOWN = -1


@dataclass(frozen=True)
class EdfHeader:
    path: Path
    header_bytes: int
    n_records: int
    record_duration: float
    labels: list[str]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    samples_per_record: np.ndarray

    @property
    def sfreqs(self) -> np.ndarray:
        return self.samples_per_record.astype(float) / self.record_duration

    @property
    def record_samples_total(self) -> int:
        return int(self.samples_per_record.sum())


def parse_ascii(value: bytes) -> str:
    return value.decode("latin-1", errors="ignore").strip()


def parse_float(value: str, default: float = np.nan) -> float:
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except ValueError:
        return default


def read_edf_header(path: Path) -> EdfHeader:
    with open(path, "rb") as handle:
        fixed = handle.read(256)
        header_bytes = parse_int(parse_ascii(fixed[184:192]))
        n_records = parse_int(parse_ascii(fixed[236:244]))
        record_duration = parse_float(parse_ascii(fixed[244:252]))
        n_signals = parse_int(parse_ascii(fixed[252:256]))
        signal_header = handle.read(header_bytes - 256)

    offset = 0

    def read_fields(width: int) -> list[str]:
        nonlocal offset
        values = [
            parse_ascii(signal_header[offset + idx * width : offset + (idx + 1) * width])
            for idx in range(n_signals)
        ]
        offset += width * n_signals
        return values

    labels = read_fields(16)
    read_fields(80)  # transducer
    read_fields(8)  # physical dimension
    physical_min = np.asarray([parse_float(value) for value in read_fields(8)], dtype=np.float64)
    physical_max = np.asarray([parse_float(value) for value in read_fields(8)], dtype=np.float64)
    digital_min = np.asarray([parse_float(value) for value in read_fields(8)], dtype=np.float64)
    digital_max = np.asarray([parse_float(value) for value in read_fields(8)], dtype=np.float64)
    read_fields(80)  # prefiltering
    samples_per_record = np.asarray([parse_int(value) for value in read_fields(8)], dtype=np.int64)

    return EdfHeader(
        path=path,
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration=record_duration,
        labels=labels,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        samples_per_record=samples_per_record,
    )


def signal_offsets(header: EdfHeader) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(header.samples_per_record[:-1])]).astype(np.int64)


def read_edf_channels(path: Path, channel_indices: list[int]) -> tuple[np.ndarray, list[str], float, int]:
    header = read_edf_header(path)
    offsets = signal_offsets(header)
    rows = np.memmap(
        path,
        dtype="<i2",
        mode="r",
        offset=header.header_bytes,
        shape=(header.n_records, header.record_samples_total),
    )
    sfreqs = header.sfreqs[channel_indices]
    if not np.allclose(sfreqs, sfreqs[0]):
        raise ValueError(f"Selected channels do not share one sampling rate: {dict(zip(channel_indices, sfreqs))}")

    channels = []
    for channel_idx in channel_indices:
        start = offsets[channel_idx]
        stop = start + header.samples_per_record[channel_idx]
        digital = np.asarray(rows[:, start:stop], dtype=np.float32).reshape(-1)
        scale = (header.physical_max[channel_idx] - header.physical_min[channel_idx]) / (
            header.digital_max[channel_idx] - header.digital_min[channel_idx]
        )
        physical = (digital - header.digital_min[channel_idx]) * scale + header.physical_min[channel_idx]
        channels.append(physical.astype(np.float32))

    return np.stack(channels, axis=0), [header.labels[idx] for idx in channel_indices], float(sfreqs[0]), header.n_records


def read_edf_annotation_text(path: Path) -> str:
    header = read_edf_header(path)
    annotation_indices = [
        idx for idx, label in enumerate(header.labels)
        if "annotation" in label.lower()
    ]
    if not annotation_indices:
        annotation_indices = list(range(len(header.labels)))
    offsets = signal_offsets(header)
    rows = np.memmap(
        path,
        dtype="<i2",
        mode="r",
        offset=header.header_bytes,
        shape=(header.n_records, header.record_samples_total),
    )
    chunks = []
    for annotation_idx in annotation_indices:
        start = offsets[annotation_idx]
        stop = start + header.samples_per_record[annotation_idx]
        chunks.append(np.asarray(rows[:, start:stop], dtype="<i2").tobytes().decode("latin-1", errors="ignore"))
    return "".join(chunks)


def find_sleep_edf_pairs(root: str) -> list[tuple[Path, Path]]:
    psg_paths = sorted(Path(root).rglob("*-PSG.edf"))
    hypnogram_paths = sorted(Path(root).rglob("*-Hypnogram.edf"))
    hypnograms_by_key = {path.name[:6].upper(): path for path in hypnogram_paths}
    pairs = []
    for psg_path in psg_paths:
        key = psg_path.name[:6].upper()
        hypnogram_path = hypnograms_by_key.get(key)
        if hypnogram_path is not None:
            pairs.append((psg_path, hypnogram_path))
    return pairs


def normalize_sleep_edf_description(description: str) -> int:
    text = description.strip().lower()
    if text in {"movement time", "sleep stage ?", "unknown"}:
        return MOVEMENT_OR_UNKNOWN
    return normalize_stage_label(text)


def annotation_epochs(hypnogram_path: Path, epoch_duration: float) -> list[tuple[float, int, str]]:
    annotation_text = read_edf_annotation_text(hypnogram_path)
    rows: list[tuple[float, int, str]] = []
    pattern = re.compile(r"([+-]\d+(?:\.\d*)?)(?:\x15(\d+(?:\.\d*)?))?\x14([^\x00]*)\x00")
    for match in pattern.finditer(annotation_text):
        onset = float(match.group(1))
        duration = float(match.group(2) or 0.0)
        descriptions = [part for part in match.group(3).split("\x14") if part]
        if not descriptions or duration <= 0:
            continue
        description = descriptions[0]
        label = normalize_sleep_edf_description(description)
        n_epochs = int(round(float(duration) / epoch_duration))
        for offset in range(n_epochs):
            rows.append((float(onset) + offset * epoch_duration, int(label), str(description)))
    rows.sort(key=lambda row: row[0])
    return rows


def trim_wake_context(
    rows: list[tuple[float, int, str]],
    wake_context_minutes: float | None,
    epoch_duration: float,
) -> list[tuple[float, int, str]]:
    valid = [(idx, row) for idx, row in enumerate(rows) if row[1] != MOVEMENT_OR_UNKNOWN]
    sleep_positions = [idx for idx, row in valid if row[1] != 0]
    if wake_context_minutes is None or not sleep_positions:
        return rows
    context_epochs = int(round((wake_context_minutes * 60.0) / epoch_duration))
    start = max(0, min(sleep_positions) - context_epochs)
    stop = min(len(rows), max(sleep_positions) + context_epochs + 1)
    return rows[start:stop]


def choose_channels(channel_names: list[str], requested_channels: list[str] | None) -> list[int]:
    if requested_channels:
        missing = [channel for channel in requested_channels if channel not in channel_names]
        if missing:
            raise KeyError(f"Requested channels not found: {missing}; available={channel_names}")
        return [channel_names.index(channel) for channel in requested_channels]

    for idx, channel in enumerate(channel_names):
        if re.search(r"\bEEG\b|EEG", channel, re.IGNORECASE):
            return [idx]
    return [0]


def compute_epoch_spectrogram(
    epoch_data: np.ndarray,
    sfreq: float,
    nperseg: int,
    noverlap: int,
    fmin: float,
    fmax: float,
    target_freq_bins: int | None,
    target_time_bins: int | None,
) -> np.ndarray:
    channel_specs = []
    for channel in epoch_data:
        freqs, _, power = spectrogram(
            channel,
            fs=sfreq,
            window="hann",
            nperseg=min(nperseg, channel.shape[-1]),
            noverlap=min(noverlap, max(0, min(nperseg, channel.shape[-1]) - 1)),
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        keep = (freqs >= fmin) & (freqs <= fmax)
        spec = np.log1p(power[keep]).astype(np.float32)
        if target_freq_bins is not None or target_time_bins is not None:
            freq_bins = target_freq_bins or spec.shape[0]
            time_bins = target_time_bins or spec.shape[1]
            spec = zoom(
                spec,
                zoom=(freq_bins / spec.shape[0], time_bins / spec.shape[1]),
                order=1,
            ).astype(np.float32)
        channel_specs.append(spec)
    return np.stack(channel_specs, axis=0)


def subject_from_path(path: Path) -> str:
    match = re.match(r"([A-Z]{2}\d{4})", path.name.upper())
    return match.group(1) if match else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Sleep-EDF log-spectrogram tensors")
    parser.add_argument("--sleep-edf-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="sleep-edf-expanded")
    parser.add_argument("--channel", action="append", default=None, help="Channel to extract; repeat to use multiple")
    parser.add_argument("--epoch-duration", type=float, default=30.0)
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--noverlap", type=int, default=64)
    parser.add_argument("--fmin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--wake-context-minutes", type=float, default=30.0)
    parser.add_argument("--target-freq-bins", type=int, default=None)
    parser.add_argument("--target-time-bins", type=int, default=None)
    parser.add_argument("--limit-recordings", type=int, default=None)
    parser.add_argument("--limit-epochs", type=int, default=None)
    args = parser.parse_args()

    wake_context_minutes = args.wake_context_minutes
    if wake_context_minutes is not None and wake_context_minutes < 0:
        wake_context_minutes = None

    pairs = find_sleep_edf_pairs(args.sleep_edf_root)
    if args.limit_recordings is not None:
        pairs = pairs[: args.limit_recordings]
    if not pairs:
        raise FileNotFoundError(f"No Sleep-EDF PSG/Hypnogram pairs found under {args.sleep_edf_root}")

    specs = []
    labels_out = []
    metadata_rows = []
    summary_rows = []
    expected_shape = None

    for psg_path, hypnogram_path in pairs:
        print(f"Loading {psg_path}")
        header = read_edf_header(psg_path)
        selected_indices = choose_channels(header.labels, args.channel)
        selected_data, selected_channels, sfreq, n_records = read_edf_channels(psg_path, selected_indices)
        n_samples = int(round(args.epoch_duration * sfreq))
        n_signal_epochs = int(selected_data.shape[-1] // n_samples)
        rows = trim_wake_context(
            annotation_epochs(hypnogram_path, args.epoch_duration),
            wake_context_minutes=wake_context_minutes,
            epoch_duration=args.epoch_duration,
        )
        if args.limit_epochs is not None:
            rows = rows[: args.limit_epochs]

        kept = 0
        skipped = 0
        for annotation_index, (onset, label, description) in enumerate(rows):
            if label == MOVEMENT_OR_UNKNOWN:
                skipped += 1
                continue
            start = int(round(onset * sfreq))
            stop = start + n_samples
            if start < 0 or stop > selected_data.shape[-1]:
                skipped += 1
                continue
            epoch_idx = int(round(onset / args.epoch_duration))
            data = selected_data[:, start:stop]
            spec = compute_epoch_spectrogram(
                epoch_data=data,
                sfreq=sfreq,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                fmin=args.fmin,
                fmax=args.fmax,
                target_freq_bins=args.target_freq_bins,
                target_time_bins=args.target_time_bins,
            )
            if expected_shape is None:
                expected_shape = spec.shape
            if spec.shape != expected_shape or not np.all(np.isfinite(spec)):
                skipped += 1
                continue

            specs.append(spec)
            labels_out.append(label)
            kept += 1
            metadata_rows.append(
                {
                    "dataset_id": args.dataset_id,
                    "subject": subject_from_path(psg_path),
                    "recording": str(psg_path.relative_to(Path(args.sleep_edf_root))),
                    "hypnogram": str(hypnogram_path.relative_to(Path(args.sleep_edf_root))),
                    "channels": "|".join(selected_channels),
                    "epoch_index": epoch_idx,
                    "annotation_index": annotation_index,
                    "epoch_start_time_sec": float(onset),
                    "label": int(label),
                    "label_name": SLEEP_STAGE_NAMES.get(int(label), str(label)),
                    "source_stage": description,
                }
            )

        summary_rows.append(
            {
                "dataset_id": args.dataset_id,
                "recording": str(psg_path),
                "hypnogram": str(hypnogram_path),
                "subject": subject_from_path(psg_path),
                "channels": "|".join(selected_channels),
                "kept_epochs": int(kept),
                "skipped_epochs": int(skipped),
                "sfreq": sfreq,
                "n_channels": int(len(selected_channels)),
                "n_signal_epochs": int(n_signal_epochs),
                "n_edf_records": int(n_records),
                "n_annotation_epochs": int(len(rows)),
            }
        )
        print(f"  Kept {kept} epochs; skipped {skipped}")

    if not specs:
        raise RuntimeError("No Sleep-EDF spectrogram epochs were extracted")

    X = np.stack(specs, axis=0).astype(np.float32)
    y = np.asarray(labels_out, dtype=np.int64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_spectrograms.npy", X)
    np.save(output_dir / "y_labels.npy", y)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "epoch_metadata.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "extraction_summary.csv", index=False)

    audit = {
        "dataset_id": args.dataset_id,
        "sleep_edf_root": args.sleep_edf_root,
        "n_rows": int(X.shape[0]),
        "spectrogram_shape": list(X.shape[1:]),
        "nperseg": int(args.nperseg),
        "noverlap": int(args.noverlap),
        "fmin": float(args.fmin),
        "fmax": float(args.fmax),
        "wake_context_minutes": wake_context_minutes,
        "target_freq_bins": args.target_freq_bins,
        "target_time_bins": args.target_time_bins,
        "class_distribution": {
            SLEEP_STAGE_NAMES.get(int(label), str(int(label))): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
    }
    with open(output_dir / "spectrogram_extraction_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Saved Sleep-EDF spectrograms to {output_dir.resolve()}")
    print(f"  X_spectrograms.npy: {X.shape}")
    print(f"  y_labels.npy: {y.shape}")


if __name__ == "__main__":
    main()
