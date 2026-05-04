"""Train a spectrogram sequence model on one corpus and evaluate another."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.train_spectrogram_sequence_model import (
    SpectrogramSequenceDataset,
    build_sequence_index,
    compute_normalization_stats,
    make_model,
    predict_loader,
    summarize,
    train_one_fold,
)

SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


def add_training_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "group_column": None,
        "n_splits": 5,
        "max_folds": None,
        "model": "cnn_gru",
        "normalization": "channel",
        "normalization_clip": None,
        "epochs": 8,
        "batch_size": 64,
        "hidden_size": 80,
        "dropout": 0.35,
        "learning_rate": 0.0007,
        "weight_decay": 0.0002,
        "loss": "cross_entropy",
        "focal_gamma": 2.0,
        "label_smoothing": 0.05,
        "grad_clip_norm": 1.0,
        "n1_weight_multiplier": 1.0,
        "balanced_sampling": False,
        "sampler_n1_multiplier": 1.0,
        "augment": False,
        "time_mask_prob": 0.0,
        "max_time_mask_fraction": 0.1,
        "freq_mask_prob": 0.0,
        "max_freq_mask_fraction": 0.1,
        "channel_mask_prob": 0.0,
        "noise_std": 0.0,
        "inner_val_splits": 5,
        "validation_fraction": 0.2,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
        "lr_plateau_patience": 2,
        "lr_plateau_factor": 0.5,
        "seed": 44,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def load_dataset(spectrograms_path: str, labels_path: str, metadata_path: str):
    X = np.load(spectrograms_path, mmap_mode="r")
    y = np.load(labels_path)
    metadata = pd.read_csv(metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Spectrogram, label, and metadata row counts must match")
    return X, y, metadata


def write_predictions(
    output_path: Path,
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: list[int],
) -> None:
    frame = metadata.copy()
    frame["true_label"] = y_true.astype(int)
    frame["pred_label"] = y_pred.astype(int)
    frame["true_stage"] = [SLEEP_STAGE_NAMES.get(int(label), str(int(label))) for label in y_true]
    frame["pred_stage"] = [SLEEP_STAGE_NAMES.get(int(label), str(int(label))) for label in y_pred]
    for pos, label in enumerate(labels):
        frame[f"prob_{SLEEP_STAGE_NAMES.get(int(label), label)}"] = y_prob[:, pos]
    frame["prediction_confidence"] = np.nanmax(y_prob, axis=1)
    frame.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train source spectrogram model and evaluate target corpus")
    parser.add_argument("--source-spectrograms-path", required=True)
    parser.add_argument("--source-labels-path", required=True)
    parser.add_argument("--source-metadata-path", required=True)
    parser.add_argument("--target-spectrograms-path", required=True)
    parser.add_argument("--target-labels-path", required=True)
    parser.add_argument("--target-metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-radius", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--normalization", default="channel")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.0002)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=44)
    args = add_training_defaults(parser.parse_args())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_source, y_source, source_metadata = load_dataset(
        args.source_spectrograms_path,
        args.source_labels_path,
        args.source_metadata_path,
    )
    X_target, y_target, target_metadata = load_dataset(
        args.target_spectrograms_path,
        args.target_labels_path,
        args.target_metadata_path,
    )
    if X_source.shape[1:] != X_target.shape[1:]:
        raise ValueError(f"Source shape {X_source.shape[1:]} does not match target shape {X_target.shape[1:]}")

    source_groups = derive_group_ids(source_metadata).astype(str).to_numpy()
    target_groups = derive_group_ids(target_metadata).astype(str).to_numpy()
    source_sequence_index = build_sequence_index(source_metadata, pd.Series(source_groups), args.sequence_radius)
    target_sequence_index = build_sequence_index(target_metadata, pd.Series(target_groups), args.sequence_radius)

    args.output_dir = str(output_dir)
    source_idx = np.arange(len(y_source))
    print(f"Training source model on {len(source_idx)} rows; evaluating target rows={len(y_target)}")
    _, _, train_info = train_one_fold(
        X_source,
        y_source,
        source_sequence_index,
        source_idx,
        source_idx,
        source_groups,
        fold_idx=1,
        args=args,
        device=device,
    )
    checkpoint = torch.load(train_info["checkpoint_path"], map_location=device, weights_only=False)
    model = make_model(args, n_channels=X_source.shape[1], n_classes=len(checkpoint["classes"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_dataset = SpectrogramSequenceDataset(
        X_target,
        y_target,
        target_sequence_index,
        np.arange(len(y_target)),
        group_ids=target_groups,
        normalization_stats=checkpoint["normalization_stats"],
        clip_value=args.normalization_clip,
    )
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
    y_pred, y_prob = predict_loader(model, target_loader, device)
    labels = [int(label) for label in checkpoint["classes"]]
    summary = summarize(y_target, y_pred, labels)

    write_predictions(output_dir / "target_predictions.csv", target_metadata, y_target, y_pred, y_prob, labels)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_rows": int(len(y_source)),
                "target_rows": int(len(y_target)),
                "source_shape": list(X_source.shape[1:]),
                "target_shape": list(X_target.shape[1:]),
                "sequence_radius": int(args.sequence_radius),
                "source_checkpoint": train_info["checkpoint_path"],
                "target_summary": summary,
            },
            handle,
            indent=2,
        )
    print(
        "Target: "
        f"balanced_accuracy={summary['balanced_accuracy']:.4f}, "
        f"macro_f1={summary['macro_f1']:.4f}, "
        f"n1_f1={summary['n1_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
