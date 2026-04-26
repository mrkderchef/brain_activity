"""Train small CNN sequence models on spectrogram epoch windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sleep_stage_prediction.metadata import derive_group_ids
from sleep_stage_prediction.training import SLEEP_STAGE_NAMES


def sort_group_indices(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    sort_frame = metadata.iloc[indices].copy()
    sort_frame["_row_index"] = indices
    sort_columns = []
    for column in ["epoch_start_time_sec", "epoch_index"]:
        if column in sort_frame.columns:
            sort_frame[column] = pd.to_numeric(sort_frame[column], errors="coerce")
            sort_columns.append(column)
    if sort_columns:
        sort_frame = sort_frame.sort_values(sort_columns + ["_row_index"], kind="mergesort")
    return sort_frame["_row_index"].to_numpy(dtype=int)


def build_sequence_index(metadata: pd.DataFrame, group_ids: pd.Series, radius: int) -> np.ndarray:
    sequence_index = np.empty((len(metadata), 2 * radius + 1), dtype=np.int64)
    group_array = group_ids.to_numpy()
    for group_id in sorted(group_ids.unique()):
        indices = np.where(group_array == group_id)[0]
        ordered = sort_group_indices(metadata, indices)
        padded = np.pad(ordered, (radius, radius), mode="edge")
        for pos, row_idx in enumerate(ordered):
            sequence_index[row_idx] = padded[pos : pos + 2 * radius + 1]
    return sequence_index


class SpectrogramSequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_index: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.X = X
        self.y = y
        self.sequence_index = sequence_index
        self.indices = indices.astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row_idx = self.indices[item]
        seq = np.asarray(self.X[self.sequence_index[row_idx]], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            seq = (seq - self.mean) / self.std
        return torch.from_numpy(seq).float(), torch.tensor(int(self.y[row_idx]), dtype=torch.long)


class CnnGruSleepNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 5, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.gru = nn.GRU(
            input_size=32 * 4 * 4,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, freqs, times = x.shape
        encoded = self.encoder(x.reshape(batch_size * seq_len, channels, freqs, times))
        encoded = encoded.reshape(batch_size, seq_len, -1)
        output, _ = self.gru(encoded)
        center = output[:, seq_len // 2, :]
        return self.classifier(center)


class CnnTcnSleepNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 5, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_size),
            nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, freqs, times = x.shape
        encoded = self.encoder(x.reshape(batch_size * seq_len, channels, freqs, times))
        encoded = encoded.reshape(batch_size, seq_len, -1).transpose(1, 2)
        temporal = self.temporal(encoded).transpose(1, 2)
        center = temporal[:, seq_len // 2, :]
        return self.classifier(center)


def make_model(args: argparse.Namespace, n_channels: int, n_classes: int) -> nn.Module:
    if args.model == "cnn_gru":
        return CnnGruSleepNet(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    if args.model == "cnn_tcn":
        return CnnTcnSleepNet(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


def compute_normalization_stats(X: np.ndarray, train_idx: np.ndarray, mode: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "none":
        return None, None
    train_values = np.asarray(X[train_idx], dtype=np.float32)
    if mode == "global":
        mean = np.asarray(train_values.mean(), dtype=np.float32)
        std = np.asarray(train_values.std(), dtype=np.float32)
    elif mode == "channel":
        mean = train_values.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = train_values.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        mean = mean.reshape(1, train_values.shape[1], 1, 1)
        std = std.reshape(1, train_values.shape[1], 1, 1)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    std = np.maximum(std, np.asarray(1e-6, dtype=np.float32))
    return mean, std


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    target_names = [SLEEP_STAGE_NAMES.get(label, f"Stage-{label}") for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "n1_precision": float(report["N1"]["precision"]),
        "n1_recall": float(report["N1"]["recall"]),
        "n1_f1": float(report["N1"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    sequence_index: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    mean, std = compute_normalization_stats(X, train_idx, args.normalization)
    train_dataset = SpectrogramSequenceDataset(X, y, sequence_index, train_idx, mean=mean, std=std)
    test_dataset = SpectrogramSequenceDataset(X, y, sequence_index, test_idx, mean=mean, std=std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = make_model(args, n_channels=X.shape[1], n_classes=len(np.unique(y))).to(device)
    classes = np.array(sorted(np.unique(y)))
    class_weights = compute_class_weight("balanced", classes=classes, y=y[train_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
        print(f"    epoch {epoch + 1}/{args.epochs} loss={total_loss / len(train_dataset):.4f}", flush=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN sequence models on spectrogram windows")
    parser.add_argument("--spectrograms-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--sequence-radius", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--model", choices=["cnn_gru", "cnn_tcn"], default="cnn_gru")
    parser.add_argument("--normalization", choices=["none", "global", "channel"], default="channel")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = np.load(args.spectrograms_path, mmap_mode="r")
    y = np.load(args.labels_path)
    metadata = pd.read_csv(args.metadata_path)
    if X.shape[0] != len(y) or len(metadata) != len(y):
        raise ValueError("Spectrogram, label, and metadata row counts must match")

    group_ids = derive_group_ids(metadata, args.group_column).astype(str)
    sequence_index = build_sequence_index(metadata, group_ids, args.sequence_radius)
    groups = group_ids.to_numpy()
    labels = sorted(int(label) for label in np.unique(y))

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_pred = np.full_like(y, fill_value=-1)
    fold_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups), start=1):
        if args.max_folds is not None and fold_idx > args.max_folds:
            break
        print(f"Fold {fold_idx}: train={len(train_idx)} test={len(test_idx)}")
        fold_pred = train_one_fold(X, y, sequence_index, train_idx, test_idx, args, device)
        y_pred[test_idx] = fold_pred
        fold_summary = summarize(y[test_idx], fold_pred, labels)
        fold_rows.append({"fold": fold_idx, **{k: fold_summary[k] for k in [
            "accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1", "n1_f1"
        ]}})
        print(
            f"  fold {fold_idx}: balanced_accuracy={fold_summary['balanced_accuracy']:.4f} "
            f"macro_f1={fold_summary['macro_f1']:.4f}",
            flush=True,
        )

    evaluated = y_pred != -1
    summary = summarize(y[evaluated], y_pred[evaluated], labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions = metadata.loc[evaluated].copy()
    predictions["true_label"] = y[evaluated]
    predictions["pred_label"] = y_pred[evaluated]
    predictions["true_stage"] = predictions["true_label"].map(SLEEP_STAGE_NAMES)
    predictions["pred_stage"] = predictions["pred_label"].map(SLEEP_STAGE_NAMES)
    predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    result = {
        "model": f"{args.model}_spectrogram",
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "max_folds": args.max_folds,
        "evaluated_rows": int(np.sum(evaluated)),
        "n_rows": int(len(y)),
        "spectrogram_shape": list(X.shape[1:]),
        "sequence_radius": int(args.sequence_radius),
        "sequence_length": int(2 * args.sequence_radius + 1),
        "normalization": args.normalization,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_size": int(args.hidden_size),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "summary": summary,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved spectrogram model results to {output_dir}")
    print(
        f"Overall: balanced_accuracy={summary['balanced_accuracy']:.4f}, "
        f"macro_f1={summary['macro_f1']:.4f}, n1_f1={summary['n1_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
