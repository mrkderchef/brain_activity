"""Train CNN sequence-to-sequence sleep-stage models on spectrogram blocks."""

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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
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


def build_blocks(
    metadata: pd.DataFrame,
    group_ids: pd.Series,
    allowed_indices: np.ndarray,
    block_length: int,
    stride: int,
) -> list[np.ndarray]:
    allowed = set(int(idx) for idx in allowed_indices)
    group_array = group_ids.to_numpy()
    blocks = []
    for group_id in sorted(group_ids.unique()):
        indices = np.where(group_array == group_id)[0]
        ordered = [int(idx) for idx in sort_group_indices(metadata, indices) if int(idx) in allowed]
        if not ordered:
            continue
        for start in range(0, len(ordered), stride):
            block = ordered[start : start + block_length]
            if block:
                blocks.append(np.asarray(block, dtype=np.int64))
    return blocks


class SpectrogramBlockDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        blocks: list[np.ndarray],
        block_length: int,
        normalization_stats: dict | None = None,
    ):
        self.X = X
        self.y = y
        self.blocks = blocks
        self.block_length = block_length
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, item: int):
        block = self.blocks[item]
        seq = np.asarray(self.X[block], dtype=np.float32)
        labels = np.asarray(self.y[block], dtype=np.int64)
        mask = np.ones(len(block), dtype=np.bool_)
        row_indices = np.asarray(block, dtype=np.int64)
        if len(block) < self.block_length:
            pad_len = self.block_length - len(block)
            seq = np.concatenate([seq, np.zeros((pad_len, *seq.shape[1:]), dtype=np.float32)], axis=0)
            labels = np.concatenate([labels, np.full(pad_len, -100, dtype=np.int64)], axis=0)
            mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.bool_)], axis=0)
            row_indices = np.concatenate([row_indices, np.full(pad_len, -1, dtype=np.int64)], axis=0)
        if self.normalization_stats is not None:
            seq = (seq - self.normalization_stats["center"]) / self.normalization_stats["scale"]
        return (
            torch.from_numpy(seq).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(mask),
            torch.from_numpy(row_indices).long(),
        )


class CnnGruSeq2SeqNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_size: int = 64, dropout: float = 0.3):
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
        return self.classifier(output)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1]])


class CnnTransformerSeq2SeqNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_size: int = 128,
        dropout: float = 0.3,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_size: int | None = None,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by transformer heads={n_heads}")
        feedforward_size = feedforward_size or hidden_size * 4
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
        self.projection = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = SinusoidalPositionalEncoding(hidden_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels, freqs, times = x.shape
        encoded = self.encoder(x.reshape(batch_size * seq_len, channels, freqs, times))
        encoded = self.projection(encoded).reshape(batch_size, seq_len, -1)
        encoded = self.position(encoded)
        output = self.transformer(encoded, src_key_padding_mask=padding_mask)
        return self.classifier(output)


def create_seq2seq_model(args: argparse.Namespace, n_channels: int, n_classes: int) -> nn.Module:
    if args.model == "cnn_gru":
        return CnnGruSeq2SeqNet(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    if args.model == "cnn_transformer":
        return CnnTransformerSeq2SeqNet(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            n_heads=args.transformer_heads,
            n_layers=args.transformer_layers,
            feedforward_size=args.transformer_ff_size,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def forward_model(model: nn.Module, xb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if isinstance(model, CnnTransformerSeq2SeqNet):
        return model(xb, padding_mask=~mask.bool())
    return model(xb)


def compute_channel_normalization(X: np.ndarray, train_idx: np.ndarray) -> dict:
    train_values = np.asarray(X[train_idx], dtype=np.float32)
    mean = train_values.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = train_values.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    return {
        "center": mean.reshape(1, train_values.shape[1], 1, 1),
        "scale": np.maximum(std.reshape(1, train_values.shape[1], 1, 1), np.asarray(1e-6, dtype=np.float32)),
    }


def split_train_validation(y: np.ndarray, train_idx: np.ndarray, group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_groups = group_ids[train_idx]
    unique_groups = np.unique(train_groups)
    if len(unique_groups) < 2:
        stratify = y[train_idx]
        _, counts = np.unique(stratify, return_counts=True)
        if np.any(counts < 2):
            stratify = None
        fit_local, val_local = train_test_split(
            np.arange(len(train_idx)),
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )
    else:
        n_splits = min(5, len(unique_groups))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit_local, val_local = next(splitter.split(np.zeros(len(train_idx)), y[train_idx], groups=train_groups))
    return train_idx[fit_local], train_idx[val_local]


def make_class_weights(y: np.ndarray, train_idx: np.ndarray, classes: np.ndarray) -> torch.Tensor:
    class_weights = np.ones(len(classes), dtype=np.float32)
    present_classes = np.array(sorted(np.unique(y[train_idx])))
    present_weights = compute_class_weight("balanced", classes=present_classes, y=y[train_idx])
    for class_label, weight in zip(present_classes, present_weights):
        class_pos = int(np.where(classes == class_label)[0][0])
        class_weights[class_pos] = float(weight)
    return torch.tensor(class_weights, dtype=torch.float32)


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
        "n1_precision": float(report.get("N1", {}).get("precision", np.nan)),
        "n1_recall": float(report.get("N1", {}).get("recall", np.nan)),
        "n1_f1": float(report.get("N1", {}).get("f1-score", np.nan)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for xb, yb, mask, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, xb, mask)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        loss.backward()
        optimizer.step()
        n_tokens = int(mask.sum().item())
        total_loss += float(loss.item()) * n_tokens
        total_tokens += n_tokens
    return total_loss / max(1, total_tokens)


def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    n_rows: int,
    n_classes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.full(n_rows, fill_value=-1, dtype=np.int64)
    y_prob = np.full((n_rows, n_classes), fill_value=np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for xb, _, mask, row_indices in loader:
            xb = xb.to(device)
            mask = mask.to(device)
            logits = forward_model(model, xb, mask)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = np.argmax(prob, axis=-1)
            mask_np = mask.numpy().astype(bool)
            rows_np = row_indices.numpy()
            valid_rows = rows_np[mask_np]
            y_pred[valid_rows] = pred[mask_np]
            y_prob[valid_rows] = prob[mask_np]
    return y_pred, y_prob


def train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    group_ids: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_idx: int,
    args: argparse.Namespace,
    labels: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    groups_array = group_ids.to_numpy()
    fit_idx, val_idx = split_train_validation(y, train_idx, groups_array, args.seed)
    normalization_stats = compute_channel_normalization(X, fit_idx)
    fit_blocks = build_blocks(metadata, group_ids, fit_idx, args.block_length, args.block_stride)
    val_blocks = build_blocks(metadata, group_ids, val_idx, args.block_length, args.block_length)
    test_blocks = build_blocks(metadata, group_ids, test_idx, args.block_length, args.block_length)

    train_loader = DataLoader(
        SpectrogramBlockDataset(X, y, fit_blocks, args.block_length, normalization_stats),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SpectrogramBlockDataset(X, y, val_blocks, args.block_length, normalization_stats),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SpectrogramBlockDataset(X, y, test_blocks, args.block_length, normalization_stats),
        batch_size=args.batch_size,
        shuffle=False,
    )

    classes = np.array(labels)
    model = create_seq2seq_model(args, n_channels=X.shape[1], n_classes=len(labels)).to(device)
    class_weights = make_class_weights(y, fit_idx, classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
    )

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_pred, _ = predict_dataset(model, val_loader, len(y), len(labels), device)
        evaluated = val_pred[val_idx] != -1
        val_summary = summarize(y[val_idx][evaluated], val_pred[val_idx][evaluated], labels)
        val_score = val_summary["balanced_accuracy"]
        scheduler.step(val_score)
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_balanced_accuracy": float(val_score),
                "val_macro_f1": float(val_summary["macro_f1"]),
                "val_n1_f1": float(val_summary["n1_f1"]),
            }
        )
        print(
            f"    epoch {epoch + 1}/{args.epochs} loss={train_loss:.4f} "
            f"val_bal_acc={val_score:.4f} val_macro_f1={val_summary['macro_f1']:.4f}",
            flush=True,
        )
        if val_score > best_score + args.early_stopping_min_delta:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"    early stopping at epoch {epoch + 1}; best epoch={best_epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold_{fold_idx:02d}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_stats": normalization_stats,
            "args": vars(args),
            "fold": int(fold_idx),
            "best_epoch": int(best_epoch),
            "best_val_balanced_accuracy": float(best_score),
        },
        checkpoint_path,
    )

    fold_pred, fold_prob = predict_dataset(model, test_loader, len(y), len(labels), device)
    info = {
        "fold": int(fold_idx),
        "fit_rows": int(len(fit_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "fit_blocks": int(len(fit_blocks)),
        "validation_blocks": int(len(val_blocks)),
        "test_blocks": int(len(test_blocks)),
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    return fold_pred, fold_prob, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN sequence-to-sequence spectrogram model")
    parser.add_argument("--spectrograms-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--block-stride", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", choices=["cnn_gru", "cnn_transformer"], default="cnn_gru")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-patience", type=int, default=2)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
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
    groups = group_ids.to_numpy()
    labels = sorted(int(label) for label in np.unique(y))
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_pred = np.full_like(y, fill_value=-1)
    y_prob = np.full((len(y), len(labels)), fill_value=np.nan, dtype=np.float32)
    fold_assignments = np.full(len(y), fill_value=-1, dtype=np.int64)
    fold_rows = []
    training_history = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups), start=1):
        if args.max_folds is not None and fold_idx > args.max_folds:
            break
        print(f"Fold {fold_idx}: train={len(train_idx)} test={len(test_idx)}")
        fold_pred, fold_prob, fold_info = train_one_fold(
            X,
            y,
            metadata,
            group_ids,
            train_idx,
            test_idx,
            fold_idx,
            args,
            labels,
            device,
        )
        valid = fold_pred[test_idx] != -1
        y_pred[test_idx[valid]] = fold_pred[test_idx[valid]]
        y_prob[test_idx[valid]] = fold_prob[test_idx[valid]]
        fold_assignments[test_idx[valid]] = fold_idx
        training_history.append(fold_info)
        fold_summary = summarize(y[test_idx][valid], fold_pred[test_idx][valid], labels)
        fold_rows.append(
            {
                "fold": fold_idx,
                **{
                    key: fold_summary[key]
                    for key in ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1", "n1_f1"]
                },
                "best_epoch": fold_info["best_epoch"],
                "best_val_balanced_accuracy": fold_info["best_val_balanced_accuracy"],
            }
        )
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
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(training_history, handle, indent=2)

    predictions = metadata.loc[evaluated].copy()
    predictions["fold"] = fold_assignments[evaluated]
    predictions["true_label"] = y[evaluated]
    predictions["pred_label"] = y_pred[evaluated]
    predictions["true_stage"] = predictions["true_label"].map(SLEEP_STAGE_NAMES)
    predictions["pred_stage"] = predictions["pred_label"].map(SLEEP_STAGE_NAMES)
    for class_idx, label in enumerate(labels):
        stage_name = SLEEP_STAGE_NAMES.get(label, f"Stage-{label}")
        predictions[f"prob_{stage_name}"] = y_prob[evaluated, class_idx]
    predictions["prediction_confidence"] = np.nanmax(y_prob[evaluated], axis=1)
    safe_prob = np.clip(y_prob[evaluated], 1e-8, 1.0)
    predictions["prediction_entropy"] = -np.sum(safe_prob * np.log(safe_prob), axis=1)
    predictions.to_csv(output_dir / "cv_predictions.csv", index=False)

    result = {
        "model": f"{args.model}_seq2seq_spectrogram",
        "cv": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "max_folds": args.max_folds,
        "evaluated_rows": int(np.sum(evaluated)),
        "n_rows": int(len(y)),
        "spectrogram_shape": list(X.shape[1:]),
        "block_length": int(args.block_length),
        "block_stride": int(args.block_stride),
        "normalization": "channel",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_size": int(args.hidden_size),
        "dropout": float(args.dropout),
        "transformer_heads": int(args.transformer_heads),
        "transformer_layers": int(args.transformer_layers),
        "transformer_ff_size": args.transformer_ff_size,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "summary": summary,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved seq2seq spectrogram model results to {output_dir}")
    print(
        f"Overall: balanced_accuracy={summary['balanced_accuracy']:.4f}, "
        f"macro_f1={summary['macro_f1']:.4f}, n1_f1={summary['n1_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
