"""Run a sequential marathon of sleep-stage accuracy experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "ds006695_spectrograms_all19"
DEFAULT_ROOT_DIR = PROJECT_DIR / "outputs" / "accuracy_marathon"


@dataclass(frozen=True)
class Experiment:
    name: str
    script: str
    args: tuple[str, ...]
    notes: str


def base_window_args(output_dir: Path) -> list[str]:
    return [
        "--spectrograms-path",
        str(DEFAULT_DATA_DIR / "X_spectrograms.npy"),
        "--labels-path",
        str(DEFAULT_DATA_DIR / "y_labels.npy"),
        "--metadata-path",
        str(DEFAULT_DATA_DIR / "epoch_metadata.csv"),
        "--output-dir",
        str(output_dir),
        "--n-splits",
        "5",
        "--early-stopping-patience",
        "7",
        "--lr-plateau-patience",
        "2",
        "--grad-clip-norm",
        "1.0",
    ]


def base_seq2seq_args(output_dir: Path) -> list[str]:
    return [
        "--spectrograms-path",
        str(DEFAULT_DATA_DIR / "X_spectrograms.npy"),
        "--labels-path",
        str(DEFAULT_DATA_DIR / "y_labels.npy"),
        "--metadata-path",
        str(DEFAULT_DATA_DIR / "epoch_metadata.csv"),
        "--output-dir",
        str(output_dir),
        "--n-splits",
        "5",
        "--early-stopping-patience",
        "6",
        "--lr-plateau-patience",
        "2",
    ]


def build_experiments(root_dir: Path) -> list[Experiment]:
    return [
        Experiment(
            name="exp_01_gru_r4_e18_smooth",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_01_gru_r4_e18_smooth")
                + [
                    "--model",
                    "cnn_gru",
                    "--sequence-radius",
                    "4",
                    "--epochs",
                    "18",
                    "--batch-size",
                    "96",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.35",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.05",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "42",
                ]
            ),
            notes="Best known setup, trained longer with mild regularization.",
        ),
        Experiment(
            name="exp_02_attention_r4_e18",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_02_attention_r4_e18")
                + [
                    "--model",
                    "cnn_gru_attention",
                    "--sequence-radius",
                    "4",
                    "--epochs",
                    "18",
                    "--batch-size",
                    "96",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.35",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.05",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "43",
                ]
            ),
            notes="Attention-pooled context to improve transition-heavy errors.",
        ),
        Experiment(
            name="exp_03_gru_r6_long_context",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_03_gru_r6_long_context")
                + [
                    "--model",
                    "cnn_gru",
                    "--sequence-radius",
                    "6",
                    "--epochs",
                    "16",
                    "--batch-size",
                    "64",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.35",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.05",
                    "--learning-rate",
                    "0.0007",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "44",
                ]
            ),
            notes="13-epoch context window, testing whether more temporal context helps.",
        ),
        Experiment(
            name="exp_04_attention_r6_long_context",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_04_attention_r6_long_context")
                + [
                    "--model",
                    "cnn_gru_attention",
                    "--sequence-radius",
                    "6",
                    "--epochs",
                    "16",
                    "--batch-size",
                    "64",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.35",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.05",
                    "--learning-rate",
                    "0.0007",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "45",
                ]
            ),
            notes="Attention plus long context.",
        ),
        Experiment(
            name="exp_05_gru_aug_r4",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_05_gru_aug_r4")
                + [
                    "--model",
                    "cnn_gru",
                    "--sequence-radius",
                    "4",
                    "--epochs",
                    "18",
                    "--batch-size",
                    "96",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.4",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.05",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--augment",
                    "--time-mask-prob",
                    "0.20",
                    "--freq-mask-prob",
                    "0.15",
                    "--noise-std",
                    "0.015",
                    "--seed",
                    "46",
                ]
            ),
            notes="Light SpecAugment-style regularization.",
        ),
        Experiment(
            name="exp_06_attention_aug_r4",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_06_attention_aug_r4")
                + [
                    "--model",
                    "cnn_gru_attention",
                    "--sequence-radius",
                    "4",
                    "--epochs",
                    "18",
                    "--batch-size",
                    "96",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.4",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.05",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--augment",
                    "--time-mask-prob",
                    "0.20",
                    "--freq-mask-prob",
                    "0.15",
                    "--noise-std",
                    "0.015",
                    "--seed",
                    "47",
                ]
            ),
            notes="Attention model with light spectrogram augmentation.",
        ),
        Experiment(
            name="exp_07_gru_balanced_sampler",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_07_gru_balanced_sampler")
                + [
                    "--model",
                    "cnn_gru",
                    "--sequence-radius",
                    "4",
                    "--epochs",
                    "18",
                    "--batch-size",
                    "96",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.35",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.03",
                    "--balanced-sampling",
                    "--sampler-n1-multiplier",
                    "1.15",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "48",
                ]
            ),
            notes="Balanced sampler with a small N1 emphasis.",
        ),
        Experiment(
            name="exp_08_gru_n1_weight",
            script="train_spectrogram_sequence_model.py",
            args=tuple(
                base_window_args(root_dir / "exp_08_gru_n1_weight")
                + [
                    "--model",
                    "cnn_gru",
                    "--sequence-radius",
                    "4",
                    "--epochs",
                    "18",
                    "--batch-size",
                    "96",
                    "--hidden-size",
                    "80",
                    "--dropout",
                    "0.35",
                    "--normalization",
                    "channel",
                    "--label-smoothing",
                    "0.03",
                    "--n1-weight-multiplier",
                    "1.2",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "49",
                ]
            ),
            notes="Small N1 class-weight bump without aggressive thresholding.",
        ),
        Experiment(
            name="exp_09_seq2seq_gru_b32_s16_e24",
            script="train_spectrogram_seq2seq_model.py",
            args=tuple(
                base_seq2seq_args(root_dir / "exp_09_seq2seq_gru_b32_s16_e24")
                + [
                    "--model",
                    "cnn_gru",
                    "--block-length",
                    "32",
                    "--block-stride",
                    "16",
                    "--epochs",
                    "24",
                    "--batch-size",
                    "32",
                    "--hidden-size",
                    "96",
                    "--dropout",
                    "0.35",
                    "--learning-rate",
                    "0.0008",
                    "--weight-decay",
                    "0.0002",
                    "--seed",
                    "50",
                ]
            ),
            notes="Stronger seq2seq GRU than the earlier 8-epoch baseline.",
        ),
        Experiment(
            name="exp_10_seq2seq_transformer_b32_e48",
            script="train_spectrogram_seq2seq_model.py",
            args=tuple(
                base_seq2seq_args(root_dir / "exp_10_seq2seq_transformer_b32_e48")
                + [
                    "--model",
                    "cnn_transformer",
                    "--block-length",
                    "32",
                    "--block-stride",
                    "16",
                    "--epochs",
                    "48",
                    "--batch-size",
                    "32",
                    "--hidden-size",
                    "128",
                    "--transformer-heads",
                    "4",
                    "--transformer-layers",
                    "2",
                    "--dropout",
                    "0.35",
                    "--learning-rate",
                    "0.0005",
                    "--weight-decay",
                    "0.0003",
                    "--seed",
                    "51",
                ]
            ),
            notes="Longer transformer seq2seq candidate.",
        ),
    ]


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n\n")
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def run_viterbi(experiment_dir: Path, python_executable: str, force: bool) -> bool:
    predictions_path = experiment_dir / "cv_predictions.csv"
    if not predictions_path.exists():
        print(f"Skipping Viterbi for {experiment_dir.name}: no cv_predictions.csv")
        return False
    output_dir = experiment_dir.parent / f"{experiment_dir.name}_viterbi"
    done_path = output_dir / "viterbi_smoothing_summary.csv"
    if done_path.exists() and not force:
        print(f"Skipping Viterbi for {experiment_dir.name}: already complete")
        return True
    command = [
        python_executable,
        "scripts/smooth_predictions_viterbi.py",
        "--predictions-path",
        str(predictions_path),
        "--output-dir",
        str(output_dir),
        "--transition-weights",
        "0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
    ]
    return run_command(command, output_dir / "run.log") == 0


def write_manifest(root_dir: Path, experiments: list[Experiment]) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_dir": str(DEFAULT_DATA_DIR),
        "created_at_epoch": time.time(),
        "experiments": [asdict(experiment) for experiment in experiments],
    }
    with open(root_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10 full accuracy experiments sequentially")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--force", action="store_true", help="rerun experiments even if metrics.json exists")
    parser.add_argument("--start-at", default=None, help="experiment name to start from")
    parser.add_argument("--skip-viterbi", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    experiments = build_experiments(root_dir)
    write_manifest(root_dir, experiments)

    missing = [
        DEFAULT_DATA_DIR / "X_spectrograms.npy",
        DEFAULT_DATA_DIR / "y_labels.npy",
        DEFAULT_DATA_DIR / "epoch_metadata.csv",
    ]
    missing = [path for path in missing if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required spectrogram data files: {missing}")

    started = args.start_at is None
    failures = []
    python_executable = sys.executable
    for index, experiment in enumerate(experiments, start=1):
        if not started:
            started = experiment.name == args.start_at
            if not started:
                continue
        experiment_dir = root_dir / experiment.name
        metrics_path = experiment_dir / "metrics.json"
        print(f"\n=== [{index}/{len(experiments)}] {experiment.name} ===")
        print(experiment.notes)
        if metrics_path.exists() and not args.force:
            print(f"Skipping training: {metrics_path} already exists")
            train_ok = True
        else:
            command = [python_executable, f"scripts/{experiment.script}", *experiment.args]
            train_ok = run_command(command, experiment_dir / "run.log") == 0
        if train_ok and not args.skip_viterbi:
            run_viterbi(experiment_dir, python_executable, force=args.force)
        if not train_ok:
            failures.append(experiment.name)
            print(f"Experiment failed: {experiment.name}. Continuing with the next candidate.")

        summary_command = [
            python_executable,
            "scripts/summarize_accuracy_experiments.py",
            "--root-dir",
            str(root_dir),
        ]
        run_command(summary_command, root_dir / "summary.log")

    print("\n=== Marathon finished ===")
    if failures:
        print("Failed experiments: " + ", ".join(failures))
    print(f"Summary: {root_dir / 'accuracy_experiment_summary.md'}")


if __name__ == "__main__":
    main()
