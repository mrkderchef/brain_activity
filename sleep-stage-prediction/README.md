# Sleep Stage Prediction

Local-first sleep stage classification from EEG band-power features using the `ds003768` dataset.

## What this project does

- extracts 30-second EEG epochs into frequency-band features
- trains a Random Forest sleep stage classifier
- saves metrics, plots, and a serialized model
- exports CSVs for TVB-style visualization

## Project layout

```text
sleep-stage-prediction/
  legacy/                 Old Kaggle-specific artifacts kept out of the main workflow
  outputs/                Generated features, models, metrics, and plots
  scripts/                CLI entrypoints for extraction, training, and visualization
  src/sleep_stage_prediction/
                          Reusable package code
  README.md
  pyproject.toml
  requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Local workflow

1. Extract features from a local `ds003768` checkout:

```bash
python scripts/extract_features.py --bids-root ..\ds003768 --output-dir outputs
```

For an external BIDS sleep dataset with per-recording `_events.tsv` labels:

```bash
python scripts\download_openneuro_subset.py --dataset ds006695 --target-dir ..\ds006695 --subject 126
python scripts/extract_external_bids_sleep.py --bids-root ..\ds006695 --dataset-id ds006695 --preset ds006695 --output-dir outputs\ds006695_features
```

For Sleep-EDF Expanded after downloading the BIDS mirror:

```bash
python scripts/extract_external_bids_sleep.py --bids-root ..\NM000185 --dataset-id NM000185 --output-dir outputs\sleep_edf_features
```

Combine the current feature set with an external feature set:

```bash
python scripts\combine_feature_sets.py --feature-dir outputs --dataset-id ds003768 --feature-dir outputs\ds006695_features --dataset-id ds006695 --output-dir outputs\combined_ds003768_ds006695
```

Add ratio/entropy features, normalize per recording/subject, and train with group-aware validation:

```bash
python scripts\augment_feature_set.py --features-path outputs\combined_ds003768_ds006695\X_features.npy --labels-path outputs\combined_ds003768_ds006695\y_labels.npy --metadata-path outputs\combined_ds003768_ds006695\epoch_metadata.csv --output-dir outputs\combined_augmented
python scripts\make_balanced_subset.py --features-path outputs\combined_augmented\X_features.npy --labels-path outputs\combined_augmented\y_labels.npy --metadata-path outputs\combined_augmented\epoch_metadata.csv --output-dir outputs\balanced_800_augmented --target-per-class 800
python scripts\normalize_feature_set.py --features-path outputs\balanced_800_augmented\X_features.npy --labels-path outputs\balanced_800_augmented\y_labels.npy --metadata-path outputs\balanced_800_augmented\epoch_metadata.csv --output-dir outputs\balanced_800_augmented_normalized
python scripts\add_sequence_context.py --features-path outputs\balanced_800_augmented_normalized\X_features.npy --labels-path outputs\balanced_800_augmented_normalized\y_labels.npy --metadata-path outputs\balanced_800_augmented_normalized\epoch_metadata.csv --output-dir outputs\balanced_800_augmented_normalized_seq1 --window 1 --include-deltas
python scripts\train_group_model.py --features-path outputs\balanced_800_augmented_normalized\X_features.npy --labels-path outputs\balanced_800_augmented_normalized\y_labels.npy --metadata-path outputs\balanced_800_augmented_normalized\epoch_metadata.csv --output-dir outputs\balanced_800_augmented_normalized_group_cv --n-splits 3
```

For an N1-focused sensitivity/recall experiment on a prepared feature set:

```bash
python scripts\evaluate_n1_focus.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1_n1_focus --n-splits 5
```

2. Train from the extracted `.npy` files:

```bash
python scripts/train_model.py --features-path outputs\X_features.npy --labels-path outputs\y_labels.npy --output-dir outputs
```

3. Optionally run the full pipeline in one command:

```bash
python scripts/run_pipeline.py --bids-root ..\ds003768 --output-dir outputs
```

4. Render TVB-style plots from an existing export:

```bash
python scripts/visualize_tvb.py --export-dir outputs\tvb_export
```

## Outputs

Training writes the following files into `outputs/`:

- `X_features.npy`
- `y_labels.npy`
- `sleep_stage_model.joblib`
- `metrics.json`
- `*.png`
- `tvb_export/*.csv`

## Notes

- The active workflow is local-only. Kaggle assets were moved to `legacy/kaggle/`.
- `outputs/` is generated project state, not source code.
- If `.eeg` files are still annex pointers, extraction will not work until the real binaries are present.
- For the current `ds003768` workflow, the practical label space is `Wake`, `N1`, `N2`, and `N3`. REM is not present in the source TSV labels.
- See `docs/current_data_audit.md` for the current data-quality and class-balance findings.
- See `docs/openneuro_dataset_research.md` for candidate external datasets and integration caveats.
