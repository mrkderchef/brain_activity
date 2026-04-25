# Sleep Stage Prediction

Local-first EEG sleep-stage classification pipeline. The project started with
`ds003768` and has grown into a more thesis-friendly benchmark workflow around
external sleep data, subject-aware validation, baseline model selection, and a
first spectrogram sequence-model experiment.

## Current Status

The strongest completed tabular baseline is:

| Model | Data | Feature set | Evaluation | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---|---:|---:|---:|
| Random Forest | `ds006695`, 19 subjects | transition features, radii `2,4,6` | 5-fold subject-wise CV | `0.5465` | `0.5344` | `0.2474` |

The most promising in-progress direction is a CNN-GRU over log-spectrogram
sequence windows:

| Model | Input | Current evaluation | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|
| CNN-GRU | 5-epoch log-spectrogram sequences | first 2 of 5 subject-wise folds | `0.6314` | `0.6270` | `0.2680` |

The CNN-GRU result is intentionally treated as a running/intermediate result
until all five folds are complete. It already suggests that rawer temporal-
spectral inputs are a better next step than squeezing more out of hand-crafted
band-power tables.

## Why The Project Changed Direction

The original dataset, `ds003768`, is useful but limited for full sleep staging:

- it contains practical labels for `Wake`, `N1`, `N2`, and `N3`
- REM is absent from the source scoring files, not lost by extraction
- N3 is extremely rare: only `42` extracted epochs in the saved feature set
- random splits can look decent but overstate generalization for sleep data

Because of that, the project should not be framed as a clean 5-class sleep-stage
classifier on `ds003768`. The better thesis-safe framing is:

1. audit `ds003768` and document its label-space limits
2. add a dataset with REM and richer sleep architecture
3. evaluate with subject-aware splits
4. compare simple tabular baselines against sequence/spectrogram models

`ds006695` became the first external corpus because it has manual 30-second
sleep staging with `Wake`, `N1`, `N2`, `N3`, and `REM`, and it is manageable
enough for fast iteration. The tradeoff is strong domain shift: `ds006695` has
only three forehead EEG channels, while `ds003768` uses a very different
MR-compatible EEG setup.

## Main Data Findings

### `ds003768`

Saved feature matrix:

| File | Shape |
|---|---:|
| `outputs/X_features.npy` | `(6773, 15)` |
| `outputs/y_labels.npy` | `(6773,)` |

Class distribution:

| Stage | Count |
|---|---:|
| Wake | `3341` |
| N1 | `2129` |
| N2 | `1261` |
| N3 | `42` |
| REM | `0` |

Key interpretation:

- REM is not recoverable from this release because it is not in the source TSVs.
- N3 estimates are unstable because the class is tiny.
- A local 3-fold run reached accuracy `0.6892`, but that result is constrained
  by the incomplete label space and should not be compared directly with full
  5-class sleep-staging papers.

### `ds006695`

All 19 extracted subjects:

| Stage | Count |
|---|---:|
| Wake | `3061` |
| N1 | `1600` |
| N2 | `7691` |
| N3 | `3737` |
| REM | `3474` |

Balanced all-19 benchmark:

| Stage | Count |
|---|---:|
| Wake | `1600` |
| N1 | `1600` |
| N2 | `1600` |
| N3 | `1600` |
| REM | `1600` |

This became the main benchmark because N1 is the rarest class and therefore
defines the clean balanced cap.

## Important Lessons So Far

### Simple pooling is risky

The first mixed `ds003768` + `ds006695` experiment proved that the integration
works technically and adds REM labels, but cross-dataset transfer was close to
chance on the shared 4-class space:

| Train | Test | Balanced accuracy | Kappa |
|---|---|---:|---:|
| `ds003768` | `ds006695` subject `126` | `0.2500` | `0.0000` |
| `ds006695` subject `126` | `ds003768` | `0.2507` | `0.0007` |

Decision: keep dataset identity in metadata and prefer subject-/dataset-aware
validation over unqualified pooled reporting.

### Normalization and temporal context help

On `ds006695`, subject-wise group CV improved when adding robust
recording/subject normalization and neighboring-epoch context:

| Feature set | Features | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| augmented + normalized | `21` | `0.4866` | `0.48` |
| sequence context + augmented + normalized | `105` | `0.5394` | `0.52` |

Decision: sleep stages are not independent rows. Neighboring epochs carry real
information, especially around transitions.

### N1 is the hard class

N1 remains the main failure mode. On the sequence-context Random Forest,
false-negative N1 epochs were mostly predicted as:

| Predicted stage | Count |
|---|---:|
| N2 | `451` |
| REM | `435` |
| Wake | `286` |
| N3 | `120` |

This fits the domain: N1 is transitional and visually ambiguous. Thresholding
can improve N1 recall but hurts balanced 5-class performance:

| Strategy | Balanced accuracy | N1 recall | N1 F1 |
|---|---:|---:|---:|
| default argmax | `0.5411` | `0.1969` | `0.2435` |
| lowered N1 threshold | `0.4319` | `0.7906` | `0.3962` |

Decision: report N1-sensitive models as sensitivity analyses, not as the main
model, unless the objective explicitly changes to N1 detection.

### Transition windows are the best completed tabular baseline

Transition features add label-free rolling context within each subject/recording:

- rolling means
- rolling standard deviations
- rolling slopes
- current-minus-local-mean deviations

The best tested window setup uses radii `2,4,6`, corresponding to 5-, 9-, and
13-epoch windows. It gives the best completed tabular main model:

| Model | Feature set | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---:|---:|---:|
| Random Forest | transition `2,4,6` | `0.5465` | `0.5344` | `0.2474` |
| CatBoost | transition `2,4,6` | `0.5400` | `0.5306` | `0.2454` |
| LightGBM | transition `2,4,6` | `0.5213` | `0.5187` | `0.2640` |

Decision: use Random Forest + transition `2,4,6` as the completed tabular
baseline, because it wins the predefined main criterion: subject-wise balanced
accuracy.

### Spectrogram sequence models look like the next ceiling

The literature and GitHub survey pointed in the same direction: stronger
sleep-staging systems usually use raw EEG/EOG/PSG or spectrogram sequences,
not only per-epoch band-power summary features.

The first CNN-GRU smoke run on `ds006695` spectrogram sequences evaluated two
of five subject-wise folds so far:

| Metric | Value |
|---|---:|
| Accuracy | `0.6747` |
| Balanced accuracy | `0.6314` |
| Macro F1 | `0.6270` |
| N1 F1 | `0.2680` |

Decision: keep the tabular Random Forest as the baseline to beat, but continue
the full 5-fold CNN-GRU evaluation before making it the selected model.

## Project Layout

```text
sleep-stage-prediction/
  docs/
    current_data_audit.md          ds003768 label/data audit
    openneuro_dataset_research.md  external dataset log and experiment notes
    sota_sleep_staging_research.md literature/GitHub direction check
  legacy/
    kaggle/                        old Kaggle artifacts, not active workflow
  outputs/                         generated features, metrics, plots, models
  scripts/                         CLI entrypoints
  src/sleep_stage_prediction/      reusable package code
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

Optional dependencies:

```bash
pip install -e .[gbm]      # xgboost, lightgbm, catboost
pip install -e .[tuning]   # optuna
pip install -e .[deep]     # torch
```

## Core Workflows

### Extract `ds003768` band-power features

```bash
python scripts\extract_features.py --bids-root ..\ds003768 --output-dir outputs
```

### Audit `ds003768`

```bash
python scripts\audit_dataset.py --bids-root ..\ds003768
```

### Download and extract `ds006695`

Single subject smoke test:

```bash
python scripts\download_openneuro_subset.py --dataset ds006695 --target-dir ..\ds006695 --subject 126
python scripts\extract_external_bids_sleep.py --bids-root ..\ds006695 --dataset-id ds006695 --preset ds006695 --output-dir outputs\ds006695_features
```

All extracted `ds006695` subjects are currently:

```text
101, 102, 104, 105, 106, 107, 109, 110, 111, 112,
114, 116, 117, 119, 122, 123, 124, 125, 126
```

### Build the all-19 tabular benchmark

```bash
python scripts\augment_feature_set.py --features-path outputs\ds006695_features_all19\X_features.npy --labels-path outputs\ds006695_features_all19\y_labels.npy --metadata-path outputs\ds006695_features_all19\epoch_metadata.csv --output-dir outputs\ds006695_augmented_all19

python scripts\make_balanced_subset.py --features-path outputs\ds006695_augmented_all19\X_features.npy --labels-path outputs\ds006695_augmented_all19\y_labels.npy --metadata-path outputs\ds006695_augmented_all19\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19 --target-per-class 1600

python scripts\normalize_feature_set.py --features-path outputs\ds006695_augmented_balanced_1600_all19\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized
```

### Add sequence or transition context

```bash
python scripts\add_sequence_context.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1 --window 1 --include-deltas

python scripts\add_transition_features.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6 --radii 2,4,6
```

### Train/evaluate group-aware tabular models

```bash
python scripts\compare_group_models.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6_rf --n-splits 5 --models random_forest
```

External GBMs:

```bash
python scripts\compare_group_models.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_r2_4_6_external_gbms --n-splits 5 --models xgboost,lightgbm,catboost
```

### Run N1-focused analysis

```bash
python scripts\evaluate_n1_focus.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1_n1_focus --n-splits 5

python scripts\analyze_n1_errors.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_seq1_n1_errors --n-splits 5
```

### Tune Random Forest

```bash
python scripts\tune_random_forest_optuna.py --features-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition\X_features.npy --labels-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition\y_labels.npy --metadata-path outputs\ds006695_augmented_balanced_1600_all19_normalized_transition\epoch_metadata.csv --output-dir outputs\ds006695_augmented_balanced_1600_all19_normalized_transition_rf_optuna_fast10 --n-splits 5 --n-trials 10 --objective combined --combined-n1-weight 0.35 --search-space fast
```

Optuna improved N1-sensitive Random Forest variants, but did not beat the
untuned transition-feature Random Forest on the main balanced-accuracy
criterion.

### Summarize model selection

```bash
python scripts\summarize_model_selection.py --outputs-root outputs --output-dir outputs\model_selection_summary
```

Main output:

- `outputs/model_selection_summary/model_selection_summary.md`
- `outputs/model_selection_summary/model_selection_main_candidates.csv`
- `outputs/model_selection_summary/model_selection_all_results.csv`

### Extract spectrograms and train CNN-GRU

```bash
python scripts\extract_ds006695_spectrograms.py --bids-root ..\ds006695 --output-dir outputs\ds006695_spectrograms_all19

python scripts\train_spectrogram_sequence_model.py --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv --output-dir outputs\ds006695_spectrograms_all19_cnn_gru --n-splits 5 --epochs 8 --batch-size 96 --sequence-radius 2
```

The current smoke run used 5-epoch sequences (`sequence_radius=2`) and has
completed two subject-wise folds in `outputs\ds006695_spectrograms_all19_cnn_gru_smoke`.

## Output Conventions

Generated outputs usually include:

- `X_features.npy` or `X_spectrograms.npy`
- `y_labels.npy`
- `epoch_metadata.csv`
- `metrics.json`
- model comparison CSV/JSON files
- audit JSON files for extraction, balancing, normalization, or feature creation
- optional `model_cv_predictions.csv` / `cv_predictions.csv`

`outputs/` is generated experiment state, not source code.

## Thesis-Friendly Takeaways

- `ds003768` is a valid 4-class dataset for Wake/NREM work, but not a complete
  5-class REM-inclusive sleep-staging benchmark.
- `ds006695` provides the missing REM class and enough N1 to build a balanced
  5-class benchmark, but subject-aware validation is essential.
- Random row-wise CV is too optimistic for this problem.
- Transition-aware tabular features are the best completed classical baseline.
- N1 remains the hardest class; threshold tuning can improve recall but changes
  the objective.
- Spectrogram sequence modeling is the most promising next direction and is
  already outperforming the tabular baseline in the current partial run.

## Further Notes

- See `docs/current_data_audit.md` for the detailed `ds003768` audit.
- See `docs/openneuro_dataset_research.md` for dataset selection, integration
  history, and experiment chronology.
- See `docs/sota_sleep_staging_research.md` for why stronger sleep-staging
  systems usually move toward raw/spectrogram sequence models.
