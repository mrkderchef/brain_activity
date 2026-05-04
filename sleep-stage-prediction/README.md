# Sleep Stage Prediction

![Haerin Kang should be named Haerin Kamm](docs/assets/haerin-kamm-gag.png)

Local-first EEG sleep-stage classification pipeline. The project started with
`ds003768` and has grown into a more thesis-friendly benchmark workflow around
external sleep data, subject-aware validation, baseline model selection, and a
first spectrogram sequence-model experiment.

## Current Status

The strongest completed raw model is now a CNN-GRU over longer log-spectrogram
sequence windows. A fold-aware Viterbi smoothing pass gives the strongest
postprocessed result:

| Model | Data | Input | Evaluation | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---|---:|---:|---:|---:|
| CNN-GRU + Viterbi | `ds006695`, 19 subjects | 13-epoch log-spectrogram sequences, channel-normalized | 5-fold subject-wise CV + fold-aware smoothing | `0.6628` | `0.6222` | `0.6084` | `0.2736` |
| CNN-GRU | `ds006695`, 19 subjects | 13-epoch log-spectrogram sequences, channel-normalized | 5-fold subject-wise CV | `0.6501` | `0.6187` | `0.6023` | `0.2839` |

The strongest completed tabular baseline remains:

| Model | Data | Feature set | Evaluation | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---|---:|---:|---:|---:|
| Random Forest | `ds006695`, 19 subjects | transition features, radii `2,4,6` | 5-fold subject-wise CV | `0.5465` | `0.5465` | `0.5344` | `0.2474` |

Decision: the longer-context CNN-GRU is now the best raw model by subject-wise
balanced accuracy. The fold-aware Viterbi result is the best postprocessed
sensitivity result. The Random Forest is still useful as a classical,
interpretable baseline, but the project should now treat spectrogram sequence
modeling as the main performance path.

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

### Spectrogram sequence models broke through the tabular ceiling

The literature and GitHub survey pointed in the same direction: stronger
sleep-staging systems usually use raw EEG/EOG/PSG or spectrogram sequences,
not only per-epoch band-power summary features.

The first full CNN-GRU run on `ds006695` spectrogram sequences used 5-epoch
windows and completed all five subject-wise folds:

| Metric | Value |
|---|---:|
| Accuracy | `0.6432` |
| Balanced accuracy | `0.6079` |
| Macro F1 | `0.5964` |
| Cohen's kappa | `0.5318` |
| N1 recall | `0.3306` |
| N1 F1 | `0.2716` |

Fold-level results:

| Fold | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---:|---:|---:|---:|---:|
| 1 | `0.6402` | `0.6300` | `0.6136` | `0.3114` |
| 2 | `0.7088` | `0.6208` | `0.6269` | `0.1677` |
| 3 | `0.5935` | `0.5632` | `0.5242` | `0.1072` |
| 4 | `0.6161` | `0.5783` | `0.5619` | `0.2578` |
| 5 | `0.6479` | `0.6279` | `0.6114` | `0.3616` |

Decision: the spectrogram model is clearly better than the tabular RF baseline
on overall metrics. Fold 3 and the N1 variance show that it is not solved yet,
but this is the first modeling direction that moves the project meaningfully
above the mid-50s balanced-accuracy ceiling.

The next improvement passes added fold-wise channel normalization and expanded
the sequence context first to 9 epochs (`sequence_radius=4`), then to 13 epochs
(`sequence_radius=6`) in the unattended accuracy marathon:

| Model | Context | Normalization | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|---:|
| CNN-GRU | 5 epochs | none | `0.6432` | `0.6079` | `0.5964` | `0.2716` |
| CNN-GRU | 9 epochs | channel-wise from training fold | `0.6449` | `0.6143` | `0.5963` | `0.2592` |
| CNN-GRU | 13 epochs | channel-wise from training fold | `0.6501` | `0.6187` | `0.6023` | `0.2839` |
| CNN-GRU + Viterbi | 13 epochs | channel-wise from training fold | `0.6628` | `0.6222` | `0.6084` | `0.2736` |
| CNN-TCN smoke, first 2 folds | 9 epochs | channel-wise from training fold | `0.6240` | `0.5310` | `0.5451` | `0.2511` |

Decision: the normalized 13-epoch CNN-GRU is the best main raw model by
balanced accuracy. Viterbi smoothing adds a small postprocessing gain, but
should be described separately from the raw neural model.

### Seq2Seq and Transformer follow-up

The next sequence-to-sequence pass re-synchronized `ds006695` from OpenNeuro and
confirmed that the local dataset still contains the same 19 EEG subjects:

```text
101, 102, 104, 105, 106, 107, 109, 110, 111, 112,
114, 116, 117, 119, 122, 123, 124, 125, 126
```

The spectrogram tensor remains `19,563` epochs with shape `(3, 20, 116)` per
epoch. Longer Seq2Seq blocks, a larger hidden state, and a longer Transformer
marathon candidate did not improve over the windowed CNN-GRU:

| Model | Block setup | Params | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|---:|
| CNN-GRU Seq2Seq | length `32`, stride `16` | hidden `64`, 8 epochs | `0.6478` | `0.6049` | `0.5925` | `0.2556` |
| CNN-GRU Seq2Seq | length `32`, stride `16` | hidden `96`, up to 24 epochs | `0.6350` | `0.5954` | `0.5791` | `0.2238` |
| CNN-GRU Seq2Seq | length `64`, stride `32` | hidden `96`, up to 160 epochs | `0.6334` | `0.5880` | `0.5783` | `0.2244` |
| CNN-Transformer Seq2Seq | length `32`, stride `16` | hidden `96`, 2 layers, 4 heads | `0.6160` | `0.5849` | `0.5664` | `0.2199` |
| CNN-Transformer Seq2Seq | length `32`, stride `16` | hidden `128`, up to 48 epochs | `0.5930` | `0.5612` | `0.5355` | `0.1633` |

Decision: the simple Transformer encoder is technically working, but it did not
beat the CNN-GRU Seq2Seq baseline. The failure mode is still subject-level
instability and weak N1 precision, not lack of epoch count. Future gains should
come from better inputs, N1-specific objectives, augmentation, or transfer
learning rather than simply scaling this Seq2Seq architecture.

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

### Download and extract Sleep-EDF Expanded

Sleep-EDF is the next external benchmark/pretraining corpus. Start with two
records before downloading the full dataset:

```bash
python scripts\download_sleep_edf_subset.py --target-dir ..\sleep-edf --record SC4001E0 --record SC4011E0
```

Extract single-channel log-spectrogram tensors from the paired PSG and
hypnogram EDF files:

```bash
python scripts\extract_sleep_edf_spectrograms.py --sleep-edf-root ..\sleep-edf --output-dir outputs\sleep_edf_spectrogram_smoke --channel "EEG Fpz-Cz" --limit-recordings 2
```

For a quick shape-compatible experiment with the existing `ds006695`
spectrogram dimensions, resize the Sleep-EDF spectrogram grids during export:

```bash
python scripts\extract_sleep_edf_spectrograms.py --sleep-edf-root ..\sleep-edf --output-dir outputs\sleep_edf_spectrogram_smoke_ds006695_shape --channel "EEG Fpz-Cz" --target-freq-bins 20 --target-time-bins 116 --limit-recordings 2
```

Then run a subject-wise Sleep-EDF smoke model:

```bash
python scripts\train_spectrogram_sequence_model.py --spectrograms-path outputs\sleep_edf_spectrogram_smoke\X_spectrograms.npy --labels-path outputs\sleep_edf_spectrogram_smoke\y_labels.npy --metadata-path outputs\sleep_edf_spectrogram_smoke\epoch_metadata.csv --output-dir outputs\sleep_edf_cnn_gru_smoke --n-splits 2 --max-folds 2 --model cnn_gru --sequence-radius 6 --epochs 8 --batch-size 64 --hidden-size 80 --normalization channel
```

First fuller Sleep-EDF subset run:

```bash
python scripts\download_sleep_edf_subset.py --target-dir ..\sleep-edf --max-records 6
python scripts\extract_sleep_edf_spectrograms.py --sleep-edf-root ..\sleep-edf --output-dir outputs\sleep_edf_spectrogram_first6 --channel "EEG Fpz-Cz" --limit-recordings 6
python scripts\train_spectrogram_sequence_model.py --spectrograms-path outputs\sleep_edf_spectrogram_first6\X_spectrograms.npy --labels-path outputs\sleep_edf_spectrogram_first6\y_labels.npy --metadata-path outputs\sleep_edf_spectrogram_first6\epoch_metadata.csv --output-dir outputs\sleep_edf_first6_cnn_gru_r6_e16_full5 --n-splits 5 --model cnn_gru --sequence-radius 6 --epochs 16 --batch-size 64 --hidden-size 80 --dropout 0.35 --normalization channel --label-smoothing 0.05 --learning-rate 0.0007 --weight-decay 0.0002 --grad-clip-norm 1.0 --early-stopping-patience 7 --lr-plateau-patience 2 --seed 44
python scripts\smooth_predictions_viterbi.py --predictions-path outputs\sleep_edf_first6_cnn_gru_r6_e16_full5\cv_predictions.csv --output-dir outputs\sleep_edf_first6_cnn_gru_r6_e16_full5_viterbi
```

Completed first-six-record result:

| Dataset | Records | Model | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---:|---|---:|---:|---:|---:|
| Sleep-EDF Expanded | 6 | CNN-GRU, 13-epoch context | `0.7476` | `0.7712` | `0.7297` | `0.4815` |
| Sleep-EDF Expanded | 6 | CNN-GRU + Viterbi | n/a | `0.7735` | n/a | n/a |

Decision: this confirms the code path and the model architecture can reach a
much stronger benchmark score on a more standard PSG corpus. The weaker
`ds006695` result is therefore mainly a data/domain/input limitation, not proof
that the CNN-GRU sequence architecture is broken.

Expanded 20-record Sleep-EDF benchmark:

```bash
python scripts\download_sleep_edf_subset.py --target-dir ..\sleep-edf --max-records 20
python scripts\extract_sleep_edf_spectrograms.py --sleep-edf-root ..\sleep-edf --output-dir outputs\sleep_edf_spectrogram_first20 --channel "EEG Fpz-Cz" --limit-recordings 20
python scripts\train_spectrogram_sequence_model.py --spectrograms-path outputs\sleep_edf_spectrogram_first20\X_spectrograms.npy --labels-path outputs\sleep_edf_spectrogram_first20\y_labels.npy --metadata-path outputs\sleep_edf_spectrogram_first20\epoch_metadata.csv --output-dir outputs\sleep_edf_first20_cnn_gru_r6_e10_full5 --n-splits 5 --model cnn_gru --sequence-radius 6 --epochs 10 --batch-size 96 --hidden-size 80 --dropout 0.35 --normalization channel --label-smoothing 0.05 --learning-rate 0.0007 --weight-decay 0.0002 --grad-clip-norm 1.0 --early-stopping-patience 5 --lr-plateau-patience 2 --seed 44
python scripts\analyze_sleep_edf_results.py --predictions-path outputs\sleep_edf_first20_cnn_gru_r6_e10_full5\cv_predictions.csv --output-dir outputs\sleep_edf_first20_cnn_gru_r6_e10_full5_analysis
```

Completed first-20-record result:

| Dataset | Records | Epochs | Model | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---:|---:|---|---:|---:|---:|---:|
| Sleep-EDF Expanded | 20 | `21,039` | CNN-GRU, 13-epoch context | `0.8337` | `0.8198` | `0.7968` | `0.5166` |
| Sleep-EDF Expanded | 20 | `21,039` | CNN-GRU + Viterbi | n/a | `0.8198` | n/a | n/a |

Fold-level range:

| Fold | Balanced accuracy | Macro F1 | N1 F1 |
|---:|---:|---:|---:|
| 1 | `0.8665` | `0.8218` | `0.5470` |
| 2 | `0.8112` | `0.7824` | `0.5213` |
| 3 | `0.8221` | `0.7980` | `0.5393` |
| 4 | `0.8406` | `0.7959` | `0.5000` |
| 5 | `0.7926` | `0.7766` | `0.4715` |

The record-level analysis is in
`outputs\sleep_edf_first20_cnn_gru_r6_e10_full5_analysis\sleep_edf_result_analysis.md`.
The weakest held-out record was `SC4011` with balanced accuracy `0.7193`; the
strongest was `SC4061` with balanced accuracy `0.9262`. N1 remains the weak
class, but improves substantially compared with `ds006695`.

### Cross-dataset transfer checks

Direct transfer was tested after resizing Sleep-EDF spectrograms to the
`ds006695` grid and averaging `ds006695` channels to one channel:

```bash
python scripts\extract_sleep_edf_spectrograms.py --sleep-edf-root ..\sleep-edf --output-dir outputs\sleep_edf_spectrogram_first20_ds006695_shape --channel "EEG Fpz-Cz" --target-freq-bins 20 --target-time-bins 116 --limit-recordings 20
python scripts\average_spectrogram_channels.py --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv --output-dir outputs\ds006695_spectrograms_all19_channelmean
```

| Train source | Test target | Target balanced accuracy | Target macro F1 | Target N1 F1 |
|---|---|---:|---:|---:|
| Sleep-EDF first20 | `ds006695` channel mean | `0.3438` | `0.2864` | `0.1796` |
| `ds006695` channel mean | Sleep-EDF first20 | `0.3792` | `0.3502` | `0.1162` |

Decision: direct cross-dataset transfer is still poor even after matching the
spectrogram tensor shape. This supports the report claim that the model learns
useful within-corpus sleep-stage structure, but Sleep-EDF PSG and `ds006695`
forehead EEG differ enough that transfer needs domain adaptation or fine-tuning,
not naive source-only training.

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

### Extract spectrograms and train CNN-GRU / Seq2Seq

```bash
python scripts\extract_ds006695_spectrograms.py --bids-root ..\ds006695 --output-dir outputs\ds006695_spectrograms_all19

python scripts\train_spectrogram_sequence_model.py --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv --output-dir outputs\ds006695_spectrograms_all19_cnn_gru_norm_r4_e20_full5 --n-splits 5 --epochs 20 --batch-size 96 --sequence-radius 4 --model cnn_gru --normalization channel --early-stopping-patience 5
```

Seq2Seq block models:

```bash
python scripts\train_spectrogram_seq2seq_model.py --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv --output-dir outputs\ds006695_spectrograms_all19_cnn_gru_seq2seq_b32_s16_e8_full5 --model cnn_gru --block-length 32 --block-stride 16 --n-splits 5 --epochs 8 --batch-size 16 --hidden-size 64

python scripts\train_spectrogram_seq2seq_model.py --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv --output-dir outputs\ds006695_spectrograms_all19_cnn_transformer_seq2seq_b32_s16_h96_l2_e40_full5 --model cnn_transformer --block-length 32 --block-stride 16 --n-splits 5 --epochs 40 --batch-size 16 --hidden-size 96 --transformer-heads 4 --transformer-layers 2 --dropout 0.25 --early-stopping-patience 8 --lr-plateau-patience 3
```

The current best raw full run uses 13-epoch sequences (`sequence_radius=6`) with
channel-wise train-fold normalization and writes results to
`outputs\accuracy_marathon\exp_03_gru_r6_long_context`. Its best Viterbi
postprocessing sweep is in
`outputs\accuracy_marathon\exp_03_gru_r6_long_context_viterbi`.

An SE/residual follow-up model is available as `cnn_gru_se`. It is a practical
local adaptation of the strongest public sleep-staging direction found in the
latest survey: U-Sleep/SE-Res-U-Net-style convolutional feature extraction, but
fitted to this repo's saved spectrogram windows. It uses multi-scale
spectrogram convolutions, residual squeeze-excitation blocks, BiGRU temporal
context, and attention pooling over the 13-epoch window:

```bash
python scripts\train_spectrogram_sequence_model.py --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv --output-dir outputs\ds006695_spectrograms_all19_cnn_gru_se_r6_e18_full5 --n-splits 5 --epochs 18 --batch-size 64 --sequence-radius 6 --model cnn_gru_se --hidden-size 96 --dropout 0.35 --normalization robust_channel --normalization-clip 6.0 --label-smoothing 0.05 --grad-clip-norm 1.0 --early-stopping-patience 7
```

The older `cnn_gru_attention` option is still available as a lighter ablation:
it keeps the center-epoch GRU representation, but also learns an
attention-pooled summary over neighboring epochs.

Completed `cnn_gru_se` result:

| Model | Context | Normalization | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|---:|
| CNN-GRU-SE | 13 epochs | robust channel, clipped | `0.5960` | `0.5687` | `0.5577` | `0.2495` |

Decision: do not switch the main model to `cnn_gru_se`. The stronger encoder
looked promising from the literature, but generalized worse across held-out
subjects than the simpler 13-epoch CNN-GRU. Treat it as a useful negative
result showing that extra convolutional capacity is not the current bottleneck.

For a longer unattended search, run the accuracy marathon. It executes the full
5-fold candidates sequentially, then applies fold-aware Viterbi smoothing to
each finished model and keeps a live ranking in
`outputs\accuracy_marathon\accuracy_experiment_summary.md`. A committed summary
of the completed run is in `docs\accuracy_marathon_summary.md`:

```powershell
.\scripts\start_accuracy_experiments.ps1
```

If local PowerShell scripts are blocked on Windows, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_accuracy_experiments.ps1
```

Foreground/resume form:

```bash
python scripts\run_accuracy_experiments.py --root-dir outputs\accuracy_marathon
python scripts\run_accuracy_experiments.py --root-dir outputs\accuracy_marathon --start-at exp_05_gru_aug_r4
python scripts\summarize_accuracy_experiments.py --root-dir outputs\accuracy_marathon
```

The spectrogram training script now uses an inner group-aware validation split
inside each outer subject-wise fold. It saves the best checkpoint per fold under
`checkpoints\`, writes epoch-level training curves to `training_history.json`,
and includes class probabilities, prediction confidence, prediction entropy,
and fold IDs in `cv_predictions.csv`.

### Analyze CV predictions by subject

```bash
python scripts\analyze_cv_predictions.py --predictions-path outputs\ds006695_spectrograms_all19_cnn_gru_norm_r4_e20_full5\cv_predictions.csv --output-dir outputs\ds006695_spectrograms_all19_cnn_gru_norm_r4_e20_full5_analysis
```

Main outputs:

- `prediction_analysis.md`
- `subject_metrics.csv`
- `fold_metrics_from_predictions.csv`
- `confusion_summary.csv`
- `confidence_summary.csv`

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
- Spectrogram sequence modeling is now the best completed direction. The best
  raw model outperforms the tabular baseline by roughly `+0.0722` balanced
  accuracy; the Viterbi-smoothed result raises that gap to roughly `+0.0757`.
- The latest marathon suggests longer temporal context helped more than
  attention pooling, light augmentation, N1 weighting, balanced sampling, or
  the tested Seq2Seq/Transformer variants.
- The next likely gains should come from better inputs, stronger N1-specific
  objectives, more principled temporal postprocessing, or transfer learning
  from Sleep-EDF, not from more Random Forest tuning or simply scaling epoch
  count.

## Further Notes

- See `docs/current_data_audit.md` for the detailed `ds003768` audit.
- See `docs/openneuro_dataset_research.md` for dataset selection, integration
  history, and experiment chronology.
- See `docs/sota_sleep_staging_research.md` for why stronger sleep-staging
  systems usually move toward raw/spectrogram sequence models.
- See `docs/accuracy_marathon_summary.md` for the completed 10-run spectrogram
  marathon and final ranking.
