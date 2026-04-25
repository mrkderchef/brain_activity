# OpenNeuro Dataset Research

Research date: 2026-04-25

## Context

The current project uses `ds003768` for EEG sleep-stage prediction. The local audit shows that the practical label space is `Wake`, `N1`, `N2`, and `N3`; REM is absent from the source labels. The main data problem is therefore not only sample size, but label coverage and class imbalance.

## Recommendation

Adding more OpenNeuro data can make sense, but not as a simple row-level merge into the current training set. The safest path is:

1. Keep `ds003768` as the primary dataset for the current 4-class experiment.
2. Add one external sleep-staging dataset as a separate corpus, normalize labels and feature extraction, then evaluate cross-dataset generalization.
3. Only merge datasets after checking whether performance improves under subject-aware and dataset-aware validation.

For the current goal, the best first candidates are:

| Priority | Dataset | Why it is useful | Main caveat |
|---|---|---|---|
| 1 | `NM000185` / Sleep-EDF Expanded | Large whole-night PSG corpus with expert 30-second labels including Wake, N1, N2, N3, REM, Unknown | Not CC0; ODbL license. Different montage and 100 Hz sampling. Source is NeMAR/PhysioNet rather than native OpenNeuro. |
| 2 | `ds006695` | Human sleep staging with manual 30-second hypnogram including Wake, REM, N1, N2, N3; modest size and BIDS | Only 3 forehead EEG channels, so it differs strongly from the 32-channel MR-compatible EEG setup in `ds003768`. |
| 3 | `ds004348` | Human overnight ear-EEG/partial PSG sleep data; manageable size; healthy participants | Ear-EEG montage differs from scalp EEG. Need inspect scoring files before merging. |
| 4 | `ds005178` | Large ear-EEG sleep monitoring dataset, 1012+ hours reported by EEGDash | Ear-EEG, not scalp EEG; may be better for pretraining/robustness than direct pooling. |
| 5 | `ds004902` | Resting-state EEG under normal sleep vs sleep deprivation; 71 participants, 61 channels | Not a sleep-stage dataset; useful for wake/rest/sleepiness features, not REM/N stages. |

## Candidate Notes

### `NM000185` / Sleep-EDF Expanded

Best candidate if the goal is to add REM and whole-night sleep architecture. EEGDash lists 100 subjects, 197 recordings, 3849 hours, 100 Hz sampling, and 5 or 7 channels. The dataset has expert-scored 30-second epochs with Wake, N1, N2, N3, REM, and Unknown. This is ideal for a true 5-class sleep-stage benchmark, but it is a domain shift from `ds003768` because the montage is Fpz-Cz/Pz-Oz-style PSG rather than 32-channel MR-compatible EEG. It also uses ODbL rather than CC0, so attribution and license notes matter.

### `ds006695`

Strong human sleep-staging candidate. EEGDash reports 19 subjects, 19 recordings, 164.3 hours, 3 channels, 500 Hz sampling, and BIDS format. The README states that `EEG.VisualHypnogram` is manual 30-second scoring with integer labels for Wake, REM, N1, N2, N3, and unknown/movement. This directly covers the REM gap. The downside is montage mismatch: a 3-channel forehead patch is not comparable to the current 32-channel scalp/MR-compatible EEG unless the pipeline uses channel-agnostic frequency-band features.

### `ds004348`

Good manageable sleep dataset from ear-EEG monitoring. EEGDash lists 9 subjects, 113 files, 34 channels, 200 Hz sampling, and 8.2 GB. It contains nightly recordings from healthy participants with partial PSG plus ear-EEG. It is promising for an additional sleep corpus, but the scoring file format and label mapping should be audited before use.

### `ds005178`

Large ear-EEG sleep monitoring dataset. EEGDash lists 10 subjects, 140 recordings, 1012.5 hours, 4 or 13 channels, 250 Hz, and 25.7 GB. This could be very useful if the feature pipeline is designed to be montage-independent. Because it is ear-EEG, it should be treated as cross-device data rather than the same distribution as `ds003768`.

### `ds004902`

Useful, but not for direct sleep-stage labels. It contains resting-state EEG from 71 participants after normal sleep and sleep deprivation, with sleepiness/mood metadata. It has 61 channels and mostly 500 Hz sampling. This could support a wake/sleepiness analysis or pretraining, but it will not solve missing REM/N3 in the current classifier.

## Practical Integration Plan

1. Add a dataset abstraction that stores `dataset_id`, `subject`, `recording`, `stage`, `epoch_start`, and feature rows.
2. Convert labels into a shared schema: `Wake`, `N1`, `N2`, `N3`, `REM`, `Unknown`.
3. Use only common, robust feature families first: delta, theta, alpha, sigma, beta band power; optionally ratios.
4. Avoid relying on exact channel names across datasets. Start with per-epoch summary features across available EEG channels.
5. Validate with three splits:
   - within-dataset subject split
   - train on `ds003768`, test on external dataset labels shared with it
   - train on external dataset, test on `ds003768`
6. Report merged results only if dataset-aware validation improves macro-F1 or balanced accuracy, not only raw accuracy.

## Bottom Line

Yes, more datasets probably help, especially for REM and stable N3 estimates. But the current project should not simply append external epochs to `ds003768`. The next best experiment is to integrate `NM000185` or `ds006695` behind the same feature extractor, keep dataset identity in the metadata, and compare within-dataset vs cross-dataset performance.

## First Integration Result

Started on 2026-04-25 with `ds006695`, subject `126`, as a small OpenNeuro smoke test.

Downloaded files:

- `../ds006695/sub-126/eeg/sub-126_task-sleep_eeg.set`
- `../ds006695/sub-126/eeg/sub-126_task-sleep_eeg.fdt`
- sidecar metadata files

Extraction result:

| Output | Shape / count |
|---|---:|
| `outputs/ds006695_features/X_features.npy` | `(1034, 15)` |
| `outputs/ds006695_features/y_labels.npy` | `(1034,)` |

Extracted class distribution for `ds006695` subject `126`:

| Stage | Count |
|---|---:|
| Wake | 217 |
| N1 | 223 |
| N2 | 350 |
| N3 | 130 |
| REM | 114 |

Combined with the existing `ds003768` features:

| Dataset | Epochs |
|---|---:|
| `ds003768` | 6773 |
| `ds006695` subject `126` | 1034 |
| Total | 7807 |

A quick 3-fold stratified mixed-dataset Random Forest smoke test completed successfully:

| Metric | Value |
|---|---:|
| Accuracy | 0.5893 |
| Cohen's kappa | 0.3773 |
| Macro F1 | 0.57 |

Interpretation: the integration works technically and adds real REM labels. The mixed-dataset score should not be treated as the final result yet; the next validation step should split by dataset and subject to measure cross-dataset generalization.

Cross-dataset transfer check on the shared 4-class label space (`Wake`, `N1`, `N2`, `N3`) showed strong domain shift:

| Train dataset | Test dataset | Test epochs | Accuracy | Balanced accuracy | Kappa |
|---|---|---:|---:|---:|---:|
| `ds003768` | `ds006695` subject `126` | 920 | 0.2424 | 0.2500 | 0.0000 |
| `ds006695` subject `126` | `ds003768` | 6773 | 0.1874 | 0.2507 | 0.0007 |

Interpretation: direct cross-dataset transfer is essentially chance-level with the current simple band-power feature space. `ds006695` is still useful because it adds REM and more N3, but it should be used with dataset-aware reporting, domain adaptation, or as a separate external-validation experiment rather than as an unqualified pooled dataset.

## Balanced Pool Update

Second collection pass on 2026-04-25 downloaded and extracted `ds006695` subjects:

- `101`
- `102`
- `104`
- `105`
- `126`

Combined pool after adding these subjects:

| Stage | Count |
|---|---:|
| Wake | 4091 |
| N1 | 2617 |
| N2 | 3546 |
| N3 | 958 |
| REM | 949 |

This is enough to create a balanced 5-class subset capped at 800 epochs per class.

Balanced output:

- `outputs/balanced_800_ds003768_ds006695_batch1/X_features.npy`
- `outputs/balanced_800_ds003768_ds006695_batch1/y_labels.npy`
- `outputs/balanced_800_ds003768_ds006695_batch1/epoch_metadata.csv`

Balanced class distribution:

| Stage | Count |
|---|---:|
| Wake | 800 |
| N1 | 800 |
| N2 | 800 |
| N3 | 800 |
| REM | 800 |

Dataset contribution by class:

| Stage | `ds003768` | `ds006695` |
|---|---:|---:|
| Wake | 646 | 154 |
| N1 | 653 | 147 |
| N2 | 290 | 510 |
| N3 | 37 | 763 |
| REM | 0 | 800 |

Quick 3-fold stratified Random Forest smoke test on the balanced 800-per-class subset:

| Metric | Value |
|---|---:|
| Accuracy | 0.6290 |
| Cohen's kappa | 0.5363 |
| Macro F1 | 0.62 |

Important caveat: REM is entirely sourced from `ds006695`, and N3 is mostly sourced from `ds006695`. This balanced subset is useful for building a 5-class classifier, but thesis reporting should include dataset-aware validation to avoid confusing dataset identity with sleep-stage signal.

## Modeling Improvement Pass

Started after the first balanced-pool experiment:

1. Added feature augmentation:
   - delta/theta ratio
   - alpha/theta ratio
   - sigma/delta ratio
   - beta/alpha ratio
   - slow/fast ratio
   - relative band entropy
2. Added recording/subject-aware robust normalization:
   - `log1p` transform for absolute mean/std bandpower columns
   - median/IQR scaling per derived recording/subject group
3. Added group-aware validation with `StratifiedGroupKFold`.

Results on the 800-per-class mixed pool:

| Feature set / validation | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Original, random 3-fold CV | 15 | 0.6290 | n/a | 0.5363 | 0.62 |
| Augmented, random 3-fold CV | 21 | 0.6322 | n/a | 0.5403 | 0.63 |
| Augmented + normalized, random 3-fold CV | 21 | 0.6593 | n/a | 0.5741 | 0.65 |
| Augmented + normalized, group 3-fold CV | 21 | 0.3738 | 0.3738 | 0.2172 | 0.29 |

Interpretation: feature augmentation and normalization improve random CV, but group-aware CV stays much lower. This is a useful warning: random splits are still too optimistic when classes are strongly tied to dataset/source.

Results on `ds006695` only, balanced to 400 per class across five subjects:

| Feature set / validation | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Augmented, subject-wise 5-fold group CV | 21 | 0.4215 | 0.4215 | 0.2769 | 0.40 |
| Augmented + normalized, subject-wise 5-fold group CV | 21 | 0.4560 | 0.4560 | 0.3200 | 0.43 |

Interpretation: normalization helps under subject-wise validation. The hardest class remains N1, which is expected because N1 is a transitional and often ambiguous sleep stage.

## Sequence Context Pass

Added neighboring-epoch context features:

- previous epoch features
- current epoch features
- next epoch features
- previous-to-current deltas
- current-to-next deltas

With 21 base/augmented features this creates 105 features per epoch for a `window=1` context.

Results on `ds006695` only, balanced to 400 per class across five subjects:

| Feature set / validation | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Augmented + normalized, subject-wise 5-fold group CV | 21 | 0.4560 | 0.4560 | 0.3200 | 0.43 |
| Sequence context + augmented + normalized, subject-wise 5-fold group CV | 105 | 0.5045 | 0.5045 | 0.3806 | 0.48 |

Results on the 800-per-class mixed pool:

| Feature set / validation | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Augmented + normalized, group 3-fold CV | 21 | 0.3738 | 0.3738 | 0.2172 | 0.29 |
| Sequence context + augmented + normalized, group 3-fold CV | 105 | 0.3952 | 0.3952 | 0.2441 | 0.30 |

Interpretation: temporal context helps under honest group-aware validation. The gain is clearer inside `ds006695` than in the mixed pool, because the mixed pool still has class/dataset confounding: REM is only from `ds006695`, and `ds003768` metadata is incomplete for subject-level grouping.

## Full `ds006695` Collection

Downloaded and extracted all 19 available `ds006695` subjects:

- `101`
- `102`
- `104`
- `105`
- `106`
- `107`
- `109`
- `110`
- `111`
- `112`
- `114`
- `116`
- `117`
- `119`
- `122`
- `123`
- `124`
- `125`
- `126`

Full extracted pool:

| Stage | Count |
|---|---:|
| Wake | 3061 |
| N1 | 1600 |
| N2 | 7691 |
| N3 | 3737 |
| REM | 3474 |

Because N1 is the rarest class, the clean balanced cap is now 1600 epochs per class:

| Stage | Count |
|---|---:|
| Wake | 1600 |
| N1 | 1600 |
| N2 | 1600 |
| N3 | 1600 |
| REM | 1600 |

Balanced all-19 output:

- `outputs/ds006695_augmented_balanced_1600_all19`
- `outputs/ds006695_augmented_balanced_1600_all19_normalized`
- `outputs/ds006695_augmented_balanced_1600_all19_normalized_seq1`

Subject-wise 5-fold group CV on all 19 `ds006695` subjects:

| Feature set / validation | Features | Subjects | Samples | Accuracy | Balanced accuracy | Kappa | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Augmented + normalized | 21 | 19 | 8000 | 0.4866 | 0.4866 | 0.3583 | 0.48 |
| Sequence context + augmented + normalized | 105 | 19 | 8000 | 0.5394 | 0.5394 | 0.4242 | 0.52 |

Interpretation: adding the remaining `ds006695` subjects materially improves the honest subject-wise benchmark. Sequence context remains useful at the larger scale. N1 is still the limiting class, with recall around 0.19 in the current Random Forest setup.

## N1-Focused Evaluation

Added an N1-focused evaluation script:

- `scripts/evaluate_n1_focus.py`

The script keeps subject-wise `StratifiedGroupKFold`, then compares:

- explicit N1 class upweighting
- post-hoc N1 probability thresholds

Run on:

- `outputs/ds006695_augmented_balanced_1600_all19_normalized_seq1`

Output:

- `outputs/ds006695_augmented_balanced_1600_all19_normalized_seq1_n1_focus/n1_focus_summary.csv`
- `outputs/ds006695_augmented_balanced_1600_all19_normalized_seq1_n1_focus/n1_focus_metrics.json`

Best result by N1 F1:

| Strategy | N1 weight | N1 threshold | Accuracy | Balanced accuracy | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 threshold | 2.0 | 0.20 | 0.4319 | 0.4319 | 0.4188 | 0.2643 | 0.7906 | 0.3962 |

Best result by balanced accuracy in the N1-focused sweep:

| Strategy | N1 weight | N1 threshold | Accuracy | Balanced accuracy | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Argmax | 1.0 | n/a | 0.5411 | 0.5411 | 0.5266 | 0.3191 | 0.1969 | 0.2435 |

Interpretation: lowering the N1 decision threshold can substantially increase N1 recall, from roughly `0.19` to `0.79`, but this reduces balanced accuracy from roughly `0.54` to `0.43`. This is not a better default classifier yet; it is a useful sensitivity analysis if the thesis wants to emphasize detecting N1 rather than optimizing balanced 5-class performance.

## N1 Error Analysis and Transition Features

Added three follow-up scripts:

- `scripts/analyze_n1_errors.py`
- `scripts/add_transition_features.py`
- `scripts/compare_group_models.py`

The N1 error analysis runs the same subject-wise `StratifiedGroupKFold` as the model benchmark and saves per-epoch predictions with true neighboring-stage context. On the previous best sequence-context feature set, N1 errors were:

| True N1 outcome | Count |
|---|---:|
| Correctly predicted N1 | 308 |
| False negatives | 1292 |

N1 false negatives were predicted as:

| Predicted stage | Count |
|---|---:|
| N2 | 451 |
| REM | 435 |
| Wake | 286 |
| N3 | 120 |

This confirms that N1 is not only rare; it is being confused with both neighboring NREM sleep and REM-like low-amplitude/transitional patterns.

The transition-feature script adds label-free rolling features within each subject/recording:

- centered 5-epoch and 9-epoch rolling means
- centered rolling standard deviations
- centered rolling slopes
- current-minus-local-mean deviations

These features expand the all-19 normalized `ds006695` benchmark from `21` to `189` features per epoch.

Subject-wise 5-fold model comparison:

| Feature set | Model | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequence context | Random Forest | 105 | 0.5394 | 0.5394 | 0.4242 | 0.5248 | 0.3136 | 0.1925 | 0.2386 |
| Sequence context | Extra Trees | 105 | 0.5394 | 0.5394 | 0.4242 | 0.5177 | 0.3096 | 0.1494 | 0.2015 |
| Sequence context | Gradient Boosting | 105 | 0.5223 | 0.5223 | 0.4028 | 0.5146 | 0.2874 | 0.2344 | 0.2582 |
| Transition features | Random Forest | 189 | 0.5458 | 0.5458 | 0.4322 | 0.5332 | 0.3102 | 0.2038 | 0.2459 |
| Transition features | Extra Trees | 189 | 0.5364 | 0.5364 | 0.4205 | 0.5186 | 0.2834 | 0.1563 | 0.2015 |
| Transition features | Fast Gradient Boosting | 189 | 0.5281 | 0.5281 | 0.4102 | 0.5186 | 0.2805 | 0.2081 | 0.2390 |
| Sequence context | Fast HistGradientBoosting | 105 | 0.5189 | 0.5189 | 0.3986 | 0.5100 | 0.2714 | 0.2100 | 0.2368 |
| Transition features | Fast HistGradientBoosting | 189 | 0.5288 | 0.5288 | 0.4109 | 0.5234 | 0.2802 | 0.2363 | 0.2564 |

Best default after this pass:

- Transition features + Random Forest
- Balanced accuracy: `0.5458`
- Macro F1: `0.5332`
- N1 recall: `0.2038`
- N1 F1: `0.2459`

Interpretation: transition features give a small but honest improvement over the previous sequence-context Random Forest. However, N1 remains the limiting class. The best N1 F1 still comes from explicit threshold tuning, while the best balanced 5-class model is now the transition-feature Random Forest.

Additional note: fast `HistGradientBoostingClassifier` is now supported in `scripts/compare_group_models.py`. It improved N1 recall on the transition features compared with Random Forest (`0.2363` vs `0.2038`) but did not beat Random Forest on balanced 5-class performance.

### External Gradient-Boosting Models

Added optional support for:

- XGBoost: `xgboost`
- LightGBM: `lightgbm`
- CatBoost: `catboost`

These can be installed with:

```bash
pip install -e .[gbm]
```

Subject-wise 5-fold comparison on transition features:

| Feature set | Model | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transition features | XGBoost | 189 | 0.5328 | 0.5328 | 0.4159 | 0.5248 | 0.2916 | 0.2256 | 0.2544 |
| Transition features | LightGBM | 189 | 0.5320 | 0.5320 | 0.4150 | 0.5278 | 0.2916 | 0.2519 | 0.2703 |
| Transition features | CatBoost | 189 | 0.5435 | 0.5435 | 0.4294 | 0.5320 | 0.3037 | 0.2025 | 0.2430 |

LightGBM was also checked on the previous sequence-context feature set:

| Feature set | Model | Features | Accuracy | Balanced accuracy | Kappa | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequence context | LightGBM | 105 | 0.5228 | 0.5228 | 0.4034 | 0.5171 | 0.2789 | 0.2369 | 0.2562 |

Interpretation: CatBoost is closest to Random Forest on balanced 5-class performance, but LightGBM gives the best N1 F1 so far without post-hoc thresholding (`0.2703`). The best overall balanced model remains transition-feature Random Forest (`0.5458`), while the best N1-sensitive non-threshold model is transition-feature LightGBM.

### Random Forest Optuna Tuning

Added:

- `scripts/tune_random_forest_optuna.py`

The tuner uses Optuna with the same subject-wise `StratifiedGroupKFold` validation. It supports three objectives:

- `balanced_accuracy`
- `n1_f1`
- `combined`, defined as a weighted mix of balanced accuracy and N1 F1

The first broad search was too expensive for this dataset: a wide `max_features`/tree-count search only completed two trials in about 30 minutes. The script now defaults to a `fast` search space and writes `optuna_trials_partial.csv` after every trial so long runs leave recoverable results.

Fast 10-trial Optuna run on transition features with objective `combined` and N1 weight `0.35`:

| Objective | Trials | Best trial | Accuracy | Balanced accuracy | Kappa | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Combined | 10 | 6 | 0.5230 | 0.5230 | 0.4038 | 0.5294 | 0.2882 | 0.3981 | 0.3344 |

Best combined-trial RF parameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 12 |
| `min_samples_leaf` | 2 |
| `min_samples_split` | 18 |
| `max_features` | `log2` |
| `max_samples` | 0.8 |
| N1 class weight | 1.75 |

The same 10-trial table also contained an even more N1-sensitive RF configuration:

| Trial | Accuracy | Balanced accuracy | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5155 | 0.5155 | 0.5239 | 0.2874 | 0.4356 | 0.3463 |

Short 8-trial Optuna run using `balanced_accuracy` as the objective:

| Objective | Trials | Best trial | Accuracy | Balanced accuracy | Kappa | Macro F1 | N1 precision | N1 recall | N1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced accuracy | 8 | 4 | 0.5365 | 0.5365 | 0.4206 | 0.5345 | 0.2983 | 0.2888 | 0.2934 |

Interpretation: Optuna did not beat the current best default Random Forest on balanced 5-class performance (`0.5458`). It did, however, find much better N1-sensitive Random Forest variants. Compared with the current transition-feature RF baseline, N1 F1 improved from `0.2459` to `0.3344` in the balanced/N1 combined setting, and up to `0.3463` in the most N1-sensitive observed trial. This makes tuned RF useful as a secondary sensitivity model, while the untuned transition-feature RF remains the best balanced default.

## Sources

- EEGDash `ds003768`: https://eegdash.org/api/dataset/eegdash.dataset.DS003768.html
- EEGDash `NM000185` / Sleep-EDF Expanded: https://eegdash.org/api/dataset/eegdash.dataset.NM000185.html
- EEGDash `ds006695`: https://eegdash.org/api/dataset/eegdash.dataset.DS006695.html
- EEGDash `ds004348`: https://eegdash.org/api/dataset/eegdash.dataset.DS004348.html
- EEGDash `ds005178`: https://eegdash.org/api/dataset/eegdash.dataset.DS005178.html
- EEGDash `ds004902`: https://eegdash.org/api/dataset/eegdash.dataset.DS004902.html
