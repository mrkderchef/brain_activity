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

## Sources

- EEGDash `ds003768`: https://eegdash.org/api/dataset/eegdash.dataset.DS003768.html
- EEGDash `NM000185` / Sleep-EDF Expanded: https://eegdash.org/api/dataset/eegdash.dataset.NM000185.html
- EEGDash `ds006695`: https://eegdash.org/api/dataset/eegdash.dataset.DS006695.html
- EEGDash `ds004348`: https://eegdash.org/api/dataset/eegdash.dataset.DS004348.html
- EEGDash `ds005178`: https://eegdash.org/api/dataset/eegdash.dataset.DS005178.html
- EEGDash `ds004902`: https://eegdash.org/api/dataset/eegdash.dataset.DS004902.html
