# Sleep-Staging SOTA Research

Research date: 2026-04-25

## Why the Current Score Feels Bad

The current best project model is:

| Dataset | Model | Features | Validation | Balanced accuracy | Macro F1 |
|---|---|---|---|---:|---:|
| `ds006695` all 19 subjects | Random Forest | Transition features, radii `2,4,6` | subject-wise 5-fold CV | 0.5465 | 0.5344 |

This is too low for a final practical sleep-stage classifier. However, it is not directly comparable to many headline `90%` sleep-staging numbers. Most high scores come from one or more of:

- Sleep-EDF or MASS, not the forehead-patch `ds006695` dataset
- raw EEG/EOG deep learning, not 21-273 hand-engineered summary features
- sequence models over minutes of context
- larger training sets
- sometimes easier splits, epoch-wise leakage, or fewer-class tasks
- sometimes accuracy rather than macro F1/balanced accuracy

The goal should be to move from hand-engineered tabular features toward raw/spectrogram sequence modeling, while keeping subject-wise validation.

## What Others Use

### DeepSleepNet

Official GitHub:

- https://github.com/akaraspt/deepsleepnet

DeepSleepNet uses CNN layers on raw single-channel EEG and a bidirectional LSTM to learn sleep-stage transition rules. Reported performance:

| Dataset | Accuracy | Macro F1 |
|---|---:|---:|
| MASS | 86.2% | 81.7% |
| Sleep-EDF | 82.0% | 76.9% |

Interpretation: a serious deep raw-signal sequence model lands around low-to-mid 80s accuracy on common benchmarks, not automatically 90+ under strict evaluation.

### TinySleepNet

Official GitHub:

- https://github.com/akaraspt/tinysleepnet

TinySleepNet is a smaller raw single-channel EEG model related to DeepSleepNet. It is designed as an efficient baseline and uses Sleep-EDF preparation/training scripts. The important takeaway for this project is architectural, not just numeric: it trains directly from raw 30-second epochs and learns temporal context, rather than compressing each epoch into a few band-power summaries.

### AttnSleep

Official GitHub:

- https://github.com/emadeldeen24/AttnSleep

AttnSleep uses:

- multi-resolution CNN feature extraction
- adaptive feature recalibration
- temporal context encoder
- multi-head attention

It evaluates on Sleep-EDF-20, Sleep-EDF-78, and SHHS. This is a good model family to copy conceptually if we want a modern raw/spectrogram sequence learner.

### DeepSleepNet-Lite

DeepSleepNet-Lite is a simplified 90-second EEG input model. Reported Sleep-EDF expanded performance:

| Dataset version | Accuracy | Macro F1 | Kappa |
|---|---:|---:|---:|
| Sleep-EDF v1 +/-30min | 84.0% | 78.0% | 0.78 |
| Sleep-EDF v2 +/-30min | 80.3% | 75.2% | 0.73 |

With uncertainty rejection, reported performance rises to about 86.1% / 79.6% on v1 and 82.3% / 76.7% on v2, but this means rejecting uncertain predictions rather than improving all-epoch performance.

### U-Sleep

U-Sleep is a large fully convolutional model trained/evaluated across many PSG datasets. Reported global F1 across 21 datasets:

| Stage | F1 |
|---|---:|
| Wake | 0.90 |
| N1 | 0.53 |
| N2 | 0.85 |
| N3 | 0.76 |
| REM | 0.90 |
| Mean across stages | 0.79 |

Interpretation: even a very strong, broad deep-learning system still has N1 as the weak class. So expecting N1 near 90 is not realistic; expecting overall accuracy near 90 may be possible only with strong raw PSG modeling and favorable data.

### SSC-EOG

GitHub:

- https://github.com/suvadeepmaiti/SSC-EOG

Reported linear evaluation:

| Dataset | Accuracy | Kappa | Macro F1 |
|---|---:|---:|---:|
| Sleep-EDF-20 | 79.3 | 0.72 | 74.7 |
| Sleep-EDF-153 | 73.6 | 0.68 | 70.6 |
| SHHS | 79.0 | 0.71 | 69.3 |

Interpretation: even recent GitHub projects with representation learning report high-70s accuracy/macro-F1 around 70 on larger settings, not 90.

## About 90% Accuracy

Some papers report near or above 90%:

- a 1D-CNN PSG model reports around 90-91% for five/six classes on Sleep-EDF/Sleep-EDFx
- spectrogram CNN+BiLSTM papers report high 80s for 5-class Sleep-EDF variants
- some hand-crafted or deep models report 86-90% on Sleep-EDF

But these results usually differ from our current setup in major ways:

1. They use Sleep-EDF/EDF Expanded, not `ds006695`.
2. They use raw signal or spectrograms, not only band-power summary rows.
3. They often use EEG plus EOG/EMG or PSG-style channels.
4. They use many more recordings/epochs.
5. Their reported metric is often accuracy, not balanced accuracy.
6. N1 remains low even in strong models.

So the project should not promise `90%` from the current tabular RF setup. To get anywhere close, we need a different modeling level.

## Recommended Next Direction

### 1. Build A Raw/Spectrogram Sequence Baseline

Current features collapse each 30-second epoch into a small number of summary statistics. That likely throws away the morphology needed to separate N1, REM, and N2.

Next model family:

- input: 30-second raw EEG or log-spectrogram
- context: previous/current/next epochs, or 5-15 epoch windows
- architecture: small CNN + GRU/LSTM/Transformer/TCN
- output: 5-class stage prediction for the center epoch

Start simple:

- single `ds006695` channel or channel-average representation
- log-mel/log-STFT spectrograms
- CNN encoder + GRU or temporal convolution
- subject-wise CV

### 2. Use Sleep-EDF For Transfer Learning

The papers and repos are mostly built around Sleep-EDF. A practical way to improve:

1. Add Sleep-EDF / `NM000185` as a source dataset.
2. Pretrain a raw/spectrogram model on Sleep-EDF.
3. Fine-tune or evaluate on `ds006695`.
4. Compare subject-wise `ds006695` performance against the current RF baseline.

This is more plausible than trying to push Random Forest from 0.55 to 0.90.

### 3. Add EOG/Multimodal Data If Available

High sleep-staging scores often use PSG context. If `ds006695` has only forehead EEG, the ceiling may be lower. If another dataset gives EEG+EOG/EMG, use it for pretraining and benchmarking.

### 4. Keep Evaluation Honest

Avoid epoch-wise random splits for headline results. They can inflate performance because neighboring epochs from the same subject leak into train and test. Continue using subject-wise splits for thesis claims.

## Immediate Implementation Plan

Implemented first pass:

1. Added spectrogram extraction script for `ds006695`:
   - `scripts/extract_ds006695_spectrograms.py`
   - output shape on all 19 subjects: `(19563, 3, 20, 116)`
2. Added a PyTorch CNN-GRU sequence baseline:
   - `scripts/train_spectrogram_sequence_model.py`
   - sequence radius `2`, i.e. 5 epochs of context
   - predicts the center epoch
   - class-weighted cross-entropy
3. First CPU smoke run:
   - 5-fold subject-wise split definition
   - evaluated first 2 folds only
   - 5 epochs per fold

| Model | Input | Evaluation | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|
| Random Forest baseline | transition tabular features | full 5-fold subject-wise CV | 0.5465 | 0.5344 | 0.2474 |
| CNN-GRU smoke | log-spectrogram sequence windows | first 2/5 subject-wise folds | 0.6314 | 0.6270 | 0.2680 |

Interpretation: the spectrogram sequence approach immediately beats the tabular Random Forest baseline on the evaluated folds, even with a small CPU-trained model. This strongly supports moving the project toward raw/spectrogram deep learning. The result is not yet the final headline number because only 2 of 5 folds were run, but it is the first clear evidence that the RF ceiling was mostly a feature/model limitation.

Next implementation steps:

1. Run the CNN-GRU on all 5 folds.
2. Increase training from 5 to 8-12 epochs if validation remains stable.
3. Try sequence radius `4` for 9-epoch context.
4. Add checkpointing and per-fold saved models.
5. Consider a CNN-TCN variant if CPU training remains manageable.

## Sources

- DeepSleepNet GitHub: https://github.com/akaraspt/deepsleepnet
- TinySleepNet GitHub: https://github.com/akaraspt/tinysleepnet
- AttnSleep GitHub: https://github.com/emadeldeen24/AttnSleep
- SSC-EOG GitHub: https://github.com/suvadeepmaiti/SSC-EOG
- DeepSleepNet abstract/performance: https://pubmed.ncbi.nlm.nih.gov/28678710/
- DeepSleepNet-Lite abstract/performance: https://pubmed.ncbi.nlm.nih.gov/34648450/
- U-Sleep paper: https://www.nature.com/articles/s41746-021-00440-5
- EEGSNet / spectrogram CNN+BiLSTM: https://pubmed.ncbi.nlm.nih.gov/35627856/
- 1D-CNN PSG sleep-stage model: https://pmc.ncbi.nlm.nih.gov/articles/PMC6406978/
