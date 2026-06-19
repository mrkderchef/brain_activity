# Current Data Audit

This note records the current state of the dataset and model inputs as of 2026-04-24. It is intended as a thesis-friendly working log.

## Scope

- Dataset: `ds003768`
- Project outputs reviewed: `outputs/X_features.npy`, `outputs/y_labels.npy`, `outputs/checkpoint.npz`, `outputs/metrics.json`
- Goal of this audit: explain why REM is missing, why the class distribution is highly imbalanced, and why the saved extracted feature matrix is smaller than the clean label pool

## Main Findings

1. REM is not missing because of a bug in the feature extraction code.
2. REM is absent in the source scoring files themselves.
3. The dataset README only describes `W`, `1`, `2`, and `3` as valid scored stages for this dataset.
4. The class imbalance is largely a property of the dataset, not only of the model.
5. The extra loss from `6858` clean labels down to `6773` extracted rows is most likely caused by the post-extraction finite-value filter, not by session matching.

## Evidence

### 1. Source labels do not include REM

The `ds003768/README` states:

- `w` represents wakefulness
- `1, 2, 3` represent NREM1, NREM2, and NREM3

No REM label is described there.

The aggregated source TSV counts confirm this:

| Stage string | Count |
|---|---:|
| `W` | 3343 |
| `1` | 2148 |
| `2` | 1325 |
| `3` | 42 |

There were no `R` labels in the source sleep-stage TSV files.

### 2. Some scored rows are explicitly uncertain or unscorable

The raw label tables contain `6990` rows, but only `6858` are clean labels after excluding uncertain/unscorable rows.

Excluded rows:

| Label string | Count |
|---|---:|
| `W (uncertain)` | 43 |
| `1 (uncertain)` | 42 |
| `2 (uncertain)` | 12 |
| `3 (unscorable)` | 11 |
| `2 (unscorable)` | 10 |
| `1 (unscorable)` | 6 |
| `2 or 3 (unscorable)` | 3 |
| `unscorable` | 2 |
| `Unscorable` | 1 |
| `3 (uncertain)` | 1 |
| `nan` | 1 |

Two EEG recordings have no clean labels at all:

- `sub-23_task-rest_run-1`
- `sub-23_task-sleep_run-2`

Those two sessions are entirely uncertain or unscorable, so dropping them is expected.

### 3. The class imbalance is severe in the clean label pool

Clean label distribution (`6858` epochs total):

| Stage | Count | Percent |
|---|---:|---:|
| Wake | 3343 | 48.75% |
| N1 | 2148 | 31.32% |
| N2 | 1325 | 19.32% |
| N3 | 42 | 0.61% |

Important consequence:

- N3 is extremely rare
- REM is entirely absent
- A 4-stage problem is already highly imbalanced before model training begins

N3 appears in only 4 subjects:

- subject 4
- subject 16
- subject 23
- subject 27

29 of 33 subjects have no N3 at all.

### 4. The saved extracted arrays are smaller than the clean label pool

Current saved feature outputs:

- `X_features.npy`: `(6773, 15)`
- `y_labels.npy`: `(6773,)`

Current saved label counts:

| Stage int | Meaning | Count |
|---|---|---:|
| `0` | Wake | 3341 |
| `1` | N1 | 2129 |
| `2` | N2 | 1261 |
| `3` | N3 | 42 |

Difference versus clean labels:

| Stage | Clean labels | Saved output | Delta |
|---|---:|---:|---:|
| Wake | 3343 | 3341 | -2 |
| N1 | 2148 | 2129 | -19 |
| N2 | 1325 | 1261 | -64 |
| N3 | 42 | 42 | 0 |

Total difference:

- clean labels: `6858`
- saved output rows: `6773`
- missing after extraction: `85`

### 5. Recording length explains part of the mismatch

A fast audit based on BrainVision header metadata and `.eeg` file sizes showed:

- recordings with clean labels: `253`
- sum of clean labeled epochs across those recordings: `6858`
- sum of available 30-second signal windows across those recordings: `7111`

This means the raw recordings are, in aggregate, long enough to cover the clean label tables. However, the aggregate view hides some recording-specific truncation.

Four `sub-16` sleep recordings are shorter than their clean label tables:

| Recording | Clean labels | Signal epochs | Missing tail epochs |
|---|---:|---:|---:|
| `sub-16_task-sleep_run-2` | 30 | 20 | 10 |
| `sub-16_task-sleep_run-3` | 30 | 21 | 9 |
| `sub-16_task-sleep_run-4` | 30 | 17 | 13 |
| `sub-16_task-sleep_run-5` | 30 | 17 | 13 |

These four truncated recordings explain `45` of the `85` missing rows exactly.

The class loss from those truncated tails is:

| Stage | Missing from truncation |
|---|---:|
| Wake | 2 |
| N1 | 8 |
| N2 | 35 |
| N3 | 0 |

Because the extraction pipeline ends with:

```python
mask = np.all(np.isfinite(X), axis=1)
X, y = X[mask], y[mask]
```

After subtracting the `45` truncation-driven losses from the total mismatch, there is still a residual unexplained loss of `40` epochs:

| Stage | Residual unexplained loss |
|---|---:|
| Wake | 0 |
| N1 | 11 |
| N2 | 29 |
| N3 | 0 |

The most plausible explanations for this residual are:

- some extracted epochs produced non-finite features and were removed silently by the finite-value filter
- or a small number of recordings/segments failed during the older extraction run that produced the current `X_features.npy` and `y_labels.npy`

One exact decomposition of the residual `40` epochs exists as whole-recording omission:

- `sub-16_task-rest_run-1` contributes `N1: 3`, `N2: 17`
- `sub-24_task-rest_run-2` contributes `N1: 8`, `N2: 12`

Together these two recordings match the residual `N1: 11`, `N2: 29`.

This is not yet proven, but it is a concrete hypothesis consistent with the saved output counts.

## Current Model Status

A local 3-fold training run on the saved `.npy` data produced:

- Accuracy: `0.6892`
- Cohen's kappa: `0.4977`

Interpretation:

- the current model is learning signal above chance
- however, evaluation is constrained by the label space itself
- the dataset does not support a REM classifier in its current form
- N3 performance estimates are unstable because there are only `42` N3 epochs total

## Likely Causes

### Missing REM

Most likely cause:

- dataset property, not extraction bug

Supporting evidence:

- README documents only `W`, `1`, `2`, `3`
- no `R` labels were found in the source TSV files
- short scanner sleep sessions also make REM biologically less likely

### Severe class imbalance

Most likely causes:

- experimental design includes resting-state wakefulness and short sleep runs
- N3 is rare in the provided scored epochs
- REM is not part of the released scoring labels

### Missing 85 extracted rows

Current best explanation:

- `45` rows are lost because four `sub-16` recordings are shorter than their label tables
- the remaining `40` rows were likely lost either through non-finite feature filtering or through one or more recording-level failures in the older extraction run

Still not fully proven:

- which exact recordings/epochs account for the final residual `40`
- whether that residual comes from non-finite feature rows, a failed recording load, or another bookkeeping issue in the previous extraction run

## Thesis-Safe Conclusions

The current modeling problem should be described as:

- a 4-class problem on `Wake`, `N1`, `N2`, and `N3`
- not a 5-class problem including REM

The main data limitations are:

- no REM labels in the dataset release used here
- extreme rarity of N3
- exclusion of uncertain/unscorable epochs
- an additional 85-epoch loss during feature cleaning

## Recommended Next Steps

1. Add per-recording extraction diagnostics so every dropped epoch is logged with subject, session, and class.
2. Re-run feature extraction with that logging enabled to identify the exact source of the 85 missing rows.
3. Reframe the current task and thesis text as a 4-class problem unless a different dataset with REM is added.
4. For accuracy work, prioritize class-imbalance handling:
   - subject-aware cross-validation
   - class-weighted models
   - balanced resampling for N1/N2/N3
   - metrics beyond plain accuracy
5. Treat N3 results cautiously because only 42 epochs are available.

## Reproducibility

Use the audit script:

```bash
python scripts/audit_dataset.py --bids-root ..\ds003768
```

This reproduces the label counts, excluded-label counts, and extracted-output comparison used in this note.
