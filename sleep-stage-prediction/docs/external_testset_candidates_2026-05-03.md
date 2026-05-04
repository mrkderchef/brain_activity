# External Testset Candidates

Research date: 2026-05-03

## Decision

The best next external benchmark is Sleep-EDF Expanded. It is large enough for a
real sleep-staging comparison, includes full-night PSG with EEG/EOG/EMG and
manual hypnograms, is directly used by many published sleep-stage models, and
is downloadable without an application gate. It should be treated as a new
benchmark/pretraining corpus, not simply merged into `ds006695`.

Recommended order:

1. Sleep-EDF Expanded / `sleep-edfx`
2. MASS SS3 or SS1, if access is practical
3. ISRUC-Sleep, if manual download/access is acceptable
4. Dreem Open Dataset, useful but has its own HDF5/tooling path
5. `ds005178` / EESM23, valuable for robustness but less comparable because it
   is ear-EEG

## Candidate Notes

### Sleep-EDF Expanded

Best first choice. PhysioNet lists 197 whole-night PSG recordings with EEG,
EOG, chin EMG, event markers, and manually scored hypnograms. The EEG channels
include Fpz-Cz and Pz-Oz, sampled at 100 Hz. Hypnograms include Wake, REM,
stage 1, stage 2, stage 3, stage 4, movement, and unknown. The total
uncompressed size is about 8.1 GB.

Use it for:

- an external benchmark where published numbers are easier to compare
- pretraining a raw/spectrogram sequence model
- checking whether our `cnn_gru` pipeline is truly weak or just limited by
  `ds006695`

Caveat: license is Open Data Commons Attribution, not CC0. Stage 3 and 4 should
be mapped to N3 for AASM-style 5-class evaluation.

### MASS

Strong PSG benchmark with multiple subsets. SS3 is attractive because it has 62
whole-night PSG recordings, 20 EEG electrodes, EOG, EMG, ECG, AASM sleep
staging, and 30-second pages. SS1 has 53 recordings and 17/19 EEG electrodes.

Use it for:

- a stronger lab PSG benchmark after Sleep-EDF
- cross-dataset validation against a richer EEG montage

Caveat: access may require contact/approval depending on subset and portal.

### ISRUC-Sleep

Good public PSG dataset with multiple groups: 100 subjects with one session, 8
subjects with two sessions, and 10 healthy subjects. It includes visual scoring
by two experts and PSG signals.

Use it for:

- robustness checks across healthy and patient sleep
- later validation once EDF/scoring import is generalized

Caveat: download and scoring formats need a separate audit before integration.

### Dreem Open Dataset

DOD-H and DOD-O are useful because they include healthy and obstructive sleep
apnea cohorts and multiple scorers. The public code documents HDF5 records with
hypnogram labels `WAKE`, `N1`, `N2`, `N3`, and `REM`.

Use it for:

- multi-scorer disagreement analysis
- a later benchmark if we want a non-EDF pipeline

Caveat: the project tooling is older and HDF5-based, so it is less direct than
Sleep-EDF for the current scripts.

### `ds005178` / EESM23

Large modern OpenNeuro/EEGDash dataset: 10 subjects, 140 recordings, about
1012.5 hours, BIDS format, CC0 license, 250 Hz sampling, and 4 or 13 ear-EEG
channels.

Use it for:

- representation learning or robustness to wearable EEG
- a separate ear-EEG benchmark

Caveat: it is not scalp PSG and may not be a fair direct testset for our current
forehead/scalp assumptions.

## Next Implementation Step

Add a Sleep-EDF importer that:

1. downloads a small subject subset first
2. reads `*-PSG.edf` and matching `*-Hypnogram.edf`
3. maps stages to `Wake`, `N1`, `N2`, `N3`, `REM`
4. creates the same spectrogram tensor format as `ds006695`
5. runs subject-wise Sleep-EDF CV before any transfer experiment

Only after that should we try:

- pretrain on Sleep-EDF, fine-tune on `ds006695`
- train on Sleep-EDF, test on `ds006695`
- train on `ds006695`, test on Sleep-EDF, mainly as a domain-shift diagnostic
