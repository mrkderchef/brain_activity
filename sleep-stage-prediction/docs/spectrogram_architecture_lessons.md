# Spectrogram Architecture Lessons

Research date: 2026-04-26

## Why The N1/Subject-Shift Problem Is Expected

The current project findings match the broader sleep-staging literature:

- N1 is consistently the weakest stage, even in strong cross-dataset systems.
- Stage proportions and transition structure matter, especially when N2 dominates.
- Subject-wise generalization is harder than random epoch-wise validation.
- Simple class weighting can raise N1 recall while lowering precision and macro
  performance, which is exactly what the local focal/N1-weighted runs showed.

## Patterns From Related Architectures

### U-Sleep

Useful ideas:

- robust per-subject, per-channel scaling with median/IQR
- clipping extreme amplitudes after robust scaling
- long receptive field, with predictions informed by minutes of context
- class-balanced segment sampling rather than only class-weighted loss
- noise replacement and channel dropout style augmentation
- training across heterogeneous datasets and channel combinations to force
  invariance
- per-subject stage-wise reporting, not only pooled epoch metrics

Project implication:

The next local experiment should prioritize robust subject/channel normalization
and training-time augmentation over more N1 loss multipliers.

### DeepSleepNet / TinySleepNet

Useful ideas:

- raw or near-raw signal modeling with CNN feature extraction
- explicit temporal modeling with bidirectional recurrent layers
- learning transition rules from sequences instead of treating epochs as
  independent rows

Project implication:

The current CNN-GRU spectrogram model is aligned with this family, but its
context is still shorter and simpler than many modern systems. Longer context or
a full-sequence model is more plausible than more tabular feature engineering.

### AttnSleep / CAttSleepNet

Useful ideas:

- multi-resolution CNN feature extraction
- attention or feature recalibration to emphasize locally useful EEG features
- temporal context encoder/attention for sequence dependencies

Project implication:

A good next architecture is a spectrogram encoder with multi-scale convolution
branches, followed by GRU or lightweight attention. This directly targets the
project's N2/N3/REM confusions, where both spectral details and context matter.

### SSC-SleepNet And Adaptive Losses

Useful ideas:

- N1 is treated as both rare and ambiguous, not merely underweighted
- adaptive loss combines weighted cross-entropy and focal-like behavior
- pseudo-Siamese or contrastive representations are used to improve difficult
  class separation

Project implication:

The local focal loss run was too blunt. If revisiting N1 optimization, use a
dynamic/adaptive schedule or contrastive representation objective, not a fixed
high N1 multiplier.

### Transition-Rule Models

Useful ideas:

- combine signal representation learning with an explicit transition-rule
  module
- model plausible stage sequences instead of only center-epoch labels
- EEG+EOG generally improves transition-sensitive staging when available

Project implication:

For this EEG-only dataset, a lightweight post-processing or auxiliary transition
head may help reduce impossible/unstable local transitions. This is especially
relevant because subject-level failures are tied to N2-heavy architecture and
N2/N3/REM confusion.

## Recommended Next Experiments

1. Add robust spectrogram normalization:
   - compute train-fold median/IQR per subject or recording/channel
   - scale spectrograms with robust statistics
   - clip extreme normalized values

2. Add training-time spectrogram augmentation:
   - random time masking
   - random frequency masking
   - low-probability channel masking
   - small Gaussian noise

3. Add class-balanced sampling:
   - sample center epochs by class during training
   - keep test evaluation unchanged
   - compare against fixed class-weighted loss

4. Try longer context:
   - radius `6` or `8`
   - keep the same CNN-GRU first
   - judge by subject-wise balanced accuracy and N1 F1

5. Try multi-scale spectrogram encoder:
   - parallel kernels such as `(3,3)`, `(5,3)`, and frequency-oriented kernels
   - concatenate pooled features before GRU

6. Add transition-aware post-processing or auxiliary analysis:
   - estimate transition matrices from training subjects
   - apply simple HMM/Viterbi smoothing as a sensitivity analysis
   - report whether N2/N3/REM confusions decrease

## Priority For This Project

Do next:

1. robust normalization + clipping
2. class-balanced center sampling
3. spectrogram masking augmentation
4. longer context

Defer:

- heavier N1 class weights
- fixed focal loss
- large transformer rewrite
- GAN augmentation

The local experiments already suggest that fixed N1 weighting does not solve the
real failure mode. The literature points toward robustness and sequence
structure first.
