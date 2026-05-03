# Accuracy Marathon Summary

This note records the completed unattended spectrogram experiment batch from
`outputs/accuracy_marathon`. Generated outputs remain ignored by git, so this
file captures the durable result summary.

## Best Result

| Experiment | Variant | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---:|---:|---:|---:|
| `exp_03_gru_r6_long_context` | Viterbi `w=0.5` | `0.6628` | `0.6222` | `0.6084` | `0.2736` |

Best raw model:

| Experiment | Model | Context | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|---:|
| `exp_03_gru_r6_long_context` | CNN-GRU | 13 epochs | `0.6501` | `0.6187` | `0.6023` | `0.2839` |

## Raw Model Ranking

| Rank | Experiment | Model | Key idea | Balanced accuracy | Accuracy | Macro F1 | N1 F1 |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `exp_03_gru_r6_long_context` | CNN-GRU | 13-epoch context | `0.6187` | `0.6501` | `0.6023` | `0.2839` |
| 2 | `exp_04_attention_r6_long_context` | CNN-GRU attention | 13-epoch context + attention | `0.6169` | `0.6542` | `0.5976` | `0.2419` |
| 3 | `exp_01_gru_r4_e18_smooth` | CNN-GRU | 9-epoch context, longer regularized run | `0.6095` | `0.6515` | `0.5980` | `0.2619` |
| 4 | `exp_06_attention_aug_r4` | CNN-GRU attention | 9-epoch context + light augmentation | `0.6028` | `0.6328` | `0.5837` | `0.2326` |
| 5 | `exp_05_gru_aug_r4` | CNN-GRU | 9-epoch context + light augmentation | `0.6021` | `0.6417` | `0.5840` | `0.2358` |
| 6 | `exp_02_attention_r4_e18` | CNN-GRU attention | 9-epoch context + attention | `0.6005` | `0.6376` | `0.5825` | `0.2036` |
| 7 | `exp_08_gru_n1_weight` | CNN-GRU | small N1 class-weight bump | `0.5998` | `0.6291` | `0.5868` | `0.2716` |
| 8 | `exp_09_seq2seq_gru_b32_s16_e24` | CNN-GRU Seq2Seq | 32-epoch blocks | `0.5954` | `0.6350` | `0.5791` | `0.2238` |
| 9 | `exp_07_gru_balanced_sampler` | CNN-GRU | balanced sampler + N1 emphasis | `0.5880` | `0.5644` | `0.5497` | `0.2310` |
| 10 | `exp_10_seq2seq_transformer_b32_e48` | CNN-Transformer Seq2Seq | 32-epoch blocks | `0.5612` | `0.5930` | `0.5355` | `0.1633` |

## Interpretation

The main gain came from expanding the local spectrogram context from 9 epochs
to 13 epochs. Attention pooling was competitive at the same context length, but
did not beat the simpler CNN-GRU. Light augmentation, balanced sampling, and a
small N1 weight bump did not improve the main balanced-accuracy objective.

Fold-aware Viterbi smoothing gave the best postprocessed number. It should be
reported separately from the raw model because its transition weight was chosen
after a sweep on the cross-validation predictions.

## SE/Residual Follow-Up

After the marathon, a stronger `cnn_gru_se` candidate was added and tested as a
standalone 5-fold subject-wise run. It used 13-epoch spectrogram windows,
multi-scale convolution branches, residual squeeze-excitation blocks, BiGRU
temporal context, attention pooling, robust channel normalization, and clipping.

| Experiment | Model | Context | Accuracy | Balanced accuracy | Macro F1 | N1 F1 |
|---|---|---|---:|---:|---:|---:|
| `ds006695_spectrograms_all19_cnn_gru_se_r6_e18_full5` | CNN-GRU-SE | 13 epochs | `0.5960` | `0.5687` | `0.5577` | `0.2495` |

Fold-level results:

| Fold | Accuracy | Balanced accuracy | Macro F1 | N1 F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.6372` | `0.6278` | `0.6001` | `0.2795` | 11 |
| 2 | `0.6560` | `0.5846` | `0.5929` | `0.2342` | 4 |
| 3 | `0.5984` | `0.5636` | `0.5476` | `0.1488` | 3 |
| 4 | `0.5713` | `0.5236` | `0.5098` | `0.2709` | 6 |
| 5 | `0.5210` | `0.5581` | `0.5072` | `0.2729` | 2 |

Decision: keep the simpler 13-epoch CNN-GRU as the best raw model. The
SE/residual encoder was inspired by stronger public architectures, but on this
small three-channel forehead EEG benchmark it overfit the inner validation
subjects and transferred worse to held-out subjects. This is a useful negative
result: more convolutional capacity is not the next bottleneck.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_accuracy_experiments.ps1
```

```bash
python scripts\run_accuracy_experiments.py --root-dir outputs\accuracy_marathon
python scripts\summarize_accuracy_experiments.py --root-dir outputs\accuracy_marathon
```
