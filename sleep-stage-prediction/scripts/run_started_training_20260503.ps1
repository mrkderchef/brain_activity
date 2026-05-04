$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectDir

$OutputDir = Join-Path $ProjectDir "outputs\started_cnn_gru_r6_e16_20260503"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

python scripts\train_spectrogram_sequence_model.py `
  --spectrograms-path outputs\ds006695_spectrograms_all19\X_spectrograms.npy `
  --labels-path outputs\ds006695_spectrograms_all19\y_labels.npy `
  --metadata-path outputs\ds006695_spectrograms_all19\epoch_metadata.csv `
  --output-dir $OutputDir `
  --n-splits 5 `
  --model cnn_gru `
  --sequence-radius 6 `
  --epochs 16 `
  --batch-size 64 `
  --hidden-size 80 `
  --dropout 0.35 `
  --normalization channel `
  --label-smoothing 0.05 `
  --learning-rate 0.0007 `
  --weight-decay 0.0002 `
  --grad-clip-norm 1.0 `
  --early-stopping-patience 7 `
  --lr-plateau-patience 2 `
  --seed 44
