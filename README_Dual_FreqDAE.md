# Dual_FreqDAE Denoising Pipeline

This document provides a focused guide for preparing data, training, and running inference with the `Dual_FreqDAE` ECG denoising model in this repository.

## 1. Overview
`Dual_FreqDAE` is a dual-branch denoising autoencoder for ECG beats combining:
- Time-domain gated convolutional encoder
- Frequency-domain encoder (FFT magnitude branch)
- Concatenated latent + positional encoding
- FAN-based transformer blocks (`FANformer_encoder`)
- Residual Conv1DTranspose decoder

Input length is fixed at **512 samples** (derived from QTDatabase beats resampled to 360 Hz). The model expects two inputs:
1. Noisy time-domain beats: shape `(N, 512, 1)`
2. Frequency-domain magnitude (FFT duplicated to 512): shape `(N, 512, 1)`

Output: Denoised time-domain beats `(N, 512, 1)`.

## 2. Environment
Create the conda environment (or activate existing `ecgdenoise`):
```bash
conda env create -f environment.yml  # first time
conda activate ecgdenoise
```
Ensure `tensorflow` and `scipy` are installed (they are in `environment.yml`).

## 3. Data Preparation
You must build the QTDatabase and combined noise pickles before training.

### 3.1. QT Database
Download the PhysioNet QT database and unpack it under `data/qt-database-1.0.0/`.
Run:
```bash
python Data_Preparation/Prepare_QTDatabase.py
```
This creates `data/QTDatabase.pkl` containing beat-separated signals resampled to 360 Hz.

### 3.2. Noise Datasets
Expected pickle files (already present if you used the original pipeline):
- `data/CombinedNoise_Train.pkl`
- `data/CombinedNoise_Test.pkl`

If missing, regenerate according to your original procedure (not included here). These are used to synthesize noisy beats.

### 3.3. Dual Branch Dataset
To produce both time-domain and frequency-domain arrays, the training script internally calls:
`Data_Preparation/data_preparation_with_fourier.py`
which returns:
```
[X_train, y_train, X_test, y_test, F_train_x, F_train_y, F_test_x, F_test_y]
shapes: (N, 512, 1) each
```
Frequency features are FFT magnitudes duplicated to length 512.

## 4. Training
Use the dedicated script:
```bash
python tools/train_dual_freqdae.py \
  --exp-dir experiments/dual_freqdae_run1 \
  --data-prep-samples 512 \
  --epochs 100000 \
  --patience 10 \
  --min-delta 0.05 \
  --reuse-cache
```
Flags:
- `--reuse-cache`: Reuses a compressed `.npz` dataset once built.
- `--resume`: Loads existing best weights to continue training.
- `--evaluate-only`: Skips training, only runs metrics.
- `--dry-run`: Builds dataset (if needed) and model, prints summary, then exits.

Artifacts inside `exp-dir`:
- `Dual_FreqDAE_weights.best.weights.h5` (best val_loss)
- `history.json` (loss curves)
- `metrics_summary.json` (aggregate)
- `metrics_per_sample.csv` (per-beat metrics)

Early stopping, LR reduction, and optimizer settings replicate the original configuration in `deepFilter/dl_pipeline.py`.

## 5. Evaluation / Metrics
Automatically computed after training (updated set):
- RMSE (Root Mean Squared Error per beat)
- PRD (%) Percentage Root-mean-square Difference
- COS_SIM (Cosine similarity between clean and denoised beat)
- SNR (dB) Signal-to-Noise Ratio improvement measure

Access aggregate summary:
```bash
cat experiments/dual_freqdae_run1/metrics_summary.json
```

## 6. Inference
Use the standalone runner for arbitrary noisy beats:
```bash
python tools/run_dual_freqdae.py \
  --input path/to/noisy_beats.npy \
  --weights experiments/dual_freqdae_run1/Dual_FreqDAE_weights.best.weights.h5 \
  --output denoised.npy
```
If your file is `(512,)` or `(N,512)` or `(N,512,1)`, it will be normalized internally. The script rebuilds the frequency branch automatically.

Large continuous 1D signal (e.g., full recording) segmentation:
```bash
python tools/run_dual_freqdae.py \
  --input long_signal.npy \
  --segment --hop 512 --pad \
  --weights experiments/dual_freqdae_run1/Dual_FreqDAE_weights.best.weights.h5 \
  --output denoised_segments.npy
```
This will slice your 1D array into sequential 512-sample windows (zero-padding the tail if needed).

Demo mode:
```bash
python tools/run_dual_freqdae.py --demo --demo-batch 8 \
  --weights 0221_FIXED/Dual_FreqDAE_weights.best.weights.h5
```

Tip: run long trainings inside tmux (session `ecgdenoise`) and check the window `dual_train`.

## 7. Reproducibility Notes
- Random seed control for dataset generation comes from the original preparation code (`np.random.seed(1234)`).
- Ensure consistent QTDatabase version and noise pickle sources to reproduce published metrics.
- Training length: Although epochs are set to `100000`, early stopping typically halts far earlier.

## 8. Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| `No module named 'deepFilter'` | Running script from wrong cwd | `cd /home/.../ecg_denoise` before running |
| Shape mismatch (not 512) | Input beats not cropped/padded | Preprocess to length 512 or adapt model code |
| Slow first run | Dataset Fourier computation | Use `--reuse-cache` on subsequent runs |

## 9. Extending
- Adjust transformer depth: change `num_transformer_blocks` in `Dual_FreqDAE` constructor.
- Try gating variants: modify `AddGatedNoise` or swap to `ParametricNoiseInjection` lines in `dl_models.py`.
- Add quantization/pruning: integrate TensorFlow Model Optimization Toolkit after training.

## 10. License & Attribution
Original architecture scripts reside in `deepFilter/dl_models.py`. Metrics code in `utils/metrics.py`. Data preparation adapted from existing PhysioNet-based scripts.

---
For questions or further automation (batch experiments, parameter sweeps), you can create an additional driver script extending `tools/train_dual_freqdae.py`.
